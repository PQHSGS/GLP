import torch
from tqdm import tqdm

from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)


def canonicalize_solver(solver):
    if solver is None:
        return "euler"

    solver = str(solver).strip().lower().replace("-", "_")
    aliases = {
        "default": "euler",
        "flowmatch_euler": "euler",
        "flowmatch_heun": "heun",
        "dpm_solver": "dpm",
        "dpmsolver": "dpm",
        "dpmsolver++": "dpm",
        "dpm_solver++": "dpm",
    }
    solver = aliases.get(solver, solver)

    if solver in {"euler", "heun", "dpm"}:
        return solver

    raise ValueError(f"Unsupported solver '{solver}'. Expected one of ['euler', 'heun', 'dpm'].")


def build_inference_scheduler(solver="euler", num_train_timesteps=1000):
    solver = canonicalize_solver(solver)

    if solver == "euler":
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    if solver == "heun":
        return FlowMatchHeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)

    return DPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        solver_order=2,
        prediction_type="flow_prediction",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        use_flow_sigmas=True,
        flow_shift=1.0,
        timestep_spacing="linspace",
        final_sigmas_type="zero",
    )


def get_inference_scheduler(model, solver=None):
    if solver is not None:
        scheduler = getattr(model, "scheduler", None)
        num_train_timesteps = getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000)
        return build_inference_scheduler(solver, num_train_timesteps=num_train_timesteps)

    scheduler = getattr(model, "inference_scheduler", None)
    if scheduler is not None:
        return scheduler

    scheduler = getattr(model, "scheduler", None)
    if scheduler is not None:
        return scheduler

    return build_inference_scheduler()


def _scale_model_input(scheduler, latents, timestep):
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latents, timestep)
    return latents


def _move_scheduler_state_to_device(scheduler, device):
    if hasattr(scheduler, "timesteps") and hasattr(scheduler.timesteps, "to"):
        scheduler.timesteps = scheduler.timesteps.to(device)
    if hasattr(scheduler, "sigmas") and hasattr(scheduler.sigmas, "to"):
        scheduler.sigmas = scheduler.sigmas.to(device)

# ==========================
#  Flow Matching Functions
# ==========================
def fm_scheduler():
    return FlowMatchEulerDiscreteScheduler()

def fm_prepare(scheduler, model_input, noise, u=None, generator=None):
    """
    Prepare inputs for flow matching training.
    Reference: https://github.com/huggingface/diffusers/blob/9f48394bf7ab75a43435d3ebb96649665e09c98b/examples/dreambooth/train_dreambooth_lora_flux.py#L1736
    """
    # sanity check
    assert isinstance(
        scheduler,
        (FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler),
    ), "Only flow-match schedulers are supported"
    assert model_input.ndim == 3, f"Expected (batch, seq, dim), got shape {model_input.shape}"
    # use uniform weighting_scheme; doesn't implement logit_normal
    if u is None:
        batch_size = model_input.shape[0]
        u = torch.rand(size=(batch_size,), generator=generator)
    indices = (u * len(scheduler.timesteps)).long()
    indices_ts = indices.to(scheduler.timesteps.device)
    timesteps = scheduler.timesteps[indices_ts]
    indices_sigmas = indices.to(scheduler.sigmas.device)
    sigmas = scheduler.sigmas[indices_sigmas].flatten()
    timesteps = timesteps.to(model_input.device)
    sigmas = sigmas.to(model_input.device)
    timesteps = timesteps[:, None, None]
    sigmas = sigmas[:, None, None]
    # interpolate between model_input and noise
    noisy_model_input = (1.0 - sigmas) * model_input.to(sigmas.dtype) + sigmas * noise
    noisy_model_input = noisy_model_input.to(model_input.dtype)
    # the target in flow matching is the "velocity"
    target = noise - model_input
    return noisy_model_input, target, timesteps, {"sigmas": sigmas, "noise": noise, "u": u}

def fm_clean_estimate(scheduler, latents, noise_pred, timesteps):
    assert isinstance(
        scheduler,
        (FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler),
    ), "Only flow-match schedulers are supported"
    step_indices = [(scheduler.timesteps == t).nonzero().item() for t in timesteps]
    sigma = scheduler.sigmas[step_indices]
    sigma = sigma.to(device=latents.device, dtype=latents.dtype)
    pred_x0 = latents - sigma * noise_pred
    return pred_x0

# ==========================
#   Generic Sampling Code
# ==========================
@torch.no_grad()
def sample(
    model,
    latents,
    num_timesteps=20,
    show_progress=True,
    solver=None,
    **kwargs
):
    """
    Generate activations from pure noise.
    We recommend setting `num_timesteps` based on your priorities:
    - 20: moderate quality at fast speed
    - 100: good quality at reasonable speed
    - 1000: best quality for diffusion purists
    """
    scheduler = get_inference_scheduler(model, solver=solver)
    scheduler.set_timesteps(num_timesteps)
    _move_scheduler_state_to_device(scheduler, latents.device)
    
    _model_dtype = next(model.denoiser.parameters()).dtype
    latents = latents.to(dtype=_model_dtype)

    for i, timestep in tqdm(enumerate(scheduler.timesteps), disable=not show_progress):
        timesteps = timestep.repeat(latents.shape[0], 1)
        model_input = _scale_model_input(scheduler, latents, timestep)
        noise_pred = model.denoiser(
            latents=model_input,
            timesteps=timesteps,
            **kwargs
        )
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents

@torch.no_grad()
def sample_on_manifold(
    model, 
    latents, 
    num_timesteps=20, 
    start_timestep=None,
    show_progress=True,
    solver=None,
    **kwargs
):
    """
    Post-process activations into their on-manifold counterpart.
    See the `sample` function above for recommendations on `num_timesteps`.
    This is essentially the activation-space analogue of SDEdit (Meng et. al., 2022).
    """
    start_latents = latents.clone()
    scheduler = get_inference_scheduler(model, solver=solver)
    scheduler.set_timesteps(num_timesteps)
    _move_scheduler_state_to_device(scheduler, latents.device)
    
    _model_dtype = next(model.denoiser.parameters()).dtype
    latents = latents.to(dtype=_model_dtype)
    start_latents = start_latents.to(dtype=_model_dtype)

    for i, timestep in tqdm(enumerate(scheduler.timesteps), disable=not show_progress):
        if start_timestep is not None and torch.is_tensor(start_timestep):
            # inject original latents until start_timestep
            timestep_mask = start_timestep[:, 0, 0] <= timestep
            latents[timestep_mask] = start_latents[timestep_mask]
        elif start_timestep is not None and timestep > start_timestep:
            continue
        timesteps = timestep[None, ...]
        model_input = _scale_model_input(scheduler, latents, timestep)
        noise_pred = model.denoiser(
            latents=model_input,
            timesteps=timesteps.repeat(latents.shape[0], 1, 1),
            **kwargs
        )
        latents = scheduler.step(noise_pred, timesteps, latents, return_dict=False)[0]
    return latents