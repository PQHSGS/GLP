import unittest
from unittest import mock

import torch

from cli.stream_glp import build_parser
from glp import denoiser
from gemma2_pipeline.streaming import split_hf_checkpoint_ref


class SamplingMethodTests(unittest.TestCase):
    def test_canonicalize_noise_sampling_method_defaults_to_uniform(self):
        self.assertEqual(denoiser._canonicalize_noise_sampling_method(None), "uniform")
        self.assertEqual(denoiser._canonicalize_noise_sampling_method("OT"), "sot")
        self.assertEqual(denoiser._canonicalize_noise_sampling_method("sinkhorn"), "sinkhorn")

    def test_canonicalize_u_sampling_method_defaults_to_uniform(self):
        self.assertEqual(denoiser._canonicalize_u_sampling_method(None), "uniform")
        self.assertEqual(denoiser._canonicalize_u_sampling_method("beta"), "beta")
        self.assertEqual(denoiser._canonicalize_u_sampling_method("logit-normal"), "logit_normal")

    def test_beta_u_sampling_uses_fixed_parameters(self):
        latents = torch.zeros(4, 2, 3)
        generator = torch.Generator().manual_seed(123)

        with mock.patch("glp.denoiser.torch.rand", return_value=torch.tensor([0.0, 0.03125, 0.5, 1.0])) as rand_mock:
            sampled_u = denoiser._sample_training_u(latents, generator=generator, u_sampling_method="beta")

        rand_mock.assert_called_once()
        self.assertTrue(torch.allclose(sampled_u, torch.tensor([0.0, 0.5, 0.8706, 1.0]), atol=1e-4))

    def test_sot_sampling_requires_sequence_shaped_latents(self):
        latents = torch.tensor([[0.0], [10.0]], dtype=torch.float32)
        noise = torch.tensor([[9.0], [1.0]], dtype=torch.float32)

        with self.assertRaises(ValueError):
            denoiser._match_noise_to_latents_ot(latents, noise)

    def test_sot_sampling_reorders_noise_using_min_cost_assignment(self):
        latents = torch.tensor([[[0.0]], [[10.0]]], dtype=torch.float32)
        noise = torch.tensor([[[9.0]], [[1.0]]], dtype=torch.float32)

        matched_noise = denoiser._match_noise_to_latents_ot(latents, noise)

        expected = torch.tensor([[[1.0]], [[9.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(matched_noise, expected))

    def test_sot_sampling_supports_chunked_matching(self):
        latents = torch.tensor([[[0.0]], [[10.0]], [[20.0]], [[30.0]]], dtype=torch.float32)
        noise = torch.tensor([[[9.0]], [[1.0]], [[29.0]], [[21.0]]], dtype=torch.float32)

        matched_noise = denoiser._match_noise_to_latents_ot(latents, noise, chunk_size=2)

        expected = torch.tensor([[[1.0]], [[9.0]], [[21.0]], [[29.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(matched_noise, expected))

    def test_sinkhorn_sampling_uses_chunk_size_plumbing(self):
        latents = torch.tensor([[[0.0]], [[10.0]], [[20.0]], [[30.0]]], dtype=torch.float32)
        noise = torch.tensor([[[9.0]], [[1.0]], [[29.0]], [[21.0]]], dtype=torch.float32)

        with mock.patch("glp.denoiser.minibatch_sinkhorn_ot", side_effect=lambda x0, x1, **kwargs: x1) as sinkhorn_mock:
            matched_noise = denoiser.match_noise_to_latents_ot(
                latents,
                noise,
                method="sinkhorn",
                chunk_size=2,
                epsilon=0.2,
                iterations=7,
            )

        self.assertEqual(sinkhorn_mock.call_count, 2)
        self.assertEqual(sinkhorn_mock.call_args_list[0].kwargs["epsilon"], 0.2)
        self.assertEqual(sinkhorn_mock.call_args_list[0].kwargs["iterations"], 7)
        self.assertTrue(torch.equal(matched_noise, noise))

    def test_sinkhorn_repairs_duplicate_hard_matches(self):
        denoiser._SINKHORN_COLLISION_WARNING_EMITTED = False
        x0 = torch.zeros(4, 1, dtype=torch.float32)
        x1 = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

        matched = denoiser.minibatch_sinkhorn_ot(x0, x1, epsilon=0.05, iterations=5)

        self.assertEqual(torch.unique(matched).numel(), x1.shape[0])


class SplitOutputProjectionTests(unittest.TestCase):
    def test_split_output_projection_reconstructs_original_dimension_order(self):
        proj = denoiser.SplitOutputProj(d_model=2, d_input=5, tail_indices=[3, 1])
        with torch.no_grad():
            proj.tail_proj.weight.zero_()
            proj.tail_proj.bias.copy_(torch.tensor([10.0, 30.0]))
            proj.nontail_proj.weight.zero_()
            proj.nontail_proj.bias.copy_(torch.tensor([0.0, 2.0, 4.0]))

        x = torch.ones(2, 2)
        out = proj(x)

        expected = torch.tensor([
            [0.0, 10.0, 2.0, 30.0, 4.0],
            [0.0, 10.0, 2.0, 30.0, 4.0],
        ])
        self.assertTrue(torch.equal(out, expected))
        self.assertNotIn("tail_indices", proj.state_dict())
        self.assertNotIn("nontail_indices", proj.state_dict())

    def test_denoiser_uses_split_output_head_when_indices_are_valid(self):
        model = denoiser.TransformerMLPDenoiser(
            d_model=4,
            d_mlp=8,
            d_input=6,
            n_layers=1,
            split=True,
            split_tail_indices=[1, 4],
        )

        self.assertIsInstance(model.out_proj, denoiser.SplitOutputProj)
        self.assertEqual(model.split_tail_indices, [1, 4])

    def test_glp_selects_top_variance_dimensions_by_proportion(self):
        model = denoiser.GLP(
            normalizer_config={"d_input": 10, "normalization_method": "gaussian"},
            denoiser_config={"d_model": 4, "d_mlp": 8, "d_input": 10, "n_layers": 1},
        )
        model.normalizer.var.copy_(
            torch.tensor([1.0, 3.0, 2.0, 99.0, 4.0, 8.0, 5.0, 7.0, 6.0, 9.0])
        )

        tail_indices = model.configure_split_output_from_normalizer(proportion=0.1)

        self.assertEqual(tail_indices, [3])
        self.assertIsInstance(model.denoiser.model.out_proj, denoiser.SplitOutputProj)

    def test_glp_rejects_invalid_split_proportions(self):
        model = denoiser.GLP(
            normalizer_config={"d_input": 10, "normalization_method": "gaussian"},
            denoiser_config={"d_model": 4, "d_mlp": 8, "d_input": 10, "n_layers": 1},
        )

        with self.assertRaises(ValueError):
            model.configure_split_output_from_normalizer(proportion=0.0)
        with self.assertRaises(ValueError):
            model.configure_split_output_from_normalizer(proportion=1.0)

    def test_compute_balanced_loss_uses_tail_std_weights(self):
        v_pred = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
            dtype=torch.float32,
        )
        v_target = torch.zeros_like(v_pred)
        tail_indices = torch.tensor([2, 3])
        sem_indices = torch.tensor([0, 1])
        tail_weights = torch.tensor([2.0, 4.0])

        loss, tail_weighted_loss, tail_loss, sem_loss = denoiser.compute_balanced_loss(
            v_pred,
            v_target,
            tail_indices,
            sem_indices,
            tail_weights,
        )

        self.assertTrue(torch.isclose(tail_weighted_loss, torch.tensor(109.0)))
        self.assertTrue(torch.isclose(tail_loss, torch.tensor(34.5)))
        self.assertTrue(torch.isclose(sem_loss, torch.tensor(7.5)))
        self.assertTrue(torch.isclose(loss, torch.tensor(125.5)))


class StreamGlpParserTests(unittest.TestCase):
    def test_hf_checkpoint_ref_supports_subfolders(self):
        repo_id, subfolder = split_hf_checkpoint_ref("PQPQPQHUST/glp-gpt2-med/1B/ot_gauss/900M")

        self.assertEqual(repo_id, "PQPQPQHUST/glp-gpt2-med")
        self.assertEqual(subfolder, "1B/ot_gauss/900M")

    def test_parser_exposes_sampling_method(self):
        parser = build_parser()

        defaults = parser.parse_args([])
        enabled = parser.parse_args([
            "--noise-sampling-method",
            "sot",
            "--u-sampling-method",
            "logit_normal",
            "--ot-chunk-size",
            "128",
            "--split",
            "--split-proportion",
            "0.2",
            "--init-ckpt",
            "org/model",
            "--load-opt",
        ])

        self.assertEqual(defaults.noise_sampling_method, "uniform")
        self.assertEqual(defaults.u_sampling_method, "uniform")
        self.assertEqual(defaults.ot_chunk_size, 4096)
        self.assertFalse(defaults.split)
        self.assertEqual(defaults.split_proportion, 0.1)
        self.assertIsNone(defaults.init_ckpt)
        self.assertFalse(defaults.load_opt)
        self.assertEqual(enabled.noise_sampling_method, "sot")
        self.assertEqual(enabled.u_sampling_method, "logit_normal")
        self.assertEqual(enabled.ot_chunk_size, 128)
        self.assertTrue(enabled.split)
        self.assertEqual(enabled.split_proportion, 0.2)
        self.assertEqual(enabled.init_ckpt, "org/model")
        self.assertTrue(enabled.load_opt)

        sinkhorn_enabled = parser.parse_args(["--noise-sampling-method", "sinkhorn"])
        self.assertEqual(sinkhorn_enabled.noise_sampling_method, "sinkhorn")


if __name__ == "__main__":
    unittest.main()
