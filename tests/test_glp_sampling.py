import unittest

import torch

from cli.stream_glp import build_parser
from glp import denoiser
from gemma2_pipeline.streaming import split_hf_checkpoint_ref


class SamplingMethodTests(unittest.TestCase):
    def test_canonicalize_sampling_method_defaults_to_uniform(self):
        self.assertEqual(denoiser._canonicalize_sampling_method(None), "uniform")
        self.assertEqual(denoiser._canonicalize_sampling_method("OT"), "ot")

    def test_ot_sampling_requires_sequence_shaped_latents(self):
        latents = torch.tensor([[0.0], [10.0]], dtype=torch.float32)
        noise = torch.tensor([[9.0], [1.0]], dtype=torch.float32)

        with self.assertRaises(ValueError):
            denoiser._match_noise_to_latents_ot(latents, noise)

    def test_ot_sampling_reorders_noise_using_min_cost_assignment(self):
        latents = torch.tensor([[[0.0]], [[10.0]]], dtype=torch.float32)
        noise = torch.tensor([[[9.0]], [[1.0]]], dtype=torch.float32)

        matched_noise = denoiser._match_noise_to_latents_ot(latents, noise)

        expected = torch.tensor([[[1.0]], [[9.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(matched_noise, expected))

    def test_ot_sampling_supports_chunked_matching(self):
        latents = torch.tensor([[[0.0]], [[10.0]], [[20.0]], [[30.0]]], dtype=torch.float32)
        noise = torch.tensor([[[9.0]], [[1.0]], [[29.0]], [[21.0]]], dtype=torch.float32)

        matched_noise = denoiser._match_noise_to_latents_ot(latents, noise, chunk_size=2)

        expected = torch.tensor([[[1.0]], [[9.0]], [[21.0]], [[29.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(matched_noise, expected))


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


class StreamGlpParserTests(unittest.TestCase):
    def test_hf_checkpoint_ref_supports_subfolders(self):
        repo_id, subfolder = split_hf_checkpoint_ref("PQPQPQHUST/glp-gpt2-med/1B/ot_gauss/900M")

        self.assertEqual(repo_id, "PQPQPQHUST/glp-gpt2-med")
        self.assertEqual(subfolder, "1B/ot_gauss/900M")

    def test_parser_exposes_sampling_method(self):
        parser = build_parser()

        defaults = parser.parse_args([])
        enabled = parser.parse_args([
            "--sampling-method",
            "ot",
            "--ot-chunk-size",
            "128",
            "--split",
            "--split-proportion",
            "0.2",
            "--init-ckpt",
            "org/model",
            "--load-opt",
        ])

        self.assertEqual(defaults.sampling_method, "uniform")
        self.assertEqual(defaults.ot_chunk_size, 256)
        self.assertFalse(defaults.split)
        self.assertEqual(defaults.split_proportion, 0.1)
        self.assertIsNone(defaults.init_ckpt)
        self.assertFalse(defaults.load_opt)
        self.assertEqual(enabled.sampling_method, "ot")
        self.assertEqual(enabled.ot_chunk_size, 128)
        self.assertTrue(enabled.split)
        self.assertEqual(enabled.split_proportion, 0.2)
        self.assertEqual(enabled.init_ckpt, "org/model")
        self.assertTrue(enabled.load_opt)


if __name__ == "__main__":
    unittest.main()
