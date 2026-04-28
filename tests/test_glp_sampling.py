import unittest

import torch

from cli.stream_glp import build_parser
from glp import denoiser


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


class StreamGlpParserTests(unittest.TestCase):
    def test_parser_exposes_sampling_method(self):
        parser = build_parser()

        defaults = parser.parse_args([])
        enabled = parser.parse_args(["--sampling-method", "ot", "--ot-chunk-size", "128"])

        self.assertEqual(defaults.sampling_method, "uniform")
        self.assertEqual(defaults.ot_chunk_size, 256)
        self.assertEqual(enabled.sampling_method, "ot")
        self.assertEqual(enabled.ot_chunk_size, 128)


if __name__ == "__main__":
    unittest.main()
