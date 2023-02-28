#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from python_coreml_stable_diffusion import unet

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import coremltools as ct
from diffusers import StableDiffusionPipeline
import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
from python_coreml_stable_diffusion import chunk_mlprogram
import requests
import shutil
import time

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# torch.set_grad_enabled(False)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from types import MethodType


def _get_coreml_inputs(sample_inputs):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, paddle.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


ABSOLUTE_MIN_PSNR = 35


def report_correctness(original_outputs, final_outputs, log_prefix):
    """ Report PSNR values across two compatible tensors
    """
    original_psnr = compute_psnr(original_outputs, original_outputs)
    final_psnr = compute_psnr(original_outputs, final_outputs)

    dB_change = final_psnr - original_psnr
    logger.info(
        f"{log_prefix}: PSNR changed by {dB_change:.1f} dB ({original_psnr:.1f} -> {final_psnr:.1f})"
    )

    if final_psnr < ABSOLUTE_MIN_PSNR:
        raise ValueError(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr


def _get_out_path(args, submodule_name):
    fname = f"Stable_Diffusion_version_{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    return os.path.join(args.o, fname)


# https://github.com/apple/coremltools/issues/1680
def _save_mlpackage(model, output_path):
    # First recreate MLModel object using its in memory spec, then save
    ct.models.MLModel(model._spec,
                      weights_dir=model._weights_dir,
                      is_temp_package=True).save(output_path)


def _convert_to_coreml(submodule_name, pdmodle_module, sample_inputs,
                       output_names, args):
    out_path = _get_out_path(args, submodule_name)

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(
            out_path, compute_units=ct.ComputeUnit[args.compute_unit])
        logger.info(
            f"Loading {out_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = ct.ComputeUnit[args.compute_unit]
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        coreml_model = ct.convert(
            pdmodle_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=_get_coreml_inputs(sample_inputs),
            outputs=[ct.TensorType(name=name) for name in output_names],
            compute_units=ct.ComputeUnit[args.compute_unit],
            # skip_model_load=True,
    )

    del pdmodle_module
    gc.collect()

    coreml_model.save(out_path)
    logger.info(f"Saved {submodule_name} model to {out_path}")

    return coreml_model, out_path


# def quantize_weights_to_8bits(args):
#     for model_name in [
#             "text_encoder", "vae_decoder", "unet", "unet_chunk1",
#             "unet_chunk2", "safety_checker"
#     ]:
#         out_path = _get_out_path(args, model_name)
#         if os.path.exists(out_path):
#             logger.info(f"Quantizing {model_name}")
#             mlmodel = ct.models.MLModel(out_path,
#                                         compute_units=ct.ComputeUnit.CPU_ONLY)
#             mlmodel = ct.compression_utils.affine_quantize_weights(
#                 mlmodel, mode="linear")
#             mlmodel.save(out_path)
#             logger.info("Done")
#         else:
#             logger.info(
#                 f"Skipped quantizing {model_name} (Not found at {out_path})")


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(
            f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(
        os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile {source_model_path} {output_dir}")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def bundle_resources_for_swift_cli(args):
    """
    - Compiles Core ML models from mlpackage into mlmodelc format
    - Download tokenizer resources for the text encoder
    """
    resources_dir = os.path.join(args.o, "Resources")
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir, exist_ok=True)
        logger.info(f"Created {resources_dir} for Swift CLI assets")

    # Compile model using coremlcompiler (Significantly reduces the load time for unet)
    for source_name, target_name in [("text_encoder", "TextEncoder"),
                                     ("vae_decoder", "VAEDecoder"),
                                     ("unet", "Unet"),
                                     ("unet_chunk1", "UnetChunk1"),
                                     ("unet_chunk2", "UnetChunk2"),
                                     ("safety_checker", "SafetyChecker")]:
        source_path = _get_out_path(args, source_name)
        if os.path.exists(source_path):
            target_path = _compile_coreml_model(source_path, resources_dir,
                                                target_name)
            logger.info(f"Compiled {source_path} to {target_path}")
        else:
            logger.warning(
                f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
            )

    # Fetch and save vocabulary JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer vocab.json")
    with open(os.path.join(resources_dir, "vocab.json"), "wb") as f:
        f.write(requests.get(args.text_encoder_vocabulary_url).content)
    logger.info("Done")

    # Fetch and save merged pairs JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer merges.txt")
    with open(os.path.join(resources_dir, "merges.txt"), "wb") as f:
        f.write(requests.get(args.text_encoder_merges_url).content)
    logger.info("Done")

    return resources_dir


def convert_text_encoder(args):
    """ Converts the text encoder component of Stable Diffusion
    """

    path = "/Users/lishicheng03/stable-diffusion-v1-5-pd/text_encoder/"

    
    out_path = _get_out_path(args, "text_encoder")
    
    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = 77

    sample_text_encoder_inputs = {
        "input_ids":
        paddle.ones(
            shape=(1, text_encoder_sequence_length),
            dtype=paddle.int32,
        )
    }

    coreml_text_encoder, out_path = _convert_to_coreml(
        "text_encoder", path, sample_text_encoder_inputs,
        ["layer_norm_97.tmp_2", "gather_nd_0.tmp_0"], args)

    _save_mlpackage(coreml_text_encoder, out_path)
    logger.info(f"Saved text_encoder into {out_path}")
    gc.collect()


def convert_vae_decoder(args):
    """ Converts the VAE Decoder component of Stable Diffusion
    """
    path = "/Users/lishicheng03/stable-diffusion-v1-5-pd/vae_decoder/"

    out_path = _get_out_path(args, "vae_decoder")

    latent_shape = (1, 4, 64, 64)

    sample_vae_decoder_inputs = {
        "latent": paddle.rand(latent_shape, dtype=paddle.float32)
    }

    coreml_vae_decoder, out_path = _convert_to_coreml(
        "vae_decoder", path, sample_vae_decoder_inputs,
        ["conv2d_197.tmp_1"], args)

    _save_mlpackage(coreml_vae_decoder, out_path)

    logger.info(f"Saved vae_decoder into {out_path}")
    gc.collect()


def convert_unet(args):
    """ Converts the UNet component of Stable Diffusion
    """
    path = "/Users/lishicheng03/stable-diffusion-v1-5-pd/unet_einsum/"

    out_path = _get_out_path(args, "unet")

    # Prepare sample input shapes and values
    batch_size = 2  # for classifier-free guidance
    sample_shape = (
        batch_size,                    # B
        4,  # C
        64,  # H
        64,  # W
    )

    encoder_hidden_states_shape = (
        batch_size,
        1024,
        1,
        77
    )

    sample_unet_inputs = {
        "sample": paddle.rand(shape=sample_shape, dtype=paddle.float32),
        "timestep":
            paddle.Tensor(np.array([981], dtype=np.int32)),
        "encoder_hidden_states": paddle.rand(shape=encoder_hidden_states_shape, dtype=paddle.float32),
    }

    coreml_unet, out_path = _convert_to_coreml("unet", path,
                                                sample_unet_inputs,
                                                ["conv2d_563.tmp_1"], args)

    gc.collect()


    _save_mlpackage(coreml_unet, out_path)
    logger.info(f"Saved unet into {out_path}")


    args.mlpackage_path = out_path
    args.remove_original = False
    chunk_mlprogram.main(args)


def convert_safety_checker(pipe, args):
    """ Converts the Safety Checker component of Stable Diffusion
    """
    if pipe.safety_checker is None:
        logger.warning(
            f"diffusers pipeline for {args.model_version} does not have a `safety_checker` module! " \
            "`--convert-safety-checker` will be ignored."
        )
        return

    out_path = _get_out_path(args, "safety_checker")
    if os.path.exists(out_path):
        logger.info(
            f"`safety_checker` already exists at {out_path}, skipping conversion."
        )
        return

    sample_image = np.random.randn(
        1,  # B
        pipe.vae.config.sample_size,  # H
        pipe.vae.config.sample_size,  # w
        3  # C
    ).astype(np.float32)

    # Note that pipe.feature_extractor is not an ML model. It simply
    # preprocesses data for the pipe.safety_checker module.
    safety_checker_input = pipe.feature_extractor(
        pipe.numpy_to_pil(sample_image),
        return_tensors="pt",
    ).pixel_values.to(torch.float32)

    sample_safety_checker_inputs = OrderedDict([
        ("clip_input", safety_checker_input),
        ("images", torch.from_numpy(sample_image)),
        ("adjustment", torch.tensor([0]).to(torch.float32)),
    ])

    sample_safety_checker_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_safety_checker_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_safety_checker_inputs_spec}")

    # Patch safety_checker's forward pass to be vectorized and avoid conditional blocks
    # (similar to pipe.safety_checker.forward_onnx)
    from diffusers.pipelines.stable_diffusion import safety_checker

    def forward_coreml(self, clip_input, images, adjustment):
        """ Forward pass implementation for safety_checker
        """

        def cosine_distance(image_embeds, text_embeds):
            return F.normalize(image_embeds) @ F.normalize(
                text_embeds).transpose(0, 1)

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = special_scores.gt(0).float().sum(dim=1).gt(0).float()
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = concept_scores.gt(0).float().sum(dim=1).gt(0)[:,
                                                                          None,
                                                                          None,
                                                                          None]

        has_nsfw_concepts_inds, _ = torch.broadcast_tensors(
            has_nsfw_concepts, images)
        images[has_nsfw_concepts_inds] = 0.0  # black image

        return images, has_nsfw_concepts.float(), concept_scores

    baseline_safety_checker = deepcopy(pipe.safety_checker.eval())
    setattr(baseline_safety_checker, "forward",
            MethodType(forward_coreml, baseline_safety_checker))

    # In order to parity check the actual signal, we need to override the forward pass to return `concept_scores` which is the
    # output before thresholding
    # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L100
    def forward_extended_return(self, clip_input, images, adjustment):

        def cosine_distance(image_embeds, text_embeds):
            normalized_image_embeds = F.normalize(image_embeds)
            normalized_text_embeds = F.normalize(text_embeds)
            return torch.mm(normalized_image_embeds,
                            normalized_text_embeds.t())

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0

        return images, has_nsfw_concepts, concept_scores

    setattr(pipe.safety_checker, "forward",
            MethodType(forward_extended_return, pipe.safety_checker))

    # Trace the safety_checker model
    logger.info("JIT tracing..")
    traced_safety_checker = torch.jit.trace(
        baseline_safety_checker, list(sample_safety_checker_inputs.values()))
    logger.info("Done.")
    del baseline_safety_checker
    gc.collect()

    # Cast all inputs to float16
    coreml_sample_safety_checker_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_safety_checker_inputs.items()
    }

    # Convert safety_checker model to Core ML
    coreml_safety_checker, out_path = _convert_to_coreml(
        "safety_checker", traced_safety_checker,
        coreml_sample_safety_checker_inputs,
        ["filtered_images", "has_nsfw_concepts", "concept_scores"], args)

    # Set model metadata
    coreml_safety_checker.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_safety_checker.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_safety_checker.version = args.model_version
    coreml_safety_checker.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_safety_checker.input_description["clip_input"] = \
        "The normalized image input tensor resized to (224x224) in channels-first (BCHW) format"
    coreml_safety_checker.input_description["images"] = \
        f"Output of the vae_decoder ({pipe.vae.config.sample_size}x{pipe.vae.config.sample_size}) in channels-last (BHWC) format"
    coreml_safety_checker.input_description["adjustment"] = \
        "Bias added to the concept scores to trade off increased recall for reduce precision in the safety checker classifier"

    # Set the output descriptions
    coreml_safety_checker.output_description["filtered_images"] = \
        f"Identical to the input `images`. If safety checker detected any sensitive content, " \
        "the corresponding image is replaced with a blank image (zeros)"
    coreml_safety_checker.output_description["has_nsfw_concepts"] = \
        "Indicates whether the safety checker model found any sensitive content in the given image"
    coreml_safety_checker.output_description["concept_scores"] = \
        "Concept scores are the scores before thresholding at zero yields the `has_nsfw_concepts` output. " \
        "These scores can be used to tune the `adjustment` input"

    _save_mlpackage(coreml_safety_checker, out_path)

    if args.check_output_correctness:
        baseline_out = pipe.safety_checker(
            **sample_safety_checker_inputs)[2].numpy()
        coreml_out = coreml_safety_checker.predict(
            coreml_sample_safety_checker_inputs)["concept_scores"]
        report_correctness(
            baseline_out, coreml_out,
            "safety_checker baseline Paddle to reference CoreML")

    del traced_safety_checker, coreml_safety_checker, pipe.safety_checker
    gc.collect()


def main(args):
    os.makedirs(args.o, exist_ok=True)
    # Convert models
    if args.convert_vae_decoder:
        logger.info("Converting vae_decoder")
        convert_vae_decoder(args)
        logger.info("Converted vae_decoder")

    if args.convert_unet:
        logger.info("Converting unet")
        convert_unet(args)
        logger.info("Converted unet")

    if args.convert_text_encoder:
        logger.info("Converting text_encoder")
        convert_text_encoder(args)
        logger.info("Converted text_encoder")

    if args.bundle_resources_for_swift_cli:
        logger.info("Bundling resources for the Swift CLI")
        bundle_resources_for_swift_cli(args)
        logger.info("Bundled resources for the Swift CLI")

    # if args.quantize_weights_to_8bits:
    #     # Note: Not recommended, significantly degrades generated image quality
    #     logger.info("Quantizing weights to 8-bit precision")
    #     quantize_weights_to_8bits(args)
    #     logger.info("Quantized weights to 8-bit precision")



def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--convert-unet", action="store_true")
    parser.add_argument("--convert-safety-checker", action="store_true")
    parser.add_argument(
        "--model-version",
        default="CompVis/stable-diffusion-v1-4",
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
         ))
    parser.add_argument("--compute-unit",
                        choices=tuple(cu
                                      for cu in ct.ComputeUnit._member_names_),
                        default="ALL")

    parser.add_argument(
        "--latent-h",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of rows) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--latent-w",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of cols) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=tuple(ai
                      for ai in unet.AttentionImplementations._member_names_),
        default=unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help=
        "The enumerated implementations trade off between ANE and GPU performance",
    )
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original Paddle and final CoreML models and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help=
        ("If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
         "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
         ))
    parser.add_argument(
        "--quantize-weights-to-8bits",
        action="store_true",
        help=
        ("If specified, quantize 16-bits weights to 8-bits weights in-place for all models. "
         "Not recommended as the generated image quality degraded significantly after 8-bit weight quantization"
         ))

    # Swift CLI Resource Bundling
    parser.add_argument(
        "--bundle-resources-for-swift-cli",
        action="store_true",
        help=
        ("If specified, creates a resources directory compatible with the sample Swift CLI. "
         "It compiles all four models and adds them to a StableDiffusionResources directory "
         "along with a `vocab.json` and `merges.txt` for the text tokenizer"))
    parser.add_argument(
        "--text-encoder-vocabulary-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
        help="The URL to the vocabulary file use by the text tokenizer")
    parser.add_argument(
        "--text-encoder-merges-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
        help="The URL to the merged pairs used in by the text tokenizer.")

    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()

    main(args)
