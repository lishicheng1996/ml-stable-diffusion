#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import coremltools as ct
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

import paddle


def _get_coreml_inputs(sample_inputs):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, paddle.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


def _get_out_path(args, submodule_name):
    fname = f"Stable_Diffusion_version_{submodule_name}.mlpackage"
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
    )

    del pdmodle_module
    gc.collect()

    coreml_model.save(out_path)
    logger.info(f"Saved {submodule_name} model to {out_path}")

    return coreml_model, out_path


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

    path = args.text_encoder_path
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
    path = args.vae_decoder_path
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
    path = args.unet_path
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



def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--text-encoder-path", type=str, default="path/to/text_encoder")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--vae-decoder-path", type=str, default="path/to/vae_decoder")
    parser.add_argument("--convert-unet", action="store_true")
    parser.add_argument("--unet-path", type=str, default="path/to/unet")
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help=
        ("If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
         "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
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
