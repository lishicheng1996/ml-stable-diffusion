# paddle2coreml stable-diffusion



## <a name="converting-models-to-coreml"></a> Converting Models to Core ML

**Step 1:** Create a Python environment and install dependencies:

```bash
conda create -n coreml_stable_diffusion python=3.8 -y
conda activate coreml_stable_diffusion
cd /path/to/cloned/ml-stable-diffusion/repository
pip install -e .
```
**Step 2:** Change the path in python_coreml_stable_diffusion/paddle2coreml.py of unet/text_encoder/vae_decoder to your corresponding directories.

**Step 3:** Execute the following command from the Terminal to generate Core ML model files (`.mlpackage`)

```shell
python -m python_coreml_stable_diffusion.paddle2coreml --convert-unet --convert-text-encoder --convert-vae-decoder -o <output-mlpackages-directory>
```


 Some additional notable arguments:


- `--bundle-resources-for-swift-cli`: Compiles all 4 models and bundles them along with necessary resources for text tokenization into `<output-mlpackages-directory>/Resources` which should provided as input to the Swift package. This flag is not necessary for the diffusers-based Python pipeline.

- `--chunk-unet`: Splits the Unet model in two approximately equal chunks (each with less than 1GB of weights) for mobile-friendly deployment. This is **required** for Neural Engine deployment on iOS and iPadOS. This is not required for macOS. Swift CLI is able to consume both the chunked and regular versions of the Unet model but prioritizes the former. Note that chunked unet is not compatible with the Python pipeline because Python pipeline is intended for macOS only. Chunking is for on-device deployment with Swift only.

