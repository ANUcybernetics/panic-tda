# TODO

- clustering

  - show transition matrix faceted by `~ embedding_model + network`
  - order embeddings by "longest stable runs" (with `.rle()`), and calculate
    longest runs for each network (both "from start" and "anytime")
  - distribution of run lengths for different cluster labels, again faceted by
    embedding_model ~ network
  - label distributions by seq no (perhaps beginning/middle/end)
  - how often does it return to the initial cluster
  - how often in a cluster vs outlier
  - how many of the labels are similar across different embeddings (or some
    metric on which embeddings get clustered together across different embedding
    models)... double-check that the clustering stuff is actually being faceted
    correctly

- get h_n subscripting for charts (maybe even using an enum at the polars level)

- by creating synthetic data with known drift values, check that both the
  euclidean and cosine "fetch and calculate" functions are returning the correct
  results

- check normality

- autoregressive parameter (time column)

- add some subsections to the design doc about the GenAIModel and EmbeddingModel
  superclasses (and why we chose the models and params we've currently chosen)

- (maybe) add the vector embedding of the initial prompt to the Run object (to
  save having to re-calculate it later)

- populate the estimated_time function (for genai and embedding models) with
  real values

- check that ExperimentConfig deletion cascades to all runs (and that there are
  no invocations or embeddings that belong to a now-deleted experiment)

- export video improvements:

  - add visual indicator for when there's a semantic "jump"
  - add colour coding to the different model names
  - "one label per row" and "one label per column"
  - re-add the `prompt_order` parameter

- tSNE chart would be cool/helpful (to see whether the different runs get
  clustered together)

- add florence2 or blip3 or some other (more modern) captioning model

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- visualise the time taken for the various different invocations

- run the tests in GitHub actions

- create similarity matrices for runs

- DB indexes

## Int8 quantization for Flux.1-schnell

- [try this approach](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47)
  to getting flux running on cybersonic, or perhaps onnx

```python
class FluxSchnell(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device with int8 weight-only quantization."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")


        # Define the int8 weight-only quantization configuration
        # Using the AOBaseConfig approach (recommended for torchao >= 0.10.0)
        quant_config = Int8WeightOnlyConfig()  # Default group_size optimizes for balance, good for VRAM
        quantization_config = TorchAoConfig(quant_type=quant_config)

        logger.info("Applying int8 weight-only quantization using torchao.")

        # Initialize the model with quantization config
        # Using device_map="auto" and torch_dtype="auto" as recommended with quantization
        self._model = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            use_fast=True,
            device_map="balanced",  # Use device_map for better handling with quantization
            quantization_config=quantization_config,
        )

        # Try to compile the UNet's forward method (torchao is compatible with torch.compile)
        try:
            if hasattr(self._model, "unet") and hasattr(self._model.unet, "forward"):
                logger.info("Attempting to compile UNet forward method...")
                original_forward = self._model.unet.forward
                self._model.unet.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",  # Good mode for inference speedup
                    fullgraph=True,
                    dynamic=False,
                )
                logger.info("Successfully compiled UNet forward method.")
            else:
                 logger.warning("Could not find UNet or its forward method for compilation.")

        except Exception as e:
            logger.warning(f"Could not compile FluxSchnell UNet forward method: {e}")

        logger.info(f"Model {self.__class__.__name__} (int8 quantized) loaded successfully")

    def invoke(self, prompt: str, seed: int) -> Image.Image:
        """Generate an image from a text prompt using the quantized model"""
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Inference parameters might need tuning after quantization, but start with original values
        image = self._model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=6, # Schnell uses fewer steps
            generator=generator,
        ).images[0]

        return image
```

(on cybersonic, the test runs in ~220 sec, so pretty slow)

## paper notes

- https://arxiv.org/abs/2401.17072
- STS explained
  https://dl.acm.org/doi/abs/10.1145/3440755?casa_token=b4hPhdIWOEMAAAAA:yQQeVE9NIVFz-DE7pjjI6F_yqM4kYr92t2O5o6qxT6kE2lPt3rPS674MnC9evHTPmiVaUYgHb_sPcjs
