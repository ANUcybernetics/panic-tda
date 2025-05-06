# TODO

- clustering charts:

  - % of outliers vs cluster members
  - cluster count histograms, faceted by text/image model and also by prompt
  - mosaic grids (stills) for different clusters
  - cluster bigrams (inc. pictures)
  - t-SNE picture (with different facetings)

- check normality

- violin plots for entropy

- idea: downsample before clustering (for computational reasons)

- generate pdf/svg plots by default (and png only for the heaps-overplotted
  ones)

- add semantic drift metrics back in (create a new polars expr-based helper
  function)

- autoregressive parameter (time column)

- group the prompts by (human-labelled) category

- order embeddings by "longest stable runs", and sample something from the
  longest ones for each network

- add some subsections to the design doc about the GenAIModel and EmbeddingModel
  superclasses (and why we chose the models and params we've currently chosen)

- populate the estimated_time function (for genai and embedding models) with
  real values

- check that ExperimentConfig deletion cascades to all runs (and that there are
  no invocations or embeddings that belong to a now-deleted experiment)

- export video improvements:

  - add a black separator between each "region" (same prompt & network)
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

- chart ideas:

  - [this one](https://altair-viz.github.io/gallery/select_detail.html) with PE
    on left, and PD on the right
  - add [strips](https://altair-viz.github.io/gallery/dot_dash_plot.html) to the
    new PD plots
  - maybe use a
    [minimap](https://altair-viz.github.io/gallery/scatter_with_minimap.html)
  - [wrapped facets](https://altair-viz.github.io/gallery/us_population_over_time_facet.html)
  - plot the
    [images in a tooltip](https://altair-viz.github.io/case_studies/numpy-tooltip-images.html)

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
