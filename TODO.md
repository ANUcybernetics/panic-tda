# TODO

- finish off the PDs for the big 5000 run

- delete any duplicate embeddings

- figure out what runs to do next

- map the generators into the PD chart (and display the images)

- implement mixed effects modelling & statistical testing

- overplot the jittered points over the boxplots

- update the "title card" mosaic image generation to create row/col title cards
  (for both prompt and network)

- add a "paper charts" CLI function

- check that ExperimentConfig deletion cascades to all runs (and therefore all
  invocations & embeddings)

- run a script to find orphaned runs and invocations in the db (not attached to
  an experiment config)

- look into indexes for the db

- tSNE chart would be cool/helpful (to see whether the different runs get
  clustered together)

- ensure persistence diagrams and mosaic videos use the same layout algo (for
  ease of comparison between the two)

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

- add florence2 or blip3 or some other (more modern) captioning model

- [try this approach](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47)
  to getting flux running on cybersonic, or perhaps onnx

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- visualise the time taken for the various different invocations

- run the tests in GitHub actions

- write an orphans (or some other validation that the run is all there)
  property/method for Run. Or maybe just a cleanup function

- create similarity matrices for runs

## Int8 quantization for Flux.1-schnell

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
