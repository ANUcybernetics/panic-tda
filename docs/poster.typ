// Copyright (c) 2025 Ben Swift
// Licensed under CC BY-NC-SA 4.0

// Import base template for colors and styling
#import "@local/anu-typst-template:0.2.0": *

#show: doc => anu(
  title: "PANIC-TDA",
  paper: "a3",
  footer_text: text(
    font: "Neon Tubes 2",
    fill: anu-colors.socy-yellow,
    "CC BY-NC-SA 4.0",
  ),
  config: (
    theme: "dark",
    logos: ("studio",),
    hide: ("page-numbers", "title-block"),
  ),
  page-settings: (
    flipped: true,
  ),
  doc,
)

// Content: 2-column layout
#grid(
  columns: (1fr, 1fr),
  gutter: 2cm,
  [
    #v(3cm)
    #text(size: 3em, fill: anu-colors.gold)[*PANIC-TDA*]

    #text(size: 1.2em)[
      Exploring the patterns that emerge when generative AI models are connected
      in networks, feeding their outputs recursively back as inputs.
    ]

    #v(1cm)

    == What you're watching

    Each square in the mosaic grid shows one independent trajectory through
    semantic space. All trajectories begin with the same text prompt and follow
    the same network of AI models, but with different random seeds.

    Watch as the images evolve over time---sometimes converging to similar
    patterns, sometimes diverging into completely different semantic
    territories.

    #v(1cm)

    == Why this matters

    These trajectories reveal how information flows through networks of
    generative AI models. By systematically exploring thousands of runs, we can
    understand the attractors, bifurcations, and topological structure of these
    high-dimensional information processing systems.
  ],
  [
    == The PANIC-TDA pipeline

    === Stage 1: runs

    Execute networks of generative AI models where outputs become inputs. Each
    run follows a specific network (cyclic graph) of models:

    - start with an initial text prompt
    - generate an image using a text-to-image model
    - describe that image using an image-to-text model
    - use that description to generate a new image
    - repeat for thousands of iterations

    === Stage 2: embeddings

    Embed each text output into high-dimensional semantic space using embedding
    models. Text that is semantically similar will be close together in this
    768-dimensional space, creating a continuous trajectory through semantic
    territory.

    === Stage 3: topological analysis

    Apply persistent homology to characterize the shape and structure of each
    trajectory. This reveals:

    - loops and cycles in semantic space
    - convergence to stable attractors
    - bifurcation points where trajectories diverge
    - topological signatures that distinguish different networks

    #v(1cm)

    == The bigger picture

    This research grows out of the PANIC! interactive art installation at the
    ANU School of Cybernetics, where visitors have been exploring these
    AI-mediated transformations since 2022. By moving from real-time interaction
    to systematic batch analysis, we can ask deeper questions about how
    information propagates through these vast, nonlinear, multi-billion
    parameter systems.
  ],
)
