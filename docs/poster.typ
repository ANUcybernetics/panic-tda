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
      What happens when AI models talk to each other in an endless loop?
    ]

    #v(1cm)

    == What you're watching

    Each square shows one trajectory starting from the same text prompt (like "a
    cat" or "a dog"). The journey is simple:

    - text→image model generates a picture
    - image→text model describes what it sees
    - that description generates a new picture
    - repeat 1000 times

    All squares follow the same network of models with the same starting prompt,
    but different random seeds. Watch how some converge to similar patterns
    while others drift into entirely different semantic territories.

    #v(1cm)

    == Semantic telephone

    Like the children's game of Telephone, information transforms with each
    iteration. But unlike random drift, these trajectories reveal hidden
    structure: some prompts create stable loops, others chaotic wandering, and
    some collapse into strange attractors (notice all that green leafy
    imagery?).
  ],
  [
    == Why study this?

    These recursive networks are more than curiosities. They reveal fundamental
    properties of how semantic information flows through AI systems. By running
    thousands of trajectories we can:

    - map the "semantic landscape" these models inhabit
    - identify stable attractors and chaotic regions
    - understand which model architectures preserve meaning vs introduce drift
    - use topological data analysis to characterise trajectory structure

    #v(1cm)

    == The three-stage pipeline

    === Stage 1: generate trajectories

    Run the text→image→text loop 1000 times for each combination of models,
    prompts, and random seeds.

    === Stage 2: embed in semantic space

    Map each text output to a point in 768-dimensional space where semantically
    similar texts are close together. Each trajectory becomes a path through
    this high-dimensional landscape.

    === Stage 3: topological analysis

    Use persistent homology to measure the shape of each trajectory: are there
    loops? Convergence points? Bifurcations? This reveals patterns invisible to
    clustering alone.

    #v(1cm)

    == From art to science

    This research grew from the PANIC! interactive art installation at ANU
    School of Cybernetics (2022), where visitors watched their prompts transform
    through AI networks in real time. These systematic batch experiments let us
    ask deeper questions about information transmission in complex,
    multi-billion parameter systems.

    #v(0.5cm)

    #text(size: 0.9em, fill: gray)[
      Code and data: #link("https://github.com/anucybernetics/panic-tda")
    ]
  ],
)
