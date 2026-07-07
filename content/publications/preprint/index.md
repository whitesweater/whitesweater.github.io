---
title: "Think Wider: Mitigating Latent Rank Collapse in Implicit Chain-of-Thought Reasoning"
authors:
- me
- Menglin Yang
date: "2026-01-01T00:00:00Z"

# Schedule page publish date (NOT publication's date).
publishDate: "2026-01-01T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["article"]

# Publication metadata — structured fields used by citation styles and BibTeX export.
# Preprints typically have no formal venue; omit `publication` until the work is accepted.

peer_reviewed: false
open_access: true
license: ""

publication:
  name: "Under review at EMNLP 2026"
  short_name: "EMNLP 2026"

funding: []

abstract: This work identifies latent rank collapse in implicit chain-of-thought reasoning, where multi-step latent tokens over-concentrate along shared dominant directions, reducing effective rank and increasing inter-step redundancy. It proposes WIDER, a lightweight spectral regularization strategy that penalizes projection onto dominant latent directions during training.

# Summary. An optional shortened abstract.
summary: A spectral regularization method for improving latent-token diversity and mitigating rank collapse in implicit reasoning.

tags:
- Implicit Chain-of-Thought
- Latent Rank Collapse
- Spectral Regularization

featured: true

hugoblox:
  ids: {}

links:
- type: preprint
  url: ""
- type: code
  url: ""
- type: slides
  url: ""
- type: dataset
  url: ""
- type: poster
  url: ""
- type: source
  url: ""
- type: video
  url: ""
- type: custom
  label: Under review
  url: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/s9CC2SKySJM)'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/projects/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---

> [!NOTE]
> This paper is currently under review.

Think Wider studies how implicit reasoning trajectories can collapse into low-rank latent representations. The proposed WIDER objective estimates dominant singular directions and discourages latent tokens from concentrating along them, improving effective rank and reducing redundant steps without adding inference-time overhead.
