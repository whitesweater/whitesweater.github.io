---
title: 'SIRCL: Stabilizing Implicit Reasoning via Centripetal Latent Trajectories'

# Authors
# If you created a profile for a user (e.g. the default `me` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - me
  - Jindong Li
  - Changjing Zhu
  - Menglin Yang

# Author notes (optional)
author_notes:
  - 'Equal contribution'

date: '2026-01-01T00:00:00Z'

# Schedule page publish date (NOT publication's date).
publishDate: '2026-01-01T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['article']

# Publication metadata — structured fields used by citation styles and BibTeX export.
publication:
  name: "Under review at NeurIPS 2026"
  short_name: "NeurIPS 2026"

peer_reviewed: false
open_access: true
license: ""

# Awards, honors, and recognitions. Surfaced as badges on the page and in listings.
awards: []

# Funders and grants. Required by many funders for compliance reporting.
funding: []

abstract: This work studies latent-space instability and semantic drift in implicit chain-of-thought reasoning. It models latent tokens as continuous reasoning trajectories and proposes a lightweight geometric regularizer that constrains excessive trajectory deviation during training.

# Summary. An optional shortened abstract.
summary: A lightweight geometric regularization method for stabilizing implicit reasoning trajectories without adding inference-time overhead.

tags:
  - Implicit Chain-of-Thought
  - Latent-Space Reasoning
  - Language Model Reasoning

# Display this page in the Featured widget?
featured: true

# Standard identifiers for auto-linking
hugoblox:
  ids: {}

# Custom links
links:
  - type: pdf
    url: ""
  - type: code
    url: ""
  - type: dataset
    url: ""
  - type: slides
    url: ""
  - type: source
    url: ""
  - type: video
    url: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
  focal_point: ''
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

SIRCL investigates instability in implicit chain-of-thought reasoning, where latent-token trajectories may drift away from the problem core as reasoning steps increase. The method introduces a sample-level center and hinge-style trust-region loss to regularize excessive latent displacement during training while keeping inference cost unchanged.
