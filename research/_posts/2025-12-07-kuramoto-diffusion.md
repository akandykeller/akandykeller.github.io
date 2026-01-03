---
layout: post
title: Kuramoto Orientation Diffusion Models
<!-- image: /assets/img/research/kuramoto-diffusion/teaser.png -->
sitemap: false
comments: true

venue: "NeurIPS '25"
award: ""
citation: "Y. Song, <strong>T. A. Keller</strong>, S. Brodjian, T. Miyato, Y. Yue, P. Perona, and M. Welling (2025). <i>Kuramoto Orientation Diffusion Models</i>. In: Advances in Neural Information Processing Systems (NeurIPS)."
paper_url: "https://arxiv.org/abs/2509.15328"

research_group: "Modeling Spatiotemporal Neural Dynamics"
---
![Kuramoto Orientation Diffusion Models](/assets/img/research/kuramoto-diffusion/teaser.png){:.lead width="500" height="320" loading="lazy"}
We build a score-based generative model on **periodic domains** by using **stochastic Kuramoto synchronization** as structured forward diffusion for orientation-rich images.
{:.figcaption}

Orientation-rich images, such as fingerprints and textures, often exhibit coherent angular directional patterns that are challenging to model using standard generative approaches based on isotropic Euclidean diffusion. Motivated by the role of phase synchronization in biological systems, we propose a score-based generative model built on periodic domains by leveraging stochastic Kuramoto dynamics in the diffusion process. In neural and physical systems, Kuramoto models capture synchronization phenomena across coupled oscillators -- a behavior that we re-purpose here as an inductive bias for structured image generation. In our framework, the forward process performs \textit{synchronization} among phase variables through globally or locally coupled oscillator interactions and attraction to a global reference phase, gradually collapsing the data into a low-entropy von Mises distribution. The reverse process then performs \textit{desynchronization}, generating diverse patterns by reversing the dynamics with a learned score function. This approach enables structured destruction during forward diffusion and a hierarchical generation process that progressively refines global coherence into fine-scale details. We implement wrapped Gaussian transition kernels and periodicity-aware networks to account for the circular geometry. Our method achieves competitive results on general image benchmarks and significantly improves generation quality on orientation-dense datasets like fingerprints and textures. Ultimately, this work demonstrates the promise of biologically inspired synchronization dynamics as structured priors in generative modeling.
{:.note title="Abstract"}
Yue Song, **T. Anderson Keller**, Sevan Brodjian, Takeru Miyato, Yisong Yue, Pietro Perona, Max Welling
{:.note title="Authors"}

*Accepted at [NeurIPS 2025](https://neurips.cc/virtual/2025/index.html) (Poster)* \\
*Paper:* <https://arxiv.org/abs/2509.15328> \\
*OpenReview:* <https://openreview.net/forum?id=dxK2QgEKvz>
{:.note title="Full Paper"}

{:.note title="Code"}
<https://github.com/KingJamesSong/OrientationDiffusion>

- Table of Contents
{:toc}
