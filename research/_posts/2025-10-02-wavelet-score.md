---
layout: post
title: "Where the Score Lives: A Wavelet View of Diffusion"
<!-- image: /assets/img/research/wavelet-score/teaser.png -->
sitemap: false
comments: true

venue: "AISTATS '26"
award: ""
citation: "E. L. Byrnes Finn, B. Wang, <strong>T. A. Keller</strong>, and D. E. Ba (2026). <i>Where the Score Lives: A Wavelet View of Diffusion</i>. In: Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS). Also accepted at SPIGM Workshop @ NeurIPS 2025."
paper_url: "https://openreview.net/forum?id=fmlzopxjxq"
---
![Where the Score Lives](/assets/img/research/wavelet-score/teaser.png){:.lead width="500" height="320" loading="lazy"}
We give an analytically solvable **wavelet-basis** parameterization of diffusion scores in terms of **data moments**, offering an architecture-agnostic view of what matters for denoising.
{:.figcaption}

Diffusion models have had remarkable success over the last decade in generating a diverse set of visually plausible images. These models work by transforming the data to a centered Gaussian and then learning the reverse process by training a neural network to approximate the score of the underlying distribution. A variety of architectures from CNNs, to U-Nets, to transformers have been used as the score-approximation network in diffusion modeling. We propose an analytically solvable parameterization of the score function using an expansion in a wavelet basis. In particular, we derive interpretable optimal score functions in a 2D, orthogonal wavelet basis in terms of the moments of the data distribution. We use this parametrization to provide an architecture-agnostic, moment-based analysis that reveals which attributes of the data distribution tend to matter most for denoising. Our score machine is flexible enough to partially mimic the relevant inductive biases of multiple architectures, including U-Nets, and CNNs, taking a step towards understanding why different score architectures can exhibit distinct generative behavior. Since our score is solvable in terms of the moments of the data, we can begin to understand how the data distribution interacts with the score network to produce the behavior we observe in diffusion models.
{:.note title="Abstract"}
Emma Lucia Byrnes Finn, Binxu Wang, **T. Anderson Keller**, Demba E. Ba
{:.note title="Authors"}

*Accepted in Proceedings of [AISTATS '26](https://virtual.aistats.org/Conferences/2026)*\\
*Also Accepted at [SPIGM Workshop @ NeurIPS 2025](https://neurips.cc/virtual/2025/workshop/109570)* \\
*Paper:* <https://openreview.net/forum?id=fmlzopxjxq>
{:.note title="Full Paper"}

<!--
{:.note title="Code"}
<add link here if/when available>
-->

- Table of Contents
{:toc}