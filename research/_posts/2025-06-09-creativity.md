---
layout: post
title: Origins of Creativity in Attention Based Diffusion Models
<!-- image: /assets/img/research/creativity/teaser.png -->
sitemap: false
comments: true

venue: "ICML '25"
award: "HiLD Workshop"
citation: "E. L. B. Finn, <strong>T. A. Keller</strong>, M. Theodosis, and D. E. Ba (2025). <i>Origins of Creativity in Attention Based Diffusion Models</i>. In: High-dimensional Learning Dynamics 2025 @ ICML ’25."
paper_url: "https://openreview.net/forum?id=xMqRYKvrD1"
---
![Origins of Creativity](/assets/img/research/creativity/teaser.png){:.lead width="500" height="320" loading="lazy"}
We extend theory on “creativity” in convolutional diffusion models to the attention setting, predicting that attention promotes **global self-consistency** beyond patch-level mosaics.
{:.figcaption}

As diffusion models have become the tool of choice for image generation and as the quality of the images continues to improve, the question of how creativity originates in diffusion has become increasingly important. The score matching perspective on diffusion has proven particularly fruitful for understanding how and why diffusion models generate images that remain visually plausible while differing significantly from their training images. In particular, as explained in (Kamb & Ganguli, 2024) and others, e.g., (Ambrogioni, 2023), theory suggests that if our score matching were optimal, we would only be able to recover training samples through our diffusion process. However, as shown by Kamb & Ganguli, (2024), in diffusion models where the score is parametrized by a simple CNN, the inductive biases of the CNN itself (translation equivariance and locality) allow the model to generate samples that globally do not match any training samples, but are rather patch-wise `mosaics'. Despite the widespread use of UNet architectures with self‐attention as the score backbone in diffusion models, the theoretical role of attention in score networks remains largely unexplored. In this work, we take a preliminary step in this direction to extend this theory to the case of diffusion models whose score is parametrized by a CNN with a final self-attention layer. We show that our theory suggests that self-attention will induce a globally image-consistent arrangement of local features beyond the patch-level in generated samples, and we verify this behavior empirically on a carefully crafted dataset.
{:.note title="Abstract"}
Emma Finn, **T. Anderson Keller**, Manos Theodosis, Demba E. Ba
{:.note title="Authors"}

*Accepted at [3rd Workshop on High-dimensional Learning Dynamics (HiLD) @ ICML 2025](https://icml.cc/virtual/2025/workshop/39953) (Poster)* \\
*Paper:* <https://openreview.net/forum?id=xMqRYKvrD1>
{:.note title="Full Paper"}

<!--
{:.note title="Code"}
<add link here if/when available>
-->

- Table of Contents
{:toc}