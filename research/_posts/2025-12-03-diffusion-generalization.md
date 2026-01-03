---
layout: post
title: "From Extrapolation to Generalization: How Conditioning Transforms Symmetry Learning in Diffusion Models"
<!-- image: /assets/img/research/diffusion-generalization/teaser.png -->
sitemap: false
comments: true

venue: "NeurIPS '25"
award: "NeurReps <br> Workshop"
citation: "S. Bharthulwar, <strong>T. A. Keller</strong>, M. Theodosis, and D. E. Ba (2025). <i>From Extrapolation to Generalization: How Conditioning Transforms Symmetry Learning in Diffusion Models</i>. In: NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations."
paper_url: "https://openreview.net/forum?id=UI82R3lwar"
---
![Conditioning for Symmetry Learning](/assets/img/research/diffusion-generalization/teaser.png){:.lead width="500" height="320" loading="lazy"}
Conditioning on group elements factorizes symmetry learning into low-dimensional **function generalization**, yielding dramatic improvements on **held-out symmetries**.
{:.figcaption}

When trained on data with missing symmetries, diffusion models face a fundamental challenge: how can they generate samples respecting symmetries they have never observed? We prove that this failure stems from the structure of the learning problem itself. Unconditional models must satisfy a global equivariance constraint, coupling all group elements into a single optimization that requires high-dimensional data extrapolation across gaps. In contrast, conditioning on group elements factorizes this into  independent problems, transforming the task into low-dimensional function generalization. Our theory predicts—and experiments confirm—that this simple change yields 5-10× error reduction on held-out symmetries. On synthetic 2D rotation tasks, conditional models maintain low error even with 300° gaps while unconditional models collapse catastrophically. We further suggest that topology-aware group embeddings may help improve this generalization by ensuring smoother functions over the group manifold.
{:.note title="Abstract"}
Sid Bharthulwar, **T. Anderson Keller**, Manos Theodosis, Demba E. Ba
{:.note title="Authors"}

*Accepted at [Symmetry and Geometry in Neural Representations (NeurReps) @ NeurIPS 2025](https://neurips.cc/virtual/2025/workshop/109551) (Poster)* \\
*Paper:* <https://openreview.net/forum?id=UI82R3lwar>
{:.note title="Full Paper"}

<!--
{:.note title="Code"}
<add link here if/when available>
-->

- Table of Contents
{:toc}
