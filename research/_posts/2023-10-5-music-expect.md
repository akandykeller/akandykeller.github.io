---
layout: post
title: Deep Generative Models of Music Expectation
<!-- image: /assets/img/research/waves/.gif -->
sitemap: false
comments: true

venue: "NeurIPS '23"
award: "Workshop"
citation: "N. L. Masclef and <strong>T. A. Keller</strong> (2023). <i>Deep Generative Models of Music Expectation</i>. arXiv: 2310.03500 [cs.SD]."
paper_url: "https://arxiv.org/abs/2310.03500"
---
<!-- ![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"} -->
![Orientation Columns](/assets/img/research/music-expect/wundt.png){:.lead width="400" height="400" loading="lazy"}
We show the expected 'inverted U-shaped' relationship between human song ratings and music likelihood under a modern diffusion model.
{:.figcaption}
 

A prominent theory of affective response to music revolves around the concepts of surprisal and expectation. In prior work, this idea has been operationalized in the form of probabilistic models of music which allow for precise computation of song (or note-by-note) probabilities, conditioned on a 'training set' of prior musical or cultural experiences. To date, however, these models have been limited to compute exact probabilities through hand-crafted features or restricted to linear models which are likely not sufficient to represent the complex conditional distributions present in music. In this work, we propose to use modern deep probabilistic generative models in the form of a Diffusion Model to compute an approximate likelihood of a musical input sequence. Unlike prior work, such a generative model parameterized by deep neural networks is able to learn complex non-linear features directly from a training set itself. In doing so, we expect to find that such models are able to more accurately represent the 'surprisal' of music for human listeners. From the literature, it is known that there is an inverted U-shaped relationship between surprisal and the amount human subjects 'like' a given song. In this work we show that pre-trained diffusion models indeed yield musical surprisal values which exhibit a negative quadratic relationship with measured subject 'liking' ratings, and that the quality of this relationship is competitive with state of the art methods such as IDyOM. We therefore present this model a preliminary step in developing modern deep generative models of music expectation and subjective likability.
{:.note title="Abstract"}
Ninon Lizé Masclef and **T. Anderson Keller** 
{:.note title="Authors"}
*Accepted at [NeurIPS Workshop on Machine Learning for Audio](https://neurips.cc/virtual/2023/79727) (Poster)* \\
*Paper:* <https://arxiv.org/abs/2310.03500>
{:.note title="Full Paper"}
<!-- [LocoRNN Github](https://github.com/q2w4/LocoRNN)
{:.note title="Code"}
  -->

<!-- {:.lead} -->

- Table of Contents
{:toc}