---
layout: post
title: Traveling Waves Integrate Spatial Information Through Time
<!-- image: /assets/img/research/waves-integrate-info/teaser.png -->
sitemap: false
comments: true

venue: "CCN '25"
award: "Oral (Top 7%)"
citation: "M. Jacobs, R. C. Budzinski, L. Muller, D. E. Ba, and <strong>T. A. Keller</strong> (2025). <i>Traveling Waves Integrate Spatial Information Through Time</i>. In: Conference on Cognitive Computational Neuroscience (CCN). Oral presentation, Top 7%."
paper_url: "https://openreview.net/forum?id=QEzqo546V5"

research_group: "Modeling Spatiotemporal Neural Dynamics"
---
![Traveling Waves](/assets/img/research/waves-integrate-info/teaser.png){:.lead width="500" height="320" loading="lazy"}
We show how learned **traveling-wave** recurrent dynamics can integrate global spatial context over time, yielding strong performance on tasks like semantic segmentation with fewer parameters than non-local baselines.
{:.figcaption}

Traveling waves of neural activity are widely observed in the brain, but their precise computational function remains unclear. One prominent hypothesis is that they enable the transfer and integration of spatial information across neural populations. However, few computational models have explored how traveling waves might be harnessed to perform such integrative processing. Drawing inspiration from the famous “Can one hear the shape of a drum?” problem -- which highlights how normal modes of wave dynamics encode geometric information -- we investigate whether similar principles can be leveraged in artificial neural networks. Specifically, we introduce convolutional recurrent neural networks that learn to produce traveling waves in their hidden states in response to visual stimuli, enabling spatial integration. By then treating these wave-like activation sequences as visual representations themselves, we obtain a powerful representational space that outperforms local feed-forward networks on tasks requiring global spatial context. In particular, we observe that traveling waves effectively expand the receptive field of locally connected neurons, supporting long-range encoding and communication of information. We demonstrate that models equipped with this mechanism solve visual semantic segmentation tasks demanding global integration, significantly outperforming local feed-forward models and rivaling non-local U-Net models with fewer parameters. As a first step toward traveling-wave-based communication and visual representation in artificial networks, our findings suggest wave-dynamics may provide efficiency and training stability benefits, while simultaneously offering a new framework for connecting models to biological recordings of neural activity.
{:.note title="Abstract"}
Mozes Jacobs, Roberto C. Budzinski, Lyle Muller, Demba E. Ba, **T. Anderson Keller**
{:.note title="Authors"}

*Accepted at [CCN 2025](https://2025.ccneuro.org) (Oral presentation, Top 7%)* \\
*Paper:* <https://openreview.net/forum?id=QEzqo546V5>
{:.note title="Full Paper"}


{:.note title="Code"}
<https://github.com/KempnerInstitute/traveling-waves-integrate>


- Table of Contents
{:toc}


## Tweet-print
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In the physical world, almost all information is transmitted through traveling waves -- why should it be any different in your neural network?<br><br>Super excited to share recent work with the brilliant <a href="https://twitter.com/mozesjacobs?ref_src=twsrc%5Etfw">@mozesjacobs</a>: &quot;Traveling Waves Integrate Spatial Information Through Time&quot;<br>1/14 <a href="https://t.co/Bs4UKR7j21">pic.twitter.com/Bs4UKR7j21</a></p>&mdash; Andy Keller (@t_andy_keller) <a href="https://twitter.com/t_andy_keller/status/1899154774227878250?ref_src=twsrc%5Etfw">March 10, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>