---
layout: post
title: Learning Artistic Signatures -- Symmetry Discovery and Style Transfer
<!-- image: /assets/img/research/artistic-signatures/teaser.png -->
sitemap: false
comments: true

venue: "ArXiv '24"
award: "Preprint"
citation: "E. Finn, <strong>T. A. Keller</strong>, E. Theodosis, and D. E. Ba (2024). <i>Learning Artistic Signatures: Symmetry Discovery and Style Transfer</i>. arXiv: 2412.04441 [cs.CV]."
paper_url: "https://arxiv.org/abs/2412.04441"
---
![Artistic Signatures](/assets/img/research/artistic-signatures/teaser.png){:.lead width="500" height="320" loading="lazy"}
We propose a definition of artistic style that combines **local texture** with **global symmetries** and show that learned symmetries help predict movements and quantify stylistic similarity across artists.
{:.figcaption}

Despite nearly a decade of literature on style transfer, there is no undisputed definition of artistic style. State-of-the-art models produce impressive results but are difficult to interpret since, without a coherent definition of style, the problem of style transfer is inherently ill-posed. Early work framed style-transfer as an optimization problem but treated style as a measure only of texture. This led to artifacts in the outputs of early models where content features from the style image sometimes bled into the output image. Conversely, more recent work with diffusion models offers compelling empirical results but provides little theoretical grounding. To address these issues, we propose an alternative definition of artistic style. We suggest that style should be thought of as a set of global symmetries that dictate the arrangement of local textures. We validate this perspective empirically by learning the symmetries of a large dataset of paintings and showing that symmetries are predictive of the artistic movement to which each painting belongs. Finally, we show that by considering both local and global features, using both Lie generators and traditional measures of texture, we can quantitatively capture the stylistic similarity between artists better than with either set of features alone. This approach not only aligns well with art historians' consensus but also offers a robust framework for distinguishing nuanced stylistic differences, allowing for a more interpretable, theoretically grounded approach to style transfer.
{:.note title="Abstract"}
Emma Finn, **T. Anderson Keller**, Emmanouil Theodosis, Demba E. Ba
{:.note title="Authors"}
*Paper:* <https://arxiv.org/abs/2412.04441>
{:.note title="Full Paper"}

<!--
{:.note title="Code"}
<add link here if/when available>
-->

- Table of Contents
{:toc}
