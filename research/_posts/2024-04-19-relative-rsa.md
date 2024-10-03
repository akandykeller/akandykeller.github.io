---
layout: post
title: Relative Representations for Model-to-Brain Mappings 
<!-- image: /assets/img/research/wrnn/waves.png -->
sitemap: false
comments: true
---
<!-- ![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"} -->
![Relative_Reps](/assets/img/research/relative_reps/Relative_Reps.png){:.lead width="800" height="400" loading="lazy"}
Relative Representations are a method for mapping points (such as the green circle) from a high dimensional space (left) to a lower dimensional space (right), by represeniting it in a new coordinate system _relative_ to a select set of _anchor points_ (red and blue star). In this work we apply such an idea of _relative representations_ to model-brain mappings and show that it improves interpretability and computational efficiency -- surprisingly model-brain RSA scores are roughly consistent even with as few as 10 randomly selected anchor points (10 dimensions) compared to the original 1000's of dimensions.
{:.figcaption}

Current model-to-brain mappings are computed over thou- sands of features. These high-dimensional mappings are computationally expensive and often difficult to interpret, due in large part to the uncertainty surrounding the re- lationship between the inherent structures of the brain and model feature spaces. Relative representations are a recent innovation from the machine learning literature that allow one to translate a feature space into a new co- ordinate frame whose dimensions are defined by a few select ‘anchor points’ chosen directly from the original input embeddings themselves. In this work, we show that computing model-to-brain mappings over these new coor- dinate spaces yields brain-predictivity scores comparable to mappings computed over full feature spaces, but with far fewer dimensions. Furthermore, since these dimen- sions are effectively the similarity of known inputs to other known inputs, we can now better interpret the structure of our mappings with respect to these known inputs. Ulti- mately, we provide a proof-of-concept that demonstrates the flexibility and performance of these relative represen- tations on a now-standard benchmark of high-level vision and firmly establishes them as a candidate model-to-brain mapping metric worthy of further exploration.
{:.note title="Abstract"}
**T. Anderson Keller***, [Talia Konkle](https://psychology.fas.harvard.edu/people/talia-konkle), and [Colin Conwell](https://colinconwell.github.io)
{:.note title="Authors"}
*Accepted at [CCN 2024](https://2024.ccneuro.org) (Poster)* \\
*Conference Abstract:* <https://2024.ccneuro.org/pdf/492_Paper_authored_AnchorEmbeddings_CCN2024_named.pdf>  
{:.note title="Full Paper"}
 

<!-- {:.lead} -->

- Table of Contents
{:toc}