---
layout: post
title: Modeling Category-Selective Cortical Regions with Topographic VAEs
<!-- image: /assets/img/research/tvae/comm_diag.png -->
sitemap: false
comments: true
---
<!-- ![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"} -->
![Face Clusters](/assets/img/research/clusters/Selectivity_FC6.png){:.lead width="3460" height="1331" loading="lazy"}
Measurement of selectivity of top-layer neurons to images of _Faces_ vs. images of _Objects_. The baseline pretrained Alexnet model (left) has randomly organized selectivity as expected. We see the Topographic VAE (middle) yeilds spatially dense clusters of neurons selective to images of faces, reminiscent of the 'face patches' observed in the primate cortex. The TVAE clusters are seen to be qualitatively similar to those produced by the supervised TDANN model of Lee et al. (2020) (right) without the need for class-labels during training. 
{:.figcaption}
 

Category-selectivity in the brain describes the observation that certain spatially localized areas of the cerebral cortex tend to respond robustly and selectively to stimuli from specific limited categories. One of the most well known examples of category-selectivity is the Fusiform Face Area (FFA), an area of the inferior temporal cortex in primates which responds preferentially to images of faces when compared with objects or other generic stimuli. In this work, we leverage the newly introduced Topographic Variational Autoencoder to model of the emergence of such localized category-selectivity in an unsupervised manner. Experimentally, we demonstrate our model yields spatially dense neural clusters selective to faces, bodies, and places through visualized maps of Cohen's d metric. We compare our model with related supervised approaches, namely the TDANN, and discuss both theoretical and empirical similarities. Finally, we show preliminary results suggesting that our model yields a nested spatial hierarchy of increasingly abstract categories, analogous to observations from the human ventral temporal cortex. 
{:.note title="Abstract"}
**T. Anderson Keller***,  [Qinghe Gao*](https://www.tudelft.nl/tnw/over-faculteit/afdelingen/chemical-engineering/about-the-department/product-and-process-engineering/people/phds/qinghe-gao), [Max Welling](https://staff.fnwi.uva.nl/m.welling/)
{:.note title="Authors -- * Equal Contribution"}
*Preprint (under review)*: <https://arxiv.org/abs/2110.13911> 
{:.note title="Full Paper"}
[Github.com/AKAndykeller/CategorySelectiveTVAE](https://github.com/akandykeller/CategorySelectiveTVAE)
{:.note title="Code"}
 

<!-- {:.lead} -->

- Table of Contents
{:toc}

<!-- ## The Fusiform Face Area
I first heard about the Fusiform Face Area (FFA) when watching [Nancy Kanwisher's excellent lectures from her course at MIT: (9.11) The Human Brain](https://www.youtube.com/watch?v=i1pdQjdAndc&list=PLyGKBDfnk-iAQx4Kw9JeVqspbg77sfAK0). The idea that there was a localized region of the cotrtex that responded specifically to faces was perhaps not  -->

<!-- 
To me, one of the most fascinating observations from neuroscience is that of the topographic organization of the brain. Not only are local regions correlated in the semantic meaning of their selectivity (at a variety of levels of abstraction), but the layout and placement of these regions is very similar across individuals. The mystery of how or why these regions are like this has been a primary interest. 
 -->

<!-- 
## What is Localized Category-Selectivity?
Category-selectivity describes the observation that certain localized regions of the cortical surface have been measured to respond preferentially to specific stimuli when compared with a set of alternative control images. It has been measured across a diversity of species through fMRI as well as through observational studies of patients with localized cortical damage. Some of the most prominent examples of category-selective areas include the Fuisform Face Area (FFA), the Parahippocampal Place Area (PPA), and the Extrastriate Body Area (EBA) which respond selectively to faces, places, and bodies respectively.

### Where does it come from?

#### Anatomical Constraints 
Anatomical constraints such as the arrangement and properties of different cell bodies can be observed to vary slightly in different regions of the cortex in loose alignment with category selectivity. The principle of 'wiring length minimization' can additionally be placed in this category, positing that evolutionary pressure has encouraged the brain to reduce the cumulative length of neural connections in order to reduce the costs associated with the volume, building, maintenance, and use of such connections. Computational models which attempt to integrate such wiring length constraints have recently have been observed to yield localized category selectivity such as 'face patches' similar to those of macaque monkeys. (See TDANN, VTC-SOM, Recurrent Models)

#### Redundancy Reduction
A second hypothesis for the emergence of category specialization, which has recently gained increasing empirical support, derives its explanatory power from information theory. Empirical studies have discovered that sufficiently deep convolutional neural networks naturally learn distinct and largely separate sets of features for certain domains such as faces and objects. Specifically, the work of Katharina Dobs et al. (2021), showed that feature maps in the later layers of deep convolutional neural networks can be effectively segregated into object and face features such that lesioning one set of feature maps does not significantly impact performance of the network on classification of the other data domain. Such experiments suggest that the specialization of neurons may simply be an optimal code for representing the natural statistics of the underlying data when given a sufficiently powerful feature extractor. 

### Our Model
Pursuant to these ideas, this work proposes that a single underlying information theoretic principle, namely the principle of redundancy reduction, may account for localized category selectivity while simultaneously serving as a principled unsupervised learning algorithm. Simply, the principle of redundancy reduction states that an optimal coding scheme is one which minimizes the transmission of redundant information. Applied to neural systems, this describes the ideal network as one which has statistically maximally independant activations -- yielding a form of specialization. This idea served as  the impetus for computational frameworks such as Sparse Coding  and Independant Component Analysis (ICA). Interestingly, however, further work showed that features learned by linear ICA models were not entirely independant, but indeed contained correlation of higher order statistics. In response, researchers proposed a more efficient code could be achieved by modeling these residual dependencies with a hierarchical topographic extension to ICA, separating out the higher order 'variance generating' variables, and combining them locally to form topographically organized latent variables. Such a framework shares a striking resemblance to models of divisive normalization, but inversely formulated as a generative model. Ultimately, the features learned by such models were reminiscent of pinwheel structures observed in V1, encouraging multiple comparisons with topographic organization in the biological visual system. 

In this work, we leverage the recently introduced Topographic Variational Autoencoder, a modern instantiation of such a topographic generative model, and demonstrate that it is capable of modeling localized category selectivity as well as higher order abstract organization, guided by a single unsupervised learning principle. We quantitatively validate category selectivity through visualization of Cohen's d effect size metric for different image classes, showing selective clusters for faces, bodies, and places. We compare our model with the supervised wiring cost proxy of Hyodong Lee et al. (denoted TDANN) and demonstrate that our model yields qualitatively similar results with an unsupervised learning rule. Finally, we show preliminary results indicating that our model contains a nested spatial hierarchy of increasingly abstract categories, similar to those observed in the human ventral temporal cortex. -->