---
layout: post
title: Change of Variables
<!-- image: /assets/img/snf_overview.png -->
<!-- srcset:
    3917w: /assets/img/snf_overview@0,5x.png
    1958w:  /assets/img/snf_overview@0,25x.png
    979w:  /assets/img/snf_overview@0,125x.png -->
sitemap: false
---
<!-- ![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"} -->
![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}
The change of variables formula can be seen as normalizng for the change of volume due to a transformation from one space to another.
{:.figcaption}

In this post we introduce a visual analogy to understand the change of variables formula for probability distributions, along with a motivation for why it could be useful. We then relate this to the framework of **normalizing flows**, and the associated training criteria. Although the analogy is not exactly water tight, we believe it is a helpful way to understand probaility distributions, and thus find value in it. 
{:.note title="Abstract"}
T. Anderson Keller
{:.note title="Author"}

<!-- {:.lead} -->

- Table of Contents
{:toc}

## Motivation
Let's take a step back in time (I assume) to when you were a young inquisitive child playing in sandboxes. If you are reading this and you are still a child, my apologies, the world really has changed. 

![Image of You and Gravel](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

Let's also say that you have a brother, and, since your parents love you both equally <font size="1">(haha)</font> both your sandboxes have an exactly equal amount of sand. 

![Image of You and Brother](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

While you are busy studying, your brother plays with his sand and builds some mountains and valleys. When you return, you notice that his sand pile is much taller than yours at some points. 

![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

Being an jealous brother, you are interested in knowing how tall his sand pile is, but you're afraid to get too close since you know your big brother doesn't like it when you touch his sand. How can you measure his sand pile without having access to it?

<!-- You start by taking your sand pile, and distributing it uniformly on a stretchy fabric (like a trampoline). You measure the height of this sand pile to be exacly 1 baby foot everywhere. You then draw some equally baby-foot spaced lines on the bottom of the trampoline.  -->

You start by taking your sand pile, and dumping it on a stretchy fabric (like a trampoline). You measure the height of this sand pile everywhere in baby feet and you record the measurements. You then draw some equally baby-foot spaced lines on the bottom of the trampoline. 

![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

Now, instead of moving the sand, you notice that if you squeeze different parts of the trampoline, it changes the height of the sand! You decide to squeeze in the middle...

![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

You decide to squeeze on the edges...

![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

You squeeze just enough that eventually, your pile looks like your brothers! 

![Change of Variables Overview](/assets/img/blog/cov/cov_small.jpg){:.lead width="2121" height="969" loading="lazy"}

Now, since you haven't touched the sand pile at all (you havent removed or added any sand), you realize that if you can just measure the height of your sand pile now, then you would know how tall your brother's pile is. Alas, since your hands are stuck holding the trampoline, you can't actually measure it directly -- and you're forced to be more clever than that... So, like all good scientists trying to come up with a new law -- [first you guess it!](https://www.youtube.com/watch?v=EYPapE-3FRw) <font size="1">- Richard Feynman</font>

You look at the lines you've drawn, and the height of the pile and you start to notice a relationship. It seems to be that, given the original height of the pile, the height of the resulting pile is directly proportional to the amount you squeezed by. So you guess the following relationship: new-height = old-height * amount-squeezed

But how can you formally define the amount squeezed? To do this, we introduce some definitions. First, let the old pile be called $$\mathbf{Z}$$ and the new pile be called $$\mathbf{X}$$. You then define your squeeze operation $$\mathcal{S}$$ as a transformation of each location $$\mathbf{x}$$ in $$\mathbf{X}$$ to a location $$\mathbf{z}$$ in $$\mathbf{Z}$$. (i.e. By squeezing, you are just moving parts of the trampline around). You then call the height of the old pile at $$\mathbf{z}$$, $$p_{\mathbf{Z}}(\mathbf{z})$$, and the height of the new pile at at $$\mathbf{x}$$ $$p_{\mathbf{X}}(\mathbf{x})$$. Now you're ready!

Well, you use your grid lines! You say the amount squeezed is equal to the space between two lines in the old pile (at the location of interest), divided by the space between two lines in the new pile (at the same corresponding area). If we call these two measurements $$\partial_\mathbf{z}$$ and $$\partial_\mathbf{x}$$ respectively, then $$amount_squeezed = \frac{\partial_\mathbf{z}}{\partial_\mathbf{x}}$$. Finally then, if we call the height of our new pile as $$p(x)$$, and the height of the old pile as $$p(z)$$


## Normalizing Flows
The goal of a probabilistic generative model is to accurately capture the true data generating process.[^1] Popular examples of such models include latent variable models (like Variational Autoencoders) or implicit probabilistic models (like Generative Adverserial Networks). In this work, we focus on another relatively recent technique for learning models of complex probability distributions -- normalizing flows.

Normalizing flows are a powerful class of models which repeatedly apply simple invertible functions to transform basic probability distribtuions into complex target distributions. The primary goal of most work on normalizing flows is thus to develop new classes of functions which are simulatenously *invertible, efficient to train, & maximally flexibile*. For a complete and insightful review of normalizing flows, and prior work in this direction, I highly recommend reading [Normalizing Flows for Probablistic Modeling and Inference (Papamakarios & Nalisnick et al. 2019)](https://arxiv.org/abs/1912.02762). 

In this work, we take a simple alternative approach which achieves the above three desired properties without having to resort to restriced function classes as prior work has done. We propose a new general framework called **Self Normalizing Flows** for the construction and training of normalizing flows which are virtually unconstrained (maximally flexible) while also being incredibly simple to train. Our work takes inspiration from 'feedback' connections observed in the biological cortex, and strives to simultaneously *learn the inverse* of each transformation performed in the flow.[^2] Our key insight is that the normally *very expensive* gradient update required to train the forward transformation is approximately given by the parameters of the learned inverse[^3] -- thereby entirely avoiding the $$\mathcal{O}(D^3)$$ computational complexity normally required for data of dimension $$D$$.

## Background
Given an observed data point $$\mathbf{x} \in \mathbb{R}^{D}$$, it is assumed that $$\mathbf{x}$$ is generated from an underlying (unknown) real vector $$\mathbf{z} \in \mathbb{R}^D$$, through a 'generating' transformation $$g(\mathbf{z}) = \mathbf{x}$$. In the normalizing flow framework, we assume that $$g$$ is invertible, and we denote the inverse as $$f$$, such that $$f = g^{-1}$$ and $$f(\mathbf{x}) = \mathbf{z}$$. Typically, normalizing flows are composed of many of such functions, i.e. $$g(\mathbf{z}) = g_K(g_{k-1}(\dots g_2(g_1(\mathbf{z}))))$$, however in this blog post we will describe flows composed of a single 'step', since the proposed methods naturally extend to compositions.

The main idea behind normalizing flows is that since $$g$$ is assumed to be a powerful transformation, we can transform from an *easy to work with* base distribution $$p_{\mathbf{Z}}$$ (such as a Gaussian), to the true complex data distribution $$p_{\mathbf{X}}$$, without having to know the data distribution directly. This transformation is achieved by the change of variables rule as follows:

<p style="text-align: center;">
$$
\begin{equation}
\begin{split}
    p_{\mathbf{X}}(\mathbf{x}) & = p_{\mathbf{Z}}(\mathbf{z}) \left|\frac{\partial \mathbf{z}}{\partial\mathbf{x}}\right|\\
    \Rightarrow p^g_{\mathbf{X}}(\mathbf{x}) & = p_{\mathbf{Z}}\left(g^{-1}(\mathbf{x})\right) \left|\mathbf{J}_{g^{-1}}\right| \\ 
    \Rightarrow p^f_{\mathbf{X}}(\mathbf{x}) & = p_{\mathbf{Z}}\big(f(\mathbf{x})\big) \left|\mathbf{J}_f\right|\\
\end{split}
\end{equation}
$$
</p>

Where $$\left| \mathbf{J}_f \right| = \left|\frac{\partial f(\mathbf{x})}{\partial\mathbf{x}}\right|$$ is the absolute value of the determinant of the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of the transformation between $$\mathbf{z}$$ and $$\mathbf{x}$$, evaluated at $$\mathbf{x}$$. This term can intuitively be seen as accounting for the change of volume of the transformation between $$\mathbf{z}$$ and $$\mathbf{x}$$. In other words, it is the ratio of the size of an infinitesimally small volume around $$\mathbf{z}$$ divided by the size of the corresponding transformed volume around $$\mathbf{x}$$. We illustrate this simply with the following figure: 
<br /><br />
![change_of_variables](/assets/img/research/snf/cov_small.jpg){: width="2121" height="969" loading="lazy"}
<br /><br />
We see then, if we parameterize a function $$f_{\theta}$$ with a set of parameters $$\theta$$, and pick a base distribution $$p_{\mathbf{Z}}$$, we can use the above equation to maximize the probability of observed data $$\mathbf{x}$$ under our induced distributions $$p_{\mathbf{X}}^f(\mathbf{x})$$. 
<br /><br />
Although in theory this allows us to exactly maximize the parameters of our model to make it a *good* model of the true data distribution, in practice, having to compute the Jacobian determinant $$\left|\mathbf{J}_f\right|$$ or even it's gradient $$\nabla_{\mathbf{J}_f} \left| \mathbf{J}_f \right| = \left(\mathbf{J}_f^{-1}\right)^T$$ makes such an approach intractible for real data (both are $$\mathcal{O}(D^3)$$ and consider even a small 256x256 image has over 65,000 dimensions).


[^1]:  Similarly, the goal of probabilistic generative models can be seen as designing models which are able to generate data which appears to come from the same distribution as real data.
