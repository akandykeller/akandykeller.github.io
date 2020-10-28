---
layout: post
title: Self Normalizing Flows
<!-- image: /assets/img/snf_overview.png -->
<!-- srcset:
    3917w: /assets/img/snf_overview@0,5x.png
    1958w:  /assets/img/snf_overview@0,25x.png
    979w:  /assets/img/snf_overview@0,125x.png -->
sitemap: false
---
<!-- ![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"} -->
![Full-width image](/assets/img/overview_long.png){:.lead width="800" height="100" loading="lazy"}
Overview of self normalizing flows. A matrix $$\mathbf{W}$$ transforms $$\mathbf{x}$$ to $$\mathbf{z}$$. The matrix $$\mathbf{R}$$ is constrained to approximate the inverse of $$\mathbf{W}$$ with a reconstruction loss $$\mathcal{E}$$. The likelihood is efficiently optimized by approximating the gradient of the log Jacobian determinant with the learned inverse.
{:.figcaption}

Efficient gradient computation of the Jacobian determinant term is a core problem of the normalizing flow framework. Thus, most proposed flow models either restrict to a function class with easy evaluation of the Jacobian determinant, or an efficient estimator thereof. However, these restrictions limit the performance of such density models, frequently requiring significant depth to reach desired performance levels. In this work, we propose *Self Normalizing Flows*, a flexible framework for training normalizing flows by replacing expensive terms in the gradient by learned approximate inverses at each layer. This reduces the computational complexity of each layer's exact update from $$\mathcal{O}(D^3)$$ to $$\mathcal{O}(D^2)$$, allowing for the training of flow architectures which were otherwise computationally infeasible, while also providing efficient sampling.  We show experimentally that such models are remarkably stable and optimize to similar data likelihood values as their exact gradient counterparts, while surpassing the performance of their functionally constrained counterparts.
{:.note title="Abstract"}
**T. Anderson Keller**, [Jorn Peters](http://jornpeters.nl), [Priyank Jaini](https://cs.uwaterloo.ca/~pjaini/home/), [Emiel Hoogeboom](https://ehoogeboom.github.io/), [Patrick Forré](https://www.uva.nl/en/profile/f/o/p.d.forre/p.d.forre.html), [Max Welling](https://staff.fnwi.uva.nl/m.welling/)
{:.note title="Authors"}
<!-- *ArXiv*: [https://arxiv.org/abs/1908.09257](https://arxiv.org/abs/1908.09257) \\ -->
*Paper*: [https://akandykeller.github.io/publications/Self_Normalizing_Flows.pdf](https://akandykeller.github.io/publications/Self_Normalizing_Flows.pdf) \\
*Submitted to:* [Beyond Backpropagation](https://beyondbackprop.github.io/) workshop at NeurIPS 2020 
{:.note title="Full Paper"}

<!-- {:.lead} -->

- Table of Contents
{:toc .large-only}





## Motivation
The framework of normalizing flows [33] allows for powerful exact density estimation through the change of variables formula [30]. A significant challenge with this approach is the Jacobian determinant in the objective, which is generally expensive to compute. A significant body of work has therefore focused on methods to evaluate the Jacobian determinant efficiently by limiting the expressivity of the transformation. Two classes of functions have been proposed to achieve this: i) those with triangular Jacobians, such that the determinant only depends on the diagonal [7, 27, 18], and ii) those which are Lipschitz continuous such that Jacobian determinant can be approximated at each iteration through an infinite series [2, 12]. The drawback of both of these approaches is that they rely on strong functional constraints.

An insight which hints at a solution to this challenge is that the derivative of the log Jacobian determinant yields the inverse of the Jacobian itself. Recently, [13] leveraged this fact, and related work on Independant Component Analysis (ICA) [8, 1], to avoid computation of the inverse by using the natural gradient. Unfortunately, this method does not extend simply to convolutional layers. 

In this work, we rely on this same insight and propose a new framework, called self normalizing flows, where flow components learn to approximate their own inverse through a self-supervised layer-wise reconstruction loss. We define a new density model as a mixture of the probability induced by both the forward and inverse transformations, and show how both transformations can be updated symmetrically using their respective inverses directly in the gradient. Ultimately, this reduces the computational complexity of each training step from $$\mathcal{O}(LD^3)$$ to $$\mathcal{O}(LD^2)$$ where $$L$$ and $$D$$ are the numbers of layers and the dimensionality of each layer respectively. 

The idea of greedy layer-wise learning has been explored in many forms for training neural networks [5, 14, 25]. One influential class of work uses stacked auto-encoders, or deep belief networks, for pre-training or representation learning [5, 14, 34, 21]. Our work leverages similar models into a modern flow framework by introducing the inverse weight matrix directly as an approximation of expensive terms in the gradient. Another class of work uses similar models to address the biological implausibility of backpropagation. Target propagation [22, 3, 23, 4] addresses the so-called weight transport problem by training auto-encoders at each layer of a network and using these learned feedback weights to propagate ‘targets’ to previous layers. Synthetic gradients [17] serve to alleviate the ‘timing problem’ of biological backpropagation by modelling error gradients at intermediate layers, and using these approximate gradients directly. Our method can also be seen in this light, and takes inspiration from all of these approaches. Specifically, our method can be viewed as a hybrid of target propagation and backpropagation [24, 35, 31] particularly suited to unsupervised density estimation in the normalizing flow framework. The novelty of our approach lies in the use of the inverse weights directly in the update, rather than in the backward propagation of updates.


## A General Framework for Self Normalizing Flows

Given an observation $$\mathbf{x} \in \mathbb{R}^{D}$$, it is assumed that $$\mathbf{x}$$ is generated from an underlying real vector $$\mathbf{z} \in \mathbb{R}^D$$, through a series of invertible mappings $$g_0 \circ g_1 \circ \dots g_K (\mathbf{z}) = \mathbf{x}$$, where $$\mathbf{z} \sim p_{\mathbf{Z}}(\mathbf{z})$$, $$\mathbf{z} = f_K \circ f_{K-1} \circ \dots f_0(\mathbf{x})$$ and $$f_k = g_k^{-1}$$. We denote these compositions of functions as simply $$g$$ and $$f$$ respectively, i.e. $$g \circ f(\mathbf{x}) = \mathbf{x}$$. The base distribution $$p_{\mathbf{Z}}$$ is usually chosen to be simple to compute, such as a standard Gaussian. The probability density $$p_{\mathbf{X}}$$ can be computed using the change of variables formula:
<br/>

$$\begin{equation}
\begin{split}
    p_{\mathbf{X}}(\mathbf{x}) & = p_{\mathbf{Z}}(\mathbf{z}) \left|\frac{\partial \mathbf{z}}{\partial\mathbf{x}}\right| = p_{\mathbf{Z}}\left(g^{-1}(\mathbf{x})\right) \left|\mathbf{J}_{g^{-1}}\right| = p_{\mathbf{Z}}\big(f(\mathbf{x})\big) \left|\mathbf{J}_f\right|\\
\end{split}
\end{equation}$$

\\
Where the change of volume term $$\left| \mathbf{J}_f \right| = \left|\frac{\partial f(\mathbf{x})}{\partial\mathbf{x}}\right|$$ is the determinant of the Jacobian of the transformation between $$\mathbf{z}$$ and $$\mathbf{x}$$, evaluated at $$\mathbf{x}$$. Typically, only the forward functions $$f_k$$ are defined and parameterized, and the inverses $$g_k$$ are computed exactly when needed. The log-likelihood of the observations is then simultaneously maximized, with respect to a given $$f_k$$'s vector of parameters $$\theta_k$$, for all $$k$$, requiring the gradient. Using the identity $$\frac{\partial}{\partial \mathbf{J}} \log |\mathbf{J}| = {\mathbf{J}^{-T}}$$ yields: 

$$
    \begin{equation}
    \begin{split}
    \frac{\partial}{\partial  \theta_k} \log p^{f}_{\mathbf{X}}(\mathbf{x}) = \frac{\partial}{\partial \theta_k} \log p_{\mathbf{Z}}(f(\mathbf{x})) + \frac{\partial\left(\mathrm{vec}\ \mathbf{J}_{f}\right)^T}{\partial \theta_k} \left( \mathrm{vec}\ \mathbf{J}_{f}^{-T} \right)
    \end{split}
    \end{equation}
$$

where $$\mathrm{vec}$$ is vectorization through column stacking \cite{matrix_derivative}, and $$\theta_k$$ is a column vector. 
In this work, in order to avoid the inverse Jacobian in the gradient, we instead propose to define and parameterize *both* the forward and  inverse functions $$f_k$$ and $$g_k$$ with parameters $$\theta_k$$ and $$\gamma_k$$ respectively. We then propose to constrain the parameterized inverse $$g_k$$ to be approximately equal to the true inverse $$f^{-1}_k$$ though a layer-wise reconstruction loss. We can thus define our maximization objective as the mixture of the log-likelihoods induced by both models minus the reconstruction penalty constraint, i.e.:

$$\begin{equation}
\begin{split}
    \mathcal{L}(\mathbf{x}) & = \frac{1}{2} \log p^{f}_{\mathbf{X}}(\mathbf{x}) + \frac{1}{2} \log p^{g}_{\mathbf{X}}(\mathbf{x}) - \lambda \sum_{k=0}^K ||g_k \left( f_k (\mathbf{h}_k) \right) - \mathbf{h}_k ||^2_2
\end{split}
\end{equation}$$

where $$\mathbf{h}_k = \mathrm{gradient\_stop}(f_{k-1} \circ ... f_0 (\mathbf{x}))$$ is the output of function $$f_{k-1}$$ with the gradients blocked such that only $$g_k$$ and $$f_k$$ receive gradients from the reconstruction loss at layer $$k$$. We see that when $$f = g^{-1}$$ exactly, this is equivalent to the traditional normalizing flow framework. By the inverse function theorem, we know that the inverse of the Jacobian of an invertible function is given by the Jacobian of the inverse function, i.e. $$\mathbf{J}_f^{-1}(\mathbf{x}) = \mathbf{J}_{f^{-1}}(\mathbf{z})$$. Therefore, we see that with the above parameterization and constraint, we can approximate both the change of variables formula, and the gradients for both functions, in terms of the Jacobians of the respective inverse functions. Explicitly:

$$\begin{equation}
    \begin{split}
    \frac{\partial}{\partial  \theta_k} \log p^{f}_{\mathbf{X}}(\mathbf{x}) \approx \frac{\partial}{\partial \theta_k} \log p_{\mathbf{Z}}(f(\mathbf{x})) + \frac{\partial \left(\mathrm{vec}\ \mathbf{J}_{f}\right)^T}{\partial \theta_k}\left(\mathrm{vec}\ \mathbf{J}_{g}^T\right)
    \end{split}
\end{equation}$$

$$\begin{equation}
    \begin{split}
    \frac{\partial}{\partial  \gamma_k} \log p^{g}_{\mathbf{X}}(\mathbf{x}) \approx \frac{\partial}{\partial \gamma_k} \log p_{\mathbf{Z}}(g^{-1}(\mathbf{x})) - \frac{\partial \left(\mathrm{vec}\ \mathbf{J}_{g}\right)^T}{\partial \gamma_k}\left(\mathrm{vec}\ \mathbf{J}_{f}^T\right)
    \end{split}
\end{equation}$$

\\
where Equation $$\ref{grad_g}$$ follows from the derivation of Equation $$\ref{grad_f}$$ and the application of the derivative of the inverse. We note that the above approximation requires that the Jacobians of the functions are approximately inverses *in addition* to the functions themselves being approximate inverses. For the models presented in this work, this property is obtained for free since the Jacobian of a linear mapping is the matrix representation of the map itself. % However, for more complex mappings, this may not be exactly the case and should be constrained explicitly.

Although there are no known convergence guarantees for such a method, we observe in practice that, with sufficiently large values of $$\lambda$$, most models quickly converge to solutions which maintain the desired constraint. Figure $$\ref{fig:samples}$$ gives an example of samples from the base distribution $$p_{\mathbf{Z}}$$ passed through both the true inverse $$f^{-1}$$ (top) and the learned approximate inverse $$g$$ (bottom) to generate samples from $$p_{\mathbf{X}}$$. As can be seen, the approximate inverse appears to be a very close match to the true inverse. The details of the model which generated these samples are in Appendix \ref{appendix:architectures}. 

## Examples
### Self Normalizing Fully Connected Layer
As a specific case of the above model, we consider a single fully connected layer, as exemplified in Figure $$\ref{fig:overview}$$. Let $$f(\mathbf{x}) = \mathbf{W} \mathbf{x} = \mathbf{z}$$, and $$g(\mathbf{z}) = \mathbf{R} \mathbf{z}$$, such that $$\mathbf{W}^{-1} \approx \mathbf{R}$$. Taking the gradients of Equation $$\ref{mixture}$$ for this model, and applying Equations $$\ref{grad_f}$$ and $$\ref{grad_g}$$, we get the following approximate gradients (see \ref{appendix:fc_derivation} for details):


$$\begin{equation}
    \begin{split}
    \frac{\partial}{\partial  \mathbf{W}} \mathcal{L}(\mathbf{x}) & = \frac{1}{2} \underbrace{\left(\frac{\partial}{\partial \mathbf{W}} \log p_{\mathbf{Z}}(\mathbf{W} \mathbf{x}) + \mathbf{W}^{-T}\right)}_{\textstyle \approx \delta_{\mathbf{z}} \mathbf{x}^T + \mathbf{R}^T} \ 
    - \frac{\partial}{\partial  \mathbf{W}} \lambda \mathcal{E}
    \end{split}
\end{equation}$$

$$\begin{equation}
    \begin{split}
    \frac{\partial}{\partial  \mathbf{R}} \mathcal{L}(\mathbf{x}) & = \frac{1}{2} \underbrace{\left(\frac{\partial}{\partial \mathbf{R}} \log p_{\mathbf{Z}}(\mathbf{R}^{-1} \mathbf{x}) - \mathbf{R}^{-T} \right)}_{\textstyle \approx - \delta_{\mathbf{x}} \mathbf{z}^T - \mathbf{W}^T}\ - \frac{\partial}{\partial  \mathbf{R}} \lambda \mathcal{E} \\
    \end{split}
\end{equation}$$

where $$\mathcal{E}$$ denotes the reconstruction error,  $$\frac{\partial}{\partial  \mathbf{W}} \lambda \mathcal{E}  = 2\lambda \mathbf{R}^T(\hat{\mathbf{x}} - \mathbf{x}) \mathbf{x}^T$$, $$\frac{\partial}{\partial  \mathbf{R}} \lambda \mathcal{E} = 2\lambda (\hat{\mathbf{x}} - \mathbf{x})\mathbf{z}^T$$, $$\hat{\mathbf{x}} = \mathbf{R}\mathbf{W}\mathbf{x}$$, $$\delta_{\mathbf{z}} = \frac{\partial \log p_{\mathbf{Z}}(\mathbf{z})}{\partial \mathbf{z}}$$, and $$\delta_{\mathbf{x}} = \frac{\partial \log p_{\mathbf{Z}}(\mathbf{z})}{\partial \mathbf{x}}$$ are ordinarily computed by backpropagation. We observe that by using such a self normalizing layer, the gradient of the log-determinant of the Jacobian term is approximately given by the weights of the inverse transformation, sidestepping computation of the Jacobian determinant and all matrix inverses.

### Self Normalizing Convolutional Layer
To construct a self normalizing convolutional layer, let $$f(\mathbf{x}) = \mathbf{w} \star \mathbf{x} = \mathbf{z}$$, and $$g(\mathbf{z}) = \mathbf{r} \star \mathbf{z}$$, such that $$f^{-1} \approx g$$, where $$\star$$ is the convolution operation. We first note that the inverse of a convolution operation is not necessarily another convolution. However, for sufficiently large $$\lambda$$, we observe that $$f$$ is simply restricted to the class of convolutions which is approximately invertible by a convolution. As we derive fully in Appendix \ref{appendix:conv_derivation}, the approximate self normalizing gradients with respect to a convolutional kernel $$\mathbf{w}$$, and corresponding inverse kernel $$\mathbf{r}$$, are given by:

$$\begin{equation}
    \begin{split}
    \frac{\partial }{\partial \mathbf{w}} \log p^{f}_{\mathbf{X}}(\mathbf{x}) & \approx \delta_{\mathbf{z}} \star \mathbf{x} + \mathrm{flip}(\mathbf{r}) \odot \mathbf{m} \\
    \end{split}
\end{equation}$$

$$\begin{equation}
    \begin{split}
    \frac{\partial }{\partial \mathbf{r}} \log p^{g}_{\mathbf{X}}(\mathbf{x}) & \approx -\delta_{\mathbf{x}} \star \mathbf{z} - \mathrm{flip}(\mathbf{w}) \odot \mathbf{m} \\
    \end{split}
\end{equation}$$

where $$\mathrm{flip}(\mathbf{r})$$ corresponds to the kernel which achieves the transpose convolution and is given by swapping the input and output channels, and mirroring the spatial dimensions. The constant $$\mathbf{m}$$ is given by the number of times each element of the kernel $$\mathbf{w}$$ is present in the matrix form of convolution. The gradients for the reconstruction loss are then added to these.


## Discussion
We see that the above framework yields an efficient update rule for flow-based models which appears to perform similarly to the exact gradient. The limitations include that that evaluation of the exact log-likelihood is still expensive, however this cost can be amortized over many samples, and that there are no known optimization guarantees (see \ref{appendix:limitations}). In future work we intend to see how this framework scales to larger models, and how it could be combined with methods such as target propagation.


footnote [^1]

[^1]: footnote