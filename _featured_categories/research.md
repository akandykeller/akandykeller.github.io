---
# Featured tags need to have either the `list` or `grid` layout (PRO only).
layout: page

# The title of the tag's page.
title: Research Statement

# The name of the tag, used in a post's front matter (e.g. tags: [<slug>]).
slug: research

# (Optional) Write a short (~150 characters) description of this featured tag.
# description: >
#   Summarized posts describing Andy Keller's publications.
# (Optional) You can disable grouping posts by date.
no_groups: false
# Exclude this example category from the sitemap.
# DON'T USE THIS SETTING IN YOUR CATEGORIES!
sitemap: false

groups:
  - Time-Parameterized Symmetries
  - Modeling Spatiotemporal Neural Dynamics
  - Learning Latent Symmetries

---


<!-- At a high level, I’m interested in a simple question: **how do intelligent systems learn representations that stay reliable when the world changes?** In everyday life, the sensory stream is full of transformations — objects move, viewpoints shift, and context drifts over time — and yet biological intelligence remains remarkably data-efficient and robust. My research explores the hypothesis that part of this advantage comes from *geometric and dynamical structure*: the brain’s tendency to organize computation using its natural physical constraints like [local connectivity](/research/2022-12-20-locornn), [topographic maps](/research/2021-10-23-class-cluster-modeling), [oscillations](2025-12-07-kuramoto-diffusion), and [traveling-wave dynamics](/research/2023-06-23-waves).

Concretely, I develop learning algorithms and neural network architectures that are *meaningfully structured with respect to real-world transformations*. A recurring theme in my work is that when you build the **right internal scaffold** (e.g., [topographic organization](/research/2021-09-03-tvae) or [wave-like recurrent dynamics](/research/2025-08-15-waves-integrate)), networks often learn to map complex sensory transformations onto that scaffold, making future inputs more predictable and improving generalization. More recently, I’ve worked on [**flow equivariance**](/research/2025-12-08-fernn) — a way of enforcing *time-parameterized* symmetries in sequence models by making the *dynamics themselves* symmetric with respect to different moving reference frames, with applications to [partially observed world modeling](/research/2025-12-09-flowm).

In the long term, my goal is to help develop a predictive theory of structured generalization simultaneouslly guides the design of the next generation of learning systems, and sharpens our understanding of organizing principles in natural intelligence. -->


Information processing systems in the real world must compute reliably under a diversity of stimulus transformations; in vision alone, these include changes in viewpoint, lighting, and appearance. To date, most artificial learning systems achieve robustness primarily through increased scale of both data and parameters, rather than ingrained structure. Natural systems, embedded in a world with physical constraints, have no such luxury; they must generalize systematically despite finite data, finite energy, and finite time.

In learning theory, such data efficiency is governed by inductive biases: a priori constraints that restrict the space of solutions a system can represent [(Wolpert 1996)](https://direct.mit.edu/neco/article-abstract/8/7/1341/6016/The-Lack-of-A-Priori-Distinctions-Between-Learning?redirectedFrom=fulltext). In artificial neural networks, many of the most powerful inductive biases arise from symmetry and geometry -- when a model is constrained to respect the abstract structure of transformations in its inputs, it can generalize predictably far beyond its training distribution with dramatically fewer examples. 

The canonical case is the convolutional layer, notably inspired by biology, which builds translation symmetry into vision models [(Fukushima, 1980)](https://link.springer.com/article/10.1007/BF00344251). Congruently, modern neuroscience continues to reveal increasing geometric structure in biological computation and connectivity. From topographic maps and toroidal grid codes, to ring attractor circuits and low-dimensional population manifolds, geometric structure appears to be a central design principle in the blueprint for natural intelligence ([Zhang 1996](https://doi.org/10.1523/JNEUROSCI.16-06-02112.1996); [Churchland et al. 2012](https://doi.org/10.1038/nature11129); [Gardner et al. 2022](https://doi.org/10.1038/s41586-021-04268-7)).

**My research investigates the hypothesis that generalizable symmetric and geometric inductive biases are fundamental to natural intelligence, and seeks to discover the computational primitives that implement them.** 

In particular, my recent research to date falls into three themes: 
1. formalizing the notion of **time-parameterized symmetries unique to recurrent computation**
2. evaluating **the computational implications of natural spatiotemporal dynamics**, and
3. modeling how natural systems leverage **the underlying low-dimensional geometry of high-dimensional data.**

{% assign all = site.categories[page.slug] %}

{% for g in page.groups %}
## {{ g }}

{% assign items = all | where: "research_group", g | sort: "date" | reverse %}

<ul class="pub-list">
{% for post in items %}
  <li class="pub-row">
    <span class="pub-venue">
      {{ post.venue | default: "" }}
      {% if post.award %}
        <span class="pub-award">{{ post.award }}</span>
      {% endif %}
    </span>
    <span class="pub-cite">
      <a href="{{ post.url | relative_url }}">{{ post.citation | default: post.title }}</a>
    </span>
  </li>
{% endfor %}
</ul>

{% endfor %}

<!-- {% assign other = all | where_exp: "p", "p.research_group == nil" | sort: "date" | reverse %}
{% if other.size > 0 %}
## Other

<ul class="pub-list">
{% for post in other %}
  <li class="pub-row">
    <span class="pub-venue">
      {{ post.venue | default: "" }}
      {% if post.award %}
        <span class="pub-award">{{ post.award }}</span>
      {% endif %}
    </span>
    <span class="pub-cite">
      <a href="{{ post.url | relative_url }}">{{ post.citation | default: post.title }}</a>
    </span>
  </li>
{% endfor %}
</ul>
{% endif %} -->