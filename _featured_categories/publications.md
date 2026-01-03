---
layout: page
title: Publications

# IMPORTANT: we still want the posts from the `research` category:
slug: research
sitemap: false
---

<!-- {% assign pubs = site.categories[page.slug] | where_exp: "p", "p.citation" | sort: "date" | reverse %} -->
<!-- {% assign last_year = "" %} -->

<!-- {% for post in pubs %} -->
<!-- - {{ post.citation }} ([summary]({{ post.url | relative_url }})) -->
<!-- [{{ post.citation | default: post.title }}]({{ post.url | relative_url }}) -->
<!-- {% endfor %} -->

{% assign pubs = site.categories[page.slug] | where_exp: "p", "p.citation" | sort: "date" | reverse %}

<ul class="pub-list">
{% for post in pubs %}
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