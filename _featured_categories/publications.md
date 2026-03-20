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
{% assign extra_pubs = site.data.publications | default: empty %}

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

  {% assign injected = extra_pubs | where: "insert_after_title", post.title %}
  {% for pub in injected %}
    <li class="pub-row">
      <span class="pub-venue">
        {{ pub.venue | default: "" }}
        {% if pub.award %}
          <span class="pub-award">{{ pub.award }}</span>
        {% endif %}
      </span>
      <span class="pub-cite">
        {% if pub.url and pub.url != "" %}
          <a href="{{ pub.url }}">{{ pub.citation | default: pub.title }}</a>
        {% else %}
          {{ pub.citation | default: pub.title }}
        {% endif %}
      </span>
    </li>
  {% endfor %}
{% endfor %}

</ul>