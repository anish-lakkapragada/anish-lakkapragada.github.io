---
layout: plain
title: blogs
icon: fas fa-pen-nib
order: 2
wide: true
---

{% assign all_pinned = site.posts | where: 'pin', 'true' %}
{% assign all_normal = site.posts | where_exp: 'item', 'item.pin != true and item.hidden != true' %}
{% assign posts = all_pinned | concat: all_normal %}
<ul class="plain">
  {% for post in posts %}
  <li><a href="{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
