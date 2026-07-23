---
layout: plain
title: notes
icon: fas fa-book
order: 1
---

<h2>college</h2>
<ul class="plain">
  <li><a href="/notes/s&amp;ds-669/course_notes.pdf">yale s&amp;ds 669 (statistical learning theory): course notes</a></li>
  <li><a href="/notes/s&amp;ds-669/final_presentation.pdf">yale s&amp;ds 669 (statistical learning theory): final presentation</a></li>
  <li><a href="/notes/math-330/anscombes-clt.pdf">yale math 330 (measure-theoretic probability): proof of randomly indexed clt</a></li>
  <li><a href="/notes/measuretheory/Sols_LebesgueMeasure.pdf">cornell/bard measure theory: solutions to "lebesgue measure"</a></li>
  <li><a href="/notes/measuretheory/Sols_IntroductionLebesgueIntegral.pdf">cornell/bard measure theory: solutions to "introduction to the lebesgue integral"</a></li>
  <li><a href="/notes/measuretheory/Sols_Probability.pdf">cornell/bard measure theory: solutions to "introduction to probability theory (with measures)"</a></li>
  <li><a href="/notes/FoundationsFE/Sols_IntroStochCalc.pdf">columbia ieor e4706: solutions to "a brief introduction to stochastic calculus"</a></li>
  <li><a href="/notes/s&amp;ds-242/Sandwich_Variance.pdf">yale s&amp;ds 242: sandwich asymptotic variance</a></li>
  <li><a href="/notes/slt/Solutions_to_SLT_Exercises.pdf">solutions to zach furman's singular learning theory exercises</a></li>
</ul>

<h2>doodles</h2>
<ul class="plain">
  {% assign doodles = site.doodles | sort: "order" %}
  {% for d in doodles %}
  <li><a href="{{ d.url }}">{{ d.title | downcase }}</a></li>
  {% endfor %}
</ul>
