---
layout: plain
title: "Proof of Triangle Inequality for Expectations"
kind: proof
order: 4
math: true
wide: true
back: /notes/
---

Suppose we have real-valued random variables $$X$$ and $$Y$$ where $$\mathbb{E}[X]$$ and $$\mathbb{E}[Y]$$ are finite (i.e. $$X$$ and $$Y$$ are integrable). Then for any values our r.v.s $$X$$ and $$Y$$ can take on we have $$\vert X + Y \vert \leq \vert X \vert + \vert Y \vert \implies \mathbb{E}[\vert X + Y \vert] \leq \mathbb{E}[\vert X \vert + \vert Y \vert ] = \mathbb{E}[\vert X \vert] + \mathbb{E}[\vert Y \vert]$$. 

This is not a terribly interesting inequality, so we'll derive another expectation inequality relating differences. First note that for any r.v. $$W$$ we have $$\vert \mathbb{E}[W] \vert \leq \mathbb{E}[\vert W \vert]$$ as: 

$$
\vert \mathbb{E}[W] \vert = \vert \int_{-\infty}^{\infty} w f_{W}(w) dw \vert \leq \int_{-\infty}^{\infty} \vert w \vert f_{W}(w) dw = \mathbb{E}[\vert W \vert]
$$

An alternative explanation is that this is an example of Jensen's inequality as the absolute value function is convex. Now defining r.v. $$W = X - Y$$, we have: 

$$ 
\vert \mathbb{E}[X] - \mathbb{E}[Y] \vert = \vert \mathbb{E}[W] \vert \leq \mathbb{E}[\vert W \vert] = \mathbb{E}[\vert X - Y \vert]
$$

or $$\vert \mathbb{E}[X] - \mathbb{E}[Y] \vert \leq \mathbb{E}[\vert X - Y \vert]$$. 

I found this particularly useful when trying to understand convergence in expectation (e.x. [this limit on page 6](https://math.mit.edu/~sheffield/2019600/martingalenotes.pdf)).
