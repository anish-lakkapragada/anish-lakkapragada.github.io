---
layout: post
title: "extending the black-scholes constant volatility assumption"
date: 2025-08-12 00:00:00
description: an exercise in itô processes and mathematical finance
math: true
tags: 
# - boosting
# - adaboost
# - gentleboost
categories: 
# - Statistics
# - Machine Learning
# image:
#   path: /assets/img/boostingcs229/ada-vs-gentleboost-preview.gif
#   alt: Plot of visual boosting decision boundary of unregularized AdaBoost versus Gentle AdaBoost, or GentleBoost, with decision stumps weak-learning algorithm
---

*TLDR: Hehhee.*

I hope you are doing well! I apologize for not posting new blogs at my previous 1 post/week pace; I decided to take a break and instead spend that time going through the [Green Book](https://academyflex.com/wp-content/uploads/2024/03/a-practical-guide-to-quantitative-finance-interviews.pdf) to learn more about all kinds of different stochastic processes. While many of my learnings did show up in my [notes](/notes/), I'm excited to present them now in a long-form post. Specifically, in this blog post, I want to play around with the Black-Scholes-Merton model by relaxing its assumption of constant volatility in the underlying stock. I was particularly inspired by this [UChicago REU paper](https://math.uchicago.edu/~may/REU2020/REUPapers/Wang,Xiaomeng.pdf) which presented different Stochastic Volatility (SV) processes; in this blog post, I hope to introduce these models and test them out with a little bit of Python.

## introduction to the problem  

> I will assume some familiarity with Itô processes, SDEs, and Black-Scholes-Merton (BSM) in this post. 

As a reminder the BSM SDE posits that our stock-price $$S_t$$ is given as a [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion): 

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

where $$W_t$$ is standard Brownian motion. From this SDE, we can derive the BSM PDE for our derivative $$V = V(S, t)$$: 

$$
\frac{\partial V}{\partial t} + rS_t \frac{\partial V}{\partial S_t} + \frac{\sigma^2 S_t^2}{2}  \cdot  \frac{\partial^2 V}{\partial S_t^2} = rV
$$

As we can see, one of the major assumptions of BSM is that the volatility $$\sigma$$ of our stock is constant. Such an assumption is extremely naive, as demonstrated by the *volatility smile*. If we define the implied volatility $$\sigma_{\text{implied}}$$ as the solved value of $$\sigma$$ such that the BSM formula computed value of our derivative equals its true market price, then we observe the following smile-y curve: 

<!-- curve being shown -->

and so we can see that volatility $$\sigma$$ indeed *does* change as a result of the current strike price. This is an empirical violation of the BSM assumption.

### some ramifications of incorrectly assuming constant volatility 

While I think we all likely share similar intuition on why the volatility of a stock is *not* constant over time, a less obvious thing is why incorrectly assuming constant volatility, as the BSM does, is an issue. 


## describing volatility as a stochastic process 



