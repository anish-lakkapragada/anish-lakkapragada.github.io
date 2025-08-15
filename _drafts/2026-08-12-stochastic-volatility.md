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

While I think we all likely share similar intuition on why the volatility of a stock is *not* constant over time, a less obvious thing is why incorrectly assuming constant volatility, as the BSM does, is an issue. We'll now demonstrate this by showing that even for a delta-hedged portfolio, PnL drift will happen if the implied volatility differs from the incurred volatility along the path. 

We can see the drift without any talk of cash accounts or self-financing. Work over a tiny time step \(dt\) and:

1. **Taylor expand the option.** For a twice–differentiable \(V(S,t)\),
   $$
   dV \;\approx\; \Delta\, dS \;+\; \tfrac{1}{2}\Gamma\,(dS)^2 \;+\; \Theta\, dt,
   $$
   where \(\Delta = \frac{\partial V}{\partial S}\), \(\Gamma = \frac{\partial^2 V}{\partial S^2}\), \(\Theta = \frac{\partial V}{\partial t}\).

2. **Delta-hedge.** Hold \(-\Delta\) shares against one option. The **hedged PnL increment** (option change minus stock change) is
   $$
   d\mathrm{PnL} \;\approx\; dV \;-\; \Delta\, dS \;=\; \tfrac{1}{2}\Gamma\,(dS)^2 \;+\; \Theta\, dt.
   $$

3. **Plug in the model’s \(\Theta\).** If you *mark and take Greeks* from Black–Scholes with model volatility \(\widehat{\sigma}\) and (for simplicity) set \(r=0\), the BS PDE is
   $$
   \Theta \;+\; \tfrac{1}{2}\widehat{\sigma}^{\,2} S^2 \Gamma \;=\; 0
   \quad\Longrightarrow\quad
   \Theta \;=\; -\,\tfrac{1}{2}\widehat{\sigma}^{\,2} S^2 \Gamma.
   $$

4. **Use the realized quadratic variation.** Over \(dt\),
   $$
   (dS)^2 \;\approx\; \sigma^2 S^2\,dt,
   $$
   where \(\sigma\) is the **realized** (path) volatility during the interval.

Putting 2–4 together,
$$
d\mathrm{PnL}
\;\approx\;
\tfrac{1}{2}\Gamma\,\sigma^2 S^2\,dt
\;-\;
\tfrac{1}{2}\widehat{\sigma}^{\,2} S^2 \Gamma\,dt
\;=\;
\tfrac{1}{2}\Gamma\,S^2\big(\sigma^2-\widehat{\sigma}^{\,2}\big)\,dt.
$$

**Integrate through time** to get the cumulative effect:
$$
\mathrm{PnL}_T
\;\approx\;
\tfrac{1}{2}\int_0^T \Gamma_t\,S_t^2\big(\sigma_t^2-\widehat{\sigma}_t^{\,2}\big)\,dt.
$$

**Reading the sign.** For plain calls/puts, \(\Gamma_t\ge 0\). So if the path’s realized variance exceeds your model variance (\(\sigma^2>\widehat{\sigma}^2\)), the delta-hedged position **earns** on average; if realized variance is lower, it **bleeds**. That is exactly the PnL drift caused by using the “wrong” volatility, seen with only a Taylor expansion and the model’s own \(\Theta\).