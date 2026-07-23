---
layout: plain
title: "Informal Proof: Itô Process is a Martingale ⇒ Itô Process is Driftless"
kind: proof
order: 2
math: true
wide: true
back: /notes/
---

Let us define our Itô process $$\{X_t\}_{t \geq 0}$$ with the standard SDE: 

$$
dX_t = a(t, X_t)dt + b(t, X_t)dW_t \iff X_t - X_k = \int_{k}^{t} a(X_s, s) \ ds + \int_{k}^{t} b(X_s, s)dW_s 
$$

where $$a(t, X_t)$$ and $$b(t, X_t)$$ are the drift and diffusion terms, and $$W_t$$ is a standard Brownian motion process. Fix $$t, s \geq 0$$. Because we are given $$X_t$$ is a martingale, we have that $$\mathbb{E}[X_{t + s} \mid \mathcal{F}_t] = X_t$$ or: 

$$
0 = \mathbb{E}[X_{t + s} - X_t \mid \mathcal{F}_t] = \mathbb{E}[\int_{t}^{t + s} a(X_s, s) \ ds \mid \mathcal{F}_t] + \mathbb{E}[\int_{t}^{t + s} b(X_s, s) dW_s \mid \mathcal{F}_t]
$$

Showing that $$\mathbb{E}[\int_{t}^{t + s} b(X_s, s) dW_s \mid \mathcal{F}_t] = 0$$ is a standard calculation of evaluating a stochastic integral as a limit of stochastic integrals of elementary processes, which are converging to $$b(X_s, s)$$. See [Example 2](https://www.columbia.edu/~mh2078/FoundationsFE/IntroStochCalc.pdf) on how this works. Furthermore, this result is unsurprising as $$\mathbb{E}[\Delta W] = 0$$ and $$\Delta W$$ is independent of $$b(X_s, s)$$ during this same time period.

So we have that $$0 = \mathbb{E}[\int_{t}^{t + s} a(X_s, s) ds \mid \mathcal{F}_t]$$. Brushing aside some technicalities, this also gives us $$0 = \int_{t}^{t + s} \mathbb{E}[a(X_s, s) \mid \mathcal{F}_t] \ ds$$. From here, we can cleverly use the first Fundamental Theorem of Calculus to get rid of the expectation: 

$$
\underset{s \to 0}{\lim} \frac{1}{s} \int_{t}^{t + s} \mathbb{E}[a(X_s, s) \ ds \mid \mathcal{F}_t] = \mathbb{E}[a(X_t, t) \mid \mathcal{F}_t] \ ds = a(X_t, t)
$$

But as we know $$\int_{t}^{t + s} \mathbb{E}[a(X_s, s) \mid \mathcal{F}_t] ds = 0$$ and so the above limit is equal to 0 $$\implies a(X_t, t) = 0$$. Furthermore $$t$$ was arbitrary and so $$\forall t, a(X_t, t) = 0 \implies X_t$$ is driftless.

Note that proving the opposite direction is simpler, at least intuitively: 

$$\mathbb{E}[X_{t + s} \mid \mathcal{F}_t] = \mathbb{E}[X_t + \int_{t}^{t + s} a(X_s, s) ds + \int_{t}^{t + s} b(X_s, s) dW_s \mid \mathcal{F}_t] = X_t + \mathbb{E}[\int_{t}^{t + s} b(X_s, s) dW_s \mid \mathcal{F}_t]= X_t$$
