---
layout: plain
title: "Asymptotic Normality of Sample Quantiles"
kind: derivation
order: 8
math: true
wide: true
back: /notes/
---

*This derivation does not use Brownian motions.* Suppose we have a fixed percentile $$p$$ and a r.v. $$X$$ with true CDF $$F$$. Then we can define the true $$p$$-th quantile as $$q_p$$ where $$F(q_p) = p$$. Given samples $$X_1, \dots, X_n$$, we can define the sample $$p$$-th quantile as $$\hat{q}_p$$ where empirical CDF $$F_{n}(\hat{q}_p) = \frac{1}{n} \sum_{i = 1}^n \mathbf{1} \{ X_i \leq \hat{q}_p \} = p$$. We aim to understand the asymptotic distribution of $$\sqrt{n}(\hat{q}_p - q_p)$$. Note however that $$F_n(q_p)$$ does not necessarily equal $$p = F_n(\hat{q}_p)$$, although they should be close. Intuitively then, we first begin with a Taylor Series expansion of $$F_n(\hat{q}_p)$$ around $$q_p$$: 

$$
p = F_n(\hat{q}_p) \approx F_n(q_p) + F'_n(q_p)[\hat{q}_p - q_p] = F_n(q_p) + f(q_p)[\hat{q}_p - q_p]
$$

$$
\implies p \approx F_n(q_p) + f(q_p)[\hat{q}_p - q_p] \implies \sqrt{n}[\hat{q}_p - q_p] \approx \frac{\sqrt{n}[p - F_n(q_p)]}{f(q_p)}
$$

We'll be interested in having some convergence in distribution argument for the numerator. To do we use, we use Donsker's Theorem, presented below without proof: 

<blockquote class="prompt-info" markdown="1">
##### Donsker's Theorem

Suppose we have IID samples $$X_1, \dots, X_n$$ with a corresponding empirical CDF $$F_n$$. Then for a fixed $$x$$ we have that: 

$$
\sqrt{n}[F_n(x) - F(x)] \overset{d}{\to} \mathcal{N}(0, F(x)[1 - F(x)])
$$

</blockquote>

Applying Donsker's Theorem above with $$x = q_p$$, we arrive at the following: 

$$
\sqrt{n}[F_n(q_p) - p] \overset{d}{\to} \mathcal{N}(0, p(1 - p))
$$

and so with Slutsky's Lemma we have: 

$$
\sqrt{n}[\hat{q}_p - q_p] \approx \frac{\sqrt{n}[p - F_n(q_p)]}{f(q_p)} \overset{d}{\to} \boxed{\mathcal{N}(0, \frac{p(1 - p)}{f(q_p)^2})}
$$
