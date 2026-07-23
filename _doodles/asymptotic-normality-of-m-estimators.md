---
layout: plain
title: "Asymptotic Normality of M-Estimators"
kind: derivation
order: 9
math: true
wide: true
back: /notes/
---

Suppose we have i.i.d samples $$X_1, \dots, X_n$$ and our goal is to estimate $$\theta$$. Brushing aside technicalities, let us define a differentiable function $$\rho: \mathcal{X} \times \Theta \to \mathbb{R}$$ where $$\psi(x, \theta) = \frac{\partial \rho(x, \theta)}{\partial \theta}$$. Then estimator $$\hat{\theta} = \underset{\theta}{\text{argmax}} \  \sum_{i = 1}^n \rho(X_i, \theta)$$ is an $$M$$-estimator. The true parameter value $$\hat{\theta}_0$$ we wish to estimate can be given as $$\theta_0 = \underset{\theta}{\text{argmax}} \ \mathbb{E}[\rho(X, \theta)]$$. Note that these expectations are over our samples, and we <em> do not </em> assume any knowledge of the distribution of $$X$$. By definition of the $$M$$-estimator, we have $$\sum_{i} \psi(X_i, \hat{\theta}) = 0$$ and so using a Taylor Series expansion:  

$$
\small
0 = \sum_{i} \psi(X_i, \hat{\theta}) \approx \sum_{i} \psi(X_i, \theta_0) + \sum_{i} \psi'(X_i, \theta_0)  \implies \sqrt{n}[\hat{\theta} - \theta_0] \approx \frac{\sqrt{n}  \sum_{i} \psi(X_i, \theta_0)}{ - \sum_{i} \psi'(X_i, \theta_0)} = \frac{ \sum_{i} \psi(X_i, \theta_0) / \sqrt{n}}{ - \sum_{i} \psi'(X_i, \theta_0) / n}
$$

This is nearly identical logic to <a href="/notes/s&ds-242/Sandwich_Variance.pdf"> this derivation</a> of the Sandwich Asymptotic Variance for MLEs in model misspecification scenarios. This is indeed because the MLE is a case of the general $$M$$-estimator. Using that derivation, we arrive at: 

$$
    \sqrt{n}[\hat{\theta} - \theta_0] \overset{d}{\to} \mathcal{N}(0, V^{-1}WV^{-1}) 
$$

where $$V = \mathbb{E}[\psi'(X, \theta_0)]$$ and $$W = \mathbb{E}[\psi(X, \theta_0)^2]$$.
