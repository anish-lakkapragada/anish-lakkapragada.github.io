---
layout: plain
title: "L1 Regularization: Laplace Prior on Weights for Linear Regression"
kind: derivation
order: 12
math: true
wide: true
back: /notes/
---

For our weights $$\mathbf{w} \in \mathbb{R}^d$$, we assume each individual weight component is independent with prior $$w_i \sim \text{Laplace}(0, b)$$. So we get the following log-density for our weights: 

$$
\log f(\mathbf{w}) = \sum_{i = 1}^d f(w_i) = \sum_{i = 1}^d \frac{-|w_i - 0 |}{b} + \text{const.} = -\frac{1}{b} \sum_{i = 1}^d |w_i| + \text{const.}
$$

and so using identical work from the previous L2 derivation we get: 


$$
\mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [-\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2  + \log f(\mathbf{w})] = \underset{\mathbf{w}}{\text{argmin}} \ [\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   + \frac{2\sigma^2}{b} \sum_{i = 1}^d |w_i|]
$$

Thus, we arrive at a similar conclusion that MAP for weights under a Laplace prior follows the same objective as L1 Regularization (where hyperparameter $$\lambda$$ is tuned to resemble the $$\frac{2\sigma^2}{b}$$ term.)
