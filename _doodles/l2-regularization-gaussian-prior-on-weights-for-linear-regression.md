---
layout: plain
title: "L2 Regularization: Gaussian Prior on Weights for Linear Regression"
kind: derivation
order: 11
math: true
wide: true
back: /notes/
---

We first assume that $$y = \mathbf{x}^T \mathbf{w} + \epsilon$$ where $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$. Second, we assume that weights $$ \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I}) $$. Using our posterior distribution $$ \mathbf{w} \mid \mathbf{y}, \mathbf{X}$$, we can get an understanding of $$\mathbf{w}_{\text{MAP}}$$: 

$$ \ 
f(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) f(\mathbf{w}) \implies \mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) f(\mathbf{w})]
$$

$$
\implies \mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [\log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) + \log f(\mathbf{w})] 
$$

Given that $$ y_i \mid \mathbf{w}, \mathbf{x_i} \sim \mathcal{N}(\mathbf{x_i}^T \mathbf{w}, \sigma^2)$$, we can give a nice understanding of $$ \log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) $$: 

$$
\log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) = \sum_{i = 1}^n \log f(y_i \mid \mathbf{w}, \mathbf{x_i}) = \sum_{i = 1}^n - \log(\sqrt{2\pi \sigma^2}) - \frac{1}{2\sigma^2} (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2 = -\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2 + \text{const}
$$

where the constant is w.r.t to $$\mathbf{w}$$. For the log-density of our prior $$\log f(\mathbf{w})$$ we have: 

$$
\log f(\mathbf{w}) = \log[\exp(-\frac{1}{2} (\mathbf{w} - 0)^T (\tau^2 \mathbf{I})^{-1} (\mathbf{w} - 0) )] + \text{const} = -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}
$$

where $$\mathbf{w}^T \mathbf{w}$$ is just the square of the L2 norm. Putting this together we have: 

$$
\mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [-\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}] = \underset{\mathbf{w}}{\text{argmin}} \ [\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   + \frac{\sigma^2}{\tau^2} \mathbf{w}^T \mathbf{w}]
$$

Thus, we can conclude that MAP for weights under a Gaussian prior follows the objective as L2/Ridge Regression (albeit with a tuned $$ \lambda $$ resembling the $$\frac{\sigma^2}{\tau^2}$$) terms.
