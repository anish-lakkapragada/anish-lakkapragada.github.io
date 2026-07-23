---
layout: plain
title: "Proof of Cauchy-Schwarz Inequality For Expectations"
kind: proof
order: 3
math: true
wide: true
back: /notes/
---

*Note that we alternatively could prove this inequality by showing that the $$\langle X, Y \rangle = \mathbb{E}[XY]$$ is an inner product space and then cite the C-S inequality.* 

Suppose we have random variables $$X$$ and $$Y$$ where $$\mathbb{E}[X^2]$$ and $$\mathbb{E}[Y^2]$$ are finite. Pick any $$t \in \mathbb{R}$$. It is obvious that $$\mathbb{E}[(tX + Y)^2] \geq 0$$ and so we have: 

$$
\mathbb{E}[(tX + Y)^2] = \mathbb{E}[X^2]t^2 + 2\mathbb{E}[XY]t + \mathbb{E}[Y^2] \geq 0 
$$

which is a quadratic in $$t$$. Note that because we are sure for any $$t \in \mathbb{R}, \mathbb{E}[(tX + Y)^2] \geq 0 \implies $$ the discriminant (i.e. $$b^2 - 4ac$$) of the above quadratic is $$\leq 0$$. Using this condition for our quadratic above we have:  

$$
4\mathbb{E}[XY]^2 - 4\mathbb{E}[X^2]\mathbb{E}[Y^2] \leq 0 \implies \boxed{\mathbb{E}[XY]^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2]}
$$
