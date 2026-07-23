---
layout: plain
title: "Ridge Regression Closed Form Solution"
kind: derivation
order: 13
math: true
wide: true
back: /notes/
---

While the closed-form Normal Equation solution & <a href="https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/"> derivation</a> is well-known, the closed-form solution for ridge regression is less so. Taken straight from the <a href="https://en.wikipedia.org/wiki/Ridge_regression"> Wikipedia page</a>, we provide our cost function below for data $$\mathbf{X} \in \mathbb{R}^{n \times p}$$, weights $$\mathbf{\beta} \in \mathbb{R}^p$$, and labels $$\mathbf{y} \in \mathbb{R}^n$$:

$$
J(\mathbf{\beta}) = (\mathbf{y} - \mathbf{X\beta})^T (\mathbf{y} - \mathbf{X\beta}) + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c) = (\mathbf{y}^T - \mathbf{\beta}^T\mathbf{X}^T)(\mathbf{y - X\beta}) + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c)
$$

$$
= \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X\beta} - \beta^T\mathbf{X}^T\mathbf{y} + \mathbf{\beta}^T\mathbf{X}^T\mathbf{X\beta} + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c)
$$

Note that $$\mathbf{y}^T\mathbf{X\beta}$$ and $$\beta^T\mathbf{X}^T\mathbf{y}$$ are both scalars and are transposes of each other. Thus, they are equal and so we can write: 

$$
J(\mathbf{\beta}) = \mathbf{y}^T\mathbf{y} - 2\beta^T\mathbf{X}^T\mathbf{y} + \mathbf{\beta}^T\mathbf{X}^T\mathbf{X\beta} + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c) 
$$

We can think of this as essentially a Lagrange multiplier optimization problem, where our constraint is that $$\mathbf{\beta}^T\mathbf{\beta} - c = 0$$ for some $$c \in \mathbb{R}$$. We'll see that this choice of $$c$$ does not matter too much; it only matters that the constraint is there. We proceed: 

$$
0 = \frac{\partial J(\mathbf{\beta})}{\partial \mathbf{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X\beta} + \lambda(2\mathbf{\beta}) \implies \mathbf{X}^T\mathbf{y} = (\mathbf{X}^T\mathbf{X + \lambda I})\mathbf{\beta} 
$$

and so we get the final closed-form solution: 

$$
\mathbf{\beta} = (\mathbf{X}^T\mathbf{X + \lambda I})^{-1} \mathbf{X}^T\mathbf{y}
$$
