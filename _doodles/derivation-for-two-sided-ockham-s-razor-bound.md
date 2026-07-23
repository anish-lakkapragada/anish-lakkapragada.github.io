---
layout: plain
title: "Derivation for Two-Sided Ockham's Razor Bound"
kind: derivation
order: 10
math: true
wide: true
back: /notes/
---

<em> Notation follows from these <a href="https://users.cs.duke.edu/~cynthia/CourseNotes/StatisticalLearningTheoryNotes.pdf">lecture notes</a>.</em> <br/>

Let us define our finite (binary) function class as $$\mathcal{F} = \{f_1, \dots, f_M \}$$ where each function in $$\mathcal{F}$$ predicts either $$-1$$ or $$1$$. Notation $$\mathbf{Z} \sim D^n$$ indicates that a probability is taken over the randomness of data draws $$Z_1, \dots, Z_n \sim D$$ where each $$Z_i = (X_i, Y_i)$$. Then $$\forall \epsilon > 0$$ we have: 

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \epsilon] \leq \sum_{j = 1}^M \mathbb{P}_{\mathbf{Z} \sim D^n}[| R^{\text{true}}(f_j) - R^{\text{emp}}(f_j)| > \epsilon]
$$

Applying the two-sided Hoeffding's Inequality, we know that $$\mathbb{P}_{\mathbf{Z} \sim D^n}[\mid R^{\text{true}}(f_j) - R^{\text{emp}}(f_j) \mid > \epsilon] \leq 2\exp(-2n\epsilon^2)$$ and so: 

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \epsilon] \leq 2M\exp(-2n\epsilon^2)
$$

To express this more nicely, let us define $$\delta = 2M \exp(-2n\epsilon^2)$$ so we can express $$\epsilon = \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}$$.

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}] \leq \delta
$$

$$
\implies \mathbb{P}_{\mathbf{Z} \sim D^n}[\forall \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| \leq \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}] \geq 1 - \delta
$$

which is exactly the two-sided Ockham's Razor Bound.
