---
layout: plain
title: "Informal Proof of Optional Stopping Theorem for Martingales"
kind: proof
order: 5
math: true
wide: true
back: /notes/
---

*I apologize in advance for any minor technicalities that might occur in informally proving this important theorem with minimal-to-no measure theory.* We first present this theorem, known as Optional Stopping Theorem (OST) or alternatively Doob's Optional Sampling theorem. 

<blockquote class="prompt-info" markdown="1">
##### Optional Stopping Theorem

Suppose we have a discrete-time martingale $$M = (X_n)_{n \geq 0}$$ where $$\mathbb{E}[\vert X_n \vert] < \infty$$ and $$\mathbb{E}[X_{n + 1} \mid X_n, \dots, X_0] = X_n$$. Furthermore, suppose we have a "bounded" stopping time $$T \in \mathbb{Z}$$ (i.e. $$\exists \ N \in \mathbb{Z}$$ s.t. $$N \geq T$$ [almost surely](https://en.wikipedia.org/wiki/Almost_surely)). Then, martingale $$M$$ *stopped* at $$T$$ is a martingale and we have: 

$$\mathbb{E}[X_T] = \mathbb{E}[X_0]$$

Note that this above property is non-trivial as $$T$$ is a random variable dependent on the observed values $$X_{0}, \dots, X_T$$.

</blockquote>

We first define the stopped process. Defining $$T \wedge n = \text{min}(\{T, n\})$$, we can define the stopped process of our martingale $$(X_n)_{n \geq 0}$$ as $$Y_n = X_{T \wedge n}$$. Note that because $$N < \infty$$ and $$\vert Y_n \vert \leq \underset{0 \leq k \leq N}{\max} \vert X_k \vert \implies $$ we can be very certain $$Y_n$$ is finite. With these definitions out of the way, we'll first prove that $$(Y_n)_{n \geq 0}$$ is a martingale (w.r.t observed events $$\{X_n, \dots, X_0\} \supseteq \{Y_n, \dots, Y_0\}$$ ) or equivalently that:  

$$
\mathbb{E}[Y_{n + 1} \mid X_n, \dots, X_0] = Y_n
$$

First note that $$\mathbb{E}[Y_{n + 1} \mid X_n, \dots, X_0] = \mathbb{E}[X_{T \wedge (n + 1)} \mid X_n, \dots, X_0]$$. Note that we can break $$\mathbb{E}[X_{T \wedge (n + 1)} \mid X_n, \dots, X_0]$$ into two different cases: 

**(1) Case One.** If $$T \leq n$$.

In this case, first note that $$T \leq n \implies T < n + 1 \implies T \wedge (n + 1) = T$$ and so: 

$$
\mathbb{E}[X_{T \wedge (n + 1)} \mid X_n, \dots, X_0] = \mathbb{E}[X_{T} \mid X_n, \dots, X_0]
$$

Furthermore, $$T \leq n \implies X_{T} \in \{X_N, \dots, X_0\}$$ and so $$\mathbb{E}[X_{T} \mid X_n, \dots, X_0] = X_T$$ (for the same reason $$\mathbb{E}[X_2 \mid X_3, \dots, X_0] = X_2$$. The value is already observed.) 

**(2) Case Two.** If $$T > n$$. 

In this case, $$T \geq n + 1 \implies T \wedge (n + 1) = n+ 1$$ and so: 

$$
\mathbb{E}[X_{T \wedge (n + 1)} \mid X_n, \dots, X_0] = \mathbb{E}[X_{n + 1} \mid X_n, \dots, X_0] = X_n
$$

Putting these two cases together, we have that: 

$$
\mathbb{E}[Y_{n + 1} \mid X_n, \dots, X_0] = \mathbb{E}[X_{T \wedge (n + 1)} \mid X_n, \dots, X_0] = X_T\mathbf{1}\{T \leq n\} + X_n\mathbf{1}\{T > n\} = X_{T \wedge n} = Y_n
$$

which concludes that $$(Y_n)_{n \geq 0}$$ is a martingale. A natural implication of this is that $$\forall \ n, \mathbb{E}[Y_n] = \mathbb{E}[Y_0] = \mathbb{E}[X_0]$$. But then $$\forall \ n \geq N \geq T, \mathbb{E}[X_T] = \mathbb{E}[Y_n] = \mathbb{E}[X_0] \implies \mathbb{E}[X_T] = \mathbb{E}[X_0]$$. 

This last property has some actually neat implications. Namely, stopping at a good time (i.e. $$T$$) does not change the expected outcome from when you first started. For example, if you're doing a symmetric random walk with $$\$1$$ in both directions, your expected outcome when stopping after hitting $$\$5$$ is still the same as when you started -- zero dollars. More profoundly, timing the martingale often has no (expected) benefit.
