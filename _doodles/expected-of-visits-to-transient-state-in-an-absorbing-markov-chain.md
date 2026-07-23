---
layout: plain
title: "Expected # of visits to transient state in an Absorbing Markov Chain"
kind: derivation
order: 6
math: true
wide: true
back: /notes/
---

Suppose we have an absorbing Markov Chain with $$t$$ transient states and $$r$$ absorbing states. Then we can order its states so its transition matrix $$P$$ can be given as a block matrix: 

$$
P = \begin{bmatrix} Q & R \\ \mathbf{0} & I_r \end{bmatrix}
$$

Note that we can call $$Q$$ the *transient-transient* matrix as it gives the probability of going from one transient state to another. We'll use $$Q$$ to understand the number of visits to a given transient state. Let's first define the problem a bit more clearly: given absorbing Markov Chain $$M = \{X_n\}_{n \geq 0}$$ with initial transient state $$X_0 = i$$, we are interested in the expectation of the number of visits $$V_{ij} = \sum_{k = 0}^{\infty} \mathbf{1}\{X_n = j\}$$ to transient state $$j$$. We now take the expectation of $$V_{ij}$$ under the assumption that $$X_0 = i$$ (denoted by $$\mathbb{E}_i$$): 

$$
\mathbb{E}_i[V_{ij}] = \sum_{k = 0}^{\infty} \mathbb{E}_i[\mathbf{1}\{X_k = j\}] = \sum_{k = 0}^{\infty} \mathbb{P}_i[X_k = j]
$$

*A few technicalities: Moving $$\mathbb{E}_i[\cdot]$$ into the infinite summation is allowed because $$V_{ij} \geq 0$$ (i.e. [Tonelli's theorem, way above my paygrade](https://faculty.fiu.edu/~meziani/Lecture18.pdf)). Furthermore, $$M$$ is absorbing $$\implies V_{ij} < \infty \implies \mathbb{E}_i[V_{ij}] < \infty$$.* 

The $$\mathbb{P}_i[X_k = j]$$ quantity is essentially the probability that $$\mathbb{P}[X_k = j \mid X_0 = i]$$. Through a simple matrix multiplication argument it should be clear that $$(Q^k)_{ij} = \mathbb{P}[X_k = j \mid X_0 = i] \implies$$ the expected number of visits to transient state $$j$$ starting from transient state $$i$$ is given by $$\mathbb{E}_i[V_{ij}] = \sum_{k = 0}^{\infty} (Q^k)_{ij}$$. 

This is not a terribly useful result due to the infinite summation. Luckily $$\sum_{k = 0}^{\infty} Q^k$$ is a [Neumann series](https://en.wikipedia.org/wiki/Neumann_series) as: 

- (1) We formally can consider $$Q$$ as an operator $$Q: \mathbb{R}^t \to \mathbb{R}^t$$ by $$v \mapsto Qv$$ 
- (2) $$Q$$ (as an operator) is linear, meaning $$Q(cv + w) = cQ(v) + Q(w)$$
- (3) $$Q$$ (as an operator) operates on a [normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space) $$\mathbb{R}^t$$ (this vector space is normed as $$t < \infty$$)
- (4) $$Q$$ (as an operator) [is bounded as it is defined on a finite dimensional normed space](https://math.stackexchange.com/questions/2983050/every-linear-operator-tx-to-y-on-a-finite-dimensional-normed-space-is-bounde) $$\ \mathbb{R}^t$$ 

So by the Neumann series theorem, we have:  

$$
\sum_{k = 0}^{\infty} Q^k = (I_t - Q)^{-1}
$$

where $$(I_t - Q)^{-1}$$ when considered as a matrix and not an operator is canonically called the *fundamental matrix*. So to summarize, the $$(i, j)$$th entry of this matrix $$(I_t - Q)^{-1}$$ will give you the expected number of visits to transient state $$j$$ starting from transient state $$i$$ in our absorbing markov chain $$M$$.
