---
layout: plain
title: "Proof of Perceptron Mistake Bound"
kind: proof
order: 7
math: true
wide: true
back: /notes/
---

<em>I am hardly the first to present this slightly unknown result, but I found it very elegant and interesting so I wanted to share it here. Specifically, its amazing how it is a bound on model performance that does NOT rely on the number of samples provided, albeit only for training and under some relatively strong assumptions. </em>

<blockquote class="prompt-info" markdown="1">
##### Perceptron Mistake Bound

Suppose that we have a binary classification dataset $$\mathcal{D} = (x_1, y_1), \dots, (x_N, y_N)$$ where two conditions are satisfied: **(1)** $$ \forall \ 1 \leq i \leq N, \Vert x_i \Vert_2 \leq R $$
and **(2)** $$\exists \ w^{\star}$$ s.t. $$\Vert w^{\star} \Vert_2 = 1$$ and $$\forall \ 1 \leq i \leq N, y_i(w^{\star} \cdot x_i) \geq \gamma$$.

Then using the standard <a href="https://cs.nyu.edu/~mohri/pub/pmb.pdf"> (online) perceptron learning algorithm </a> (see Figure 1), the total number of mistakes made during training $$\leq \frac{R^2}{\gamma^2}$$. Note that this is *online* as this bound is referring to the number of mistakes made while training on each new example $$(x_i, y_i)$$ as they come in. 
</blockquote>

<em> Proof</em>. This proof works by cleverly bounding $$\Vert w^{k + 1} \Vert$$, where $$k$$ refers to the number of updates (i.e. mistakes) incurred hitherto by training on this dataset. Essentially $$w^{k}$$ gives the current weight after training on $$\mathcal{D}$$ and incurring $$k$$ mistakes; $$w^{k + 1}$$ would give the weight after encountering a *new* sample, making a mistake with $$w^{k}$$, and updating $$w^{k}$$. 

We first start with the lower bound. Note that here $$(x_i, y_i)$$ below refer to the specific sample on which weight $$w^k$$ was updated (not necessarily $$i = k$$!) -- the specific value of $$(x_i, y_i)$$ nor the index $$i$$ is not important.

$$
w^{k + 1} \cdot w^{\star} = (w^{k} + y_ix_i) w^{\star} = w^{k} \cdot w^{\star} + y_i(w^{\star} \cdot x_i) \geq w_k \cdot w^{*} + \gamma
$$

In the perceptron learning algorithm, we initialize $$w_0 = \vec{0}$$. Thus by induction  we have that $$w^{k + 1} \cdot w^{\star} \geq w_k \cdot w^{*} + \gamma \implies w^{k + 1} \cdot w^{\star} \geq k \gamma$$. And now applying Cauchy-Schwarz Inequality, we have: 

$$
\Vert w^{k + 1} \Vert \times \Vert w^{\star} \Vert = \Vert w^{k + 1} \Vert \geq w^{k + 1} \cdot w^{\star} \geq k \gamma \implies \Vert w^{k + 1} \Vert \geq k\gamma
$$

We now proceed with an upper bound for $$\Vert w^{k + 1} \Vert$$, starting with the perceptron update again. Note again that index $$i$$ below is for a completely new sample on which the $$(k + 1)$$th mistake was made:

$$
\Vert w^{k + 1} \Vert^2 = \Vert w^{k} + y_ix_i \Vert^2 = \Vert w^k \Vert^2 + (y_i)^2 \Vert x_i\Vert^2 + 2y_i(w^k \cdot x_i)
$$

Because $$w_k$$ made a mistake on $$(x_i, y_i) \implies y_i(w^k \cdot x_i) < 0 \implies \Vert w^{k + 1} \Vert^2 \leq \Vert w^k \Vert^2 + (y_i)^2 \Vert x_i \Vert^2$$. But $$(y_i)^2 = 1$$ and $$\Vert x_i \Vert \leq R$$ and so we have: 

$$
\Vert w^{k + 1} \Vert^2 \leq \Vert w^k \Vert^2 + R^2 \implies \Vert w^{k + 1}\Vert^2 \leq kR^2 \implies \Vert w^{k + 1} \Vert \leq R\sqrt{k}
$$

through another induction argument. Putting these two bounds together we have: 

$$
k\gamma \leq \Vert w^{k + 1} \Vert \leq R\sqrt{k} \implies \sqrt{k} \leq \frac{R}{\gamma} \implies \boxed{k \leq \frac{R^2}{\gamma^2}}
$$

In other words, the number of mistakes made on $$\mathcal{D}$$ while training with the online perceptron learning algorithm is $$\leq \frac{R^2}{\gamma^2}$$. Proof here is taken from <a href="https://www.cs.cmu.edu/~mgormley/courses/606-607-f18/slides607/lecture4-pmb.pdf"> CMU's 10-607 slides</a> with slightly more commentary.
