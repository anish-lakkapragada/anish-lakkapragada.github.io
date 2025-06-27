---
layout: post
title: "supplement to the supplemental cs229 boosting lecture notes"
date: 2025-06-29 00:00:00
description: monitoring margins to regularize boosting algorithms
math: true
tags: 
- ADD
categories: 
- ADD
---

<!-- **This post assumes a little understanding of boosting algorithms. Reading the first few pages of [this](https://cs229.stanford.edu/extra-notes/boosting.pdf) should suffice.** -->

*TLDR: An exploration of regularizing boosting algorithms using basic margin theory. Then testing this regularization on same tree stump boosting classifer presented in the CS229 Supplemental Lecture Notes.*

Hey all! Hope you are well and thank you for taking the time to check out this post. After spending nearly all of last month on Statistical Learning Theory bounds, I am ready to find something else to dig into. Having heard of boosting (and bagging) so much in statistical machine learning for a while now, I've been curious for a while now on what boosting really is. After reading through a few online sources on boosting, I really enjoyed these [supplemental lecture notes](https://cs229.stanford.edu/extra-notes/boosting.pdf) on boosting from CS229. To the best of my knowledge, the notes' "infinite feature function" perspective to boosting is unique and very clean to work with.

One thing the notes did not cover, though, was handling overfitting when boosting classifiers. While the boosting procedure given a weak-learning algorithm is guaranteed to achieve zero training error eventually, it might not generalize as well as we would hope. In this really straightforward post, I'll provide another way of thinking about training progress to regularize our boosted classifiers. I'll also test this on the same example in the CS229 lecture notes. In true supplement fashion, I'll use the same notation as the original notes and only focus on binary classification[^label]. Because I won't spend much time covering boosting theory, it's probably worth skimming the notes if unfamiliar with boosting.

[^label]: We'll continue using the standard $$\{-1, 1\}$$ label space.

## preventing overfitting with an understanding of margins 

While there are many ways to prevent a classifier from overfitting, we focus on one for this post on boosted classifiers: early stopping. Namely, early stopping in this context means that we intentionally stop boosting[^why] our classifier *before* achieving zero training error. Naturally then, it would be nice to have some measurable criteria of when to stop training. Doing that requires understanding what *meaningful* training progress is, which can be measured by the *margin distribution* across all samples. 

### a little bit on margins 

We first start with some intuition. For a given sample $$(x_i, y_i)$$ with prediction $$\hat{y}_i$$, let $$\gamma_i \in \mathbb{R}$$ be its margin. At a high-level, $$\lvert \gamma_i \rvert$$ measures the distance between $$\hat{y}_i$$ and the prediction decision boundary (i.e. confidence) whereas $$\text{sign}(\gamma_i) > 0$$ measures whether the prediction was correct or not. We now give a formal definition: 

<blockquote class="prompt-info" markdown="1">
#### Definition 1.1 -- Margin of a Sample in a Boosting Classifier

Suppose we an infinite set of feature functions $$\phi = \{\phi_j\}_{j = 1}^{\mathbb{R}}$$ where each $$\phi_j: \mathbb{R}^n \to \{-1, 1\}$$ and an infinite set of feature weights in vector $$\vec{\theta} = \begin{bmatrix} \theta_1 & \theta_2 & \dots \end{bmatrix}^T$$. Then for a given observation $$(x_i, y_i)$$, we can define its margin $$\gamma_i$$ as: 

$$
\gamma_i = y_i \frac{\theta^T \phi(x_i) }{\|\theta\|_1} = y_i \frac{\sum_{j = 1}^{\infty} \theta_j \phi_j(x_i)}{\sum_{j = 1}^{\infty} |\theta_j|}
$$

For later convenience, we'll define a function to give this margin as $$\text{mrgn}((x_i, y_i), \phi, \theta)$$.

</blockquote>

Suppose we have a dataset $$ \{(x_i, y_i)\}_{i = 1}^m$$ of $$m$$ examples. Furthermore, suppose we are in iteration $t$ in our boosting procedure and have a feature functions set $$\phi^{(t)}$$ with feature weights $$\theta^{(t)}$$. Then using these above definition of a per-sample margin, we can define the *empirical margin distribution* of our dataset to be $$\{\gamma_1^{(t)}, \dots, \gamma_m^{(t)}\}$$. During training, we would aim for our (empirical) margin distribution to place high mass on large, positive margin values. If it's not already clear, we'll be using our empirical margin distribution to define our early stopping criteria. 

### when to stop boosting

Since the margin of a sample depends fundamentally on the data’s geometry and distribution, there is no single numerical threshold that universally defines a “good” margin. As such, we'll need to be a bit more clever on defining what a satisfiable empirical margin distribution looks like. One way to do this is with a bit of teleology: first boost the classifier until there are no training mistakes (call this # of iterations $$T_{\text{max}}$$) but ultimately use the boosting classifier (i.e. $$\phi^{(t)}$$ and $$\theta^{(t)}$$) at the iteration where the mean (or some other statistic) of the empirical margin distribution is greatest. In practice, we'll do something slightly more sophisticated. Let us define the final empirical margin distribution  after $$T_{\text{max}}$$ iterations as $$\mathcal{M}$$. Then, we can define $$\gamma^* = \text{perc}_{0.10}(\mathcal{M})$$ and select boosting classifiers at iteration $$T^*$$: 

$$
\begin{equation}
T^* = \underset{1 \leq t \leq T_{\text{max}}}{\text{argmin}} \sum_{i = 1}^m \mathbf{1} \{\text{mrgn}((x_i, y_i), \phi^{(t)}, \theta^{(t)}) < \gamma^*  \}
\label{t-star}
\end{equation}
$$

Note that this is is in contrast to conventional early stopping methods, which just end training after a pre-specified number of iterations or a certain training loss has been achieved. Here, we are choosing the least number of iterations possible so as to minimize the percentage of margins less than $$\gamma^*$$.

### some reasoning about this regularization procedure 

While this early stopping procedure might feel a bit strange, there is actually some [literature](https://arxiv.org/pdf/math/0508276) showing early stopping helps on AdaBoost, the boosting algorithm presented in the CS229 notes. Furthermore, this bound is meant to ensure that our margins are as positively great as possible -- which tightens this [margin-based generalization error bound](https://faculty.cc.gatech.edu/~isbell/tutorials/boostingmargins.pdf) (see Theorem 1.)

<!-- While it may seem like our above procedure is a bit strange, it actually has some foundation in a generalization bound. While I won't numerically compute this generalization bound like I did for the [last](/vc-rademacher-test) [two](/pac-bayes-stability) posts, I provide it below (adapted from [original paper](https://faculty.cc.gatech.edu/~isbell/tutorials/boostingmargins.pdf)):

<blockquote class="prompt-info" markdown="1">
#### Theorem 1.2 -- Generalization Bound for Margins 

Suppose that our data $$(X, y) \sim D$$ and we have a finite training sample $$S \sim D^m$$.

</blockquote> -->

## example: adaboost-ed decision stumps for binary classification 

We are now ready to test this regularization method on the same classification problem as presented in the original notes. We first start by giving some context for the original problem.

### problem statement

We restate the problem from the notes: we have our data in $$[0, 1] \times [0, 1] \subset \mathbb{R}^2$$ where the optimal binary classifier $$h^*(x)$$ is: 

$$
h^*(x) = \begin{cases}
1 & \text{if } x_1 \leq 0.6 \text{ and } x_2 \leq 0.6 \\ 
-1 & \text{else} 
\end{cases}
$$

It's worth repeating that this is not an easy problem for Logistic Regression. Following the notes, we'll train our boosted classifier with $$m = 150$$ examples. 

 <!-- Note that if our data distribution always followed this $$h^*(x)$$ rule, as they do in the notes example, there would be no need for any regularization as achieving zero training error $$\implies$$ achieving zero test error. As such, we'll add a little bit of label noise in our data generation process to give the model some room to overfit. That way, we can actually see the effect early stopping will have. Note though that $$h^*(x)$$ will still remain the optimal classifier. -->

### our weak-learner: decision stumps

Every boosting procedure relies on a weak-learning algorithm, which can construct a weak classifier on the dataset (given some weighted distribution of the samples). Our weak classifier will be a decision stump, which is a function that makes a classification decision based on a single feature's value. Specifically, for our problem, a decision stump $$\phi_{j, s}: \mathbb{R}^2 \to \{-1, 1\}$$ is parametrized by threshold $$s \in \mathbb{R}$$ and feature $$j \in \{1, 2\}$$: 

$$
\phi_{j, s}(x) = \text{sign}(x_j - s) = \begin{cases}
1 & \text{if } x_j \geq s \\ 
-1 & \text{else}
\end{cases}
$$

There is a weak-learning algorithm, described in the notes, to actually find the best $$j, s$$ to parametrize our weak classifier. For our boosting procedure, we'll use AdaBoost  (see Figure 2 in the notes).


[^why]: Here, "boosting" means conducting another training iteration to add another classifier to our entire classifier. 


### comparing early stopping boosting vs. vanilla boosting

Having gotten all preliminaries out of the way, we are ready to have some fun! We first provide a plot of the unregularized AdaBoost decision boundary after 10 iterations, when there are no training mistakes: 

<div style="text-align: center;">
  <img alt="Plot of visual boosting decision boundary." src="/assets/img/boostingcs229/adaboost_decision_boundary_unregularized.gif" style="max-width: 90%; height: auto;">
</div>

*Note that in the above plot the opaque dots and crosses represent test data, whereas the 150 bolder dots and crosses are the labeled data our boosted classifier was trained on. The point of showing these unseen test data is to visually make clear how sensitive our boosting procedure is.*

Let's apply our procedure to do better. First, we compute the empirical margin distribution $$\mathcal{M}$$ at the end of all 10 unregularized iterations, and then define $$\gamma^*$$ as the 10th percentile of $$\mathcal{M}$$ (we get $$\gamma^* = 0.2245$$.) Next, we find $$T^*$$ in \eqref{t-star} to be $$8$$.



## code 

## resources 


_Thank you for reading this blog post. If you have any questions or noticed any errata, please [email me](mailto:anish.lakkapragada@yale.edu)._