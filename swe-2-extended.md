---
layout: post
title: extending swe-2's reward function to steer the pareto frontier
permalink: /swe-2-extended/
date: 2026-09-25 00:00:00 -0700
description: Experiments extending the reward function in Cognition's SWE-2 model release.
math: true
wide: true
back: /
---

This is the full explanation for a project I worked on extending SWE-2’s reward function. For a short summary, please see the video below! 
{% include swe2-video.html %}

## What did Cognition do in SWE-2?

Earlier in September, Cognition released [SWE-2](https://cognition.com/blog/swe-2) which made great progress on the capability vs. cost pareto curve, as measured by Cognition’s own benchmark [FrontierCode 1.1](https://cognition.com/blog/frontier-code-1.1). 

In their blog, Cognition explains how SWE-2’s post-training reward function is explicitly constructed to improve the Pareto curve. Rigorously speaking, the reward function for a given effort level $$e$$ is given by $$R(S, C) = S - \lambda^{(e)} C$$, where $$S$$ is the average solve rate and $$C$$ is the average cost and $$\lambda^{(e)} > 0$$. What Cognition [elegantly derives](https://cognition.com/blog/swe-2#appendix-b) is that the slope $$\lambda^{(e)}$$ should be set to the slope of the Pareto curve of the base model at effort level $$e$$. 

For a visual explanation, please see the <a href="/assets/swe-2-extended/approximating-pareto-curve-cog.png" target="_blank" rel="noopener noreferrer"><em>“Approximating the Pareto curve tangents of Kimi K3”</em></a> figure from the SWE-2 blog post.

## Problem Statement: Steering Pareto Frontier Improvements 

Overall, this methodology is great as it yields pareto frontier improvement throughout post-training. The key thing to note here is that while this “slope-matched penalty” will push the pareto frontier, it is unopinionated on *where* the model should land on this frontier. This can be seen in the <a href="/assets/swe-2-extended/slope-matched-penalty.png" target="_blank" rel="noopener noreferrer">slope-matched penalty figure</a>, where any point along an iso-reward line gets the same reward.

**But, what if we could steer the direction in which the Pareto Frontier improved?** More explicitly, there’s two clear basis vectors on which to improve: 

- Same performance, lower cost 
- Higher performance, same cost 

and a spectrum in between. For example, **if we wanted to get the same performance for a lower cost, how would we accomplish this?** 

## Solution: Update $$\lambda^{(e)}$$ throughout post-training based on performance-cost tradeoff param $$\alpha \in [0, 1]$$

This section will require more math to derive the update rule and will be technical. That said, the final update rule is in the red box at the end of this section. 

We start with some notation. If we define $$e$$ to be some categorical variable representing the effort level (low/medium/high) and $$\pi$$ to be our policy, we can define the following two metrics: 

$$
s_e(\pi) = \mathbb{E}_{\pi}[S \mid e], \quad c_e(\pi) = \mathbb{E}_{\pi}[C \mid e] 
$$

or the average solve rate and average cost, respectively, for effort level $$e$$. The base model performance for this effort can similarly be given by $$s_{e, 0} = \mathbb{E}_{\pi_{0}}[S \mid e]$$ and $$c_{e, 0} = \mathbb{E}_{\pi_{0}}[C \mid e]$$ where $$\pi_0$$ is the initial base model policy. The idea here is then that the relative improvement of a new policy $$\pi$$ over $$\pi_0$$ can be quantified like such: 

$$
u_e(\pi) = \frac{s(\pi) - s_{e, 0}}{s_0}, \quad v_e(\pi) = \frac{c_{e, 0} - c_e(\pi)}{c_{e, 0}}
$$

The idea here is then to see how we can optimize $$\pi$$ to maximize success $$u_e(\pi)$$ and cost $$v_e(\pi)$$ improvement in a way that balances a ratio of both of them. **This is done through a parameter $$\alpha \in [0, 1]$$ which we set.** This problem can be written as essentially a constrained optimization problem of real-valued quantity $$\tau > 0$$: 

$$
\boxed{ \begin{aligned} \max_{\pi,\tau}\quad & \tau \\[4pt] \text{subject to}\quad & u_e(\pi) \ge \alpha\tau, \\ & v_e(\pi) \ge (1-\alpha)\tau. \end{aligned} }
$$

Here we can see that: 
- $$\alpha = 0$$ means we demand $$u_e(\pi) \geq 0$$ and $$v_e(\pi) \geq \tau \implies$$ we only care about cost improvement, while keeping success the same 
- $$\alpha = 1$$ vice versa means we only care about success improvement, while keeping cost the same  

and $$0 < \alpha < 1$$ optimizes the spectrum in between. 
As we will see, the solution elegantly yields an update rule for $$\lambda_{\text{prev}}^{(e)}$$ throughout training. 

We now derive this solution.

<details class="details-block derivation" markdown="1">
<summary>Solution Derivation</summary>

To start, we’ll first assume $$\alpha \in (0, 1)$$ and deal with the endpoints later. Under this assumption, the conditions imply that $$\tau \leq \frac{u_e(\pi)}{\alpha}$$ and $$\tau \leq \frac{v_e(\pi)}{1 - \alpha} \implies $$ the maximum value of $$\tau$$ is $$\min(\frac{u_e(\pi)}{\alpha}, \frac{v_e(\pi)}{1 - \alpha})$$. Hence observing $$\max_{\pi, \tau} \tau = \max_{\pi} (\max \tau)$$ our original constrained optimization problem can be re-expressed as $$\max_{\pi} \min(\frac{u_e(\pi)}{\alpha}, \frac{v_e(\pi)}{1 - \alpha})$$. But $$\alpha, 1 - \alpha \in \mathbb{R}_{> 0}$$ so this is equivalent to $$\max_{\pi} \min( (1-\alpha) u_e(\pi), \alpha v_e(\pi))$$.

Now in standard math fashion, we will pull an identity out of our ass: 

$$
\min(a, b) = \min_{\beta \in [0, 1]} [\beta a + (1 - \beta) b]
$$ 

and so applied here our optimization problem is written as: 

$$
\max_{\pi} \min_{\beta \in [0, 1]} \beta(1 - \alpha)u_e(\pi) + (1 - \beta)\alpha v_e(\pi)
$$

expanding this out we get:

$$
\max_{\pi} \min_{\beta \in [0, 1]} \underbrace{\frac{\beta(1 - \alpha) s_e(\pi)}{s_{e, 0}} - \frac{(1 - \beta)\alpha c_e(\pi)}{c_{e, 0}} + \text{const. w.r.t. $\pi, \beta$}}_{L(\pi, \beta)}
$$

and so if we call this function $$L(\pi, \beta)$$. Notice that if we *fix* $$\beta$$ and multiply $$L(\pi, \beta)$$ by $$\frac{s_{e, 0}}{\beta(1 - \alpha)} > 0$$, we get: 

$$
R_e(\pi, \beta) := s(\pi) - \lambda^{(e)} c(\pi), \quad \lambda^{(e)} := \frac{s_{e, 0}}{c_{e, 0}} (\frac{\alpha}{1 - \alpha})(\frac{1 - \beta}{\beta})
$$

Looks familiar to the SWE-2 function, correct! The main point here is that the reward function $$\min_{\beta \in [0, 1]} L(\pi, \beta)$$ we will be updating $$\pi$$ against is exactly the same as in SWE-2, just with $$\lambda$$ as a function of $$\beta$$. 

The remaining question is how to solve $$\min_{\beta \in [0, 1]} L(\pi, \beta)$$ for a reward function of the $$\pi$$ (which can then be RL’d against with your favorite policy optimization algorithm.) To start with, recall $$u_e(\pi)$$ and $$v_e(\pi)$$ relied on expectations and hence for this iteration let’s call our empirical estimates $$\hat{u}_e := \frac{\hat{s_e}(\pi) - s_{e, 0}}{s_{e, 0}}$$ and $$\hat{v}_e := \frac{c_{e, 0} - \hat{c_e}(\pi)}{c_{e, 0}}$$. Then one way we can update $$\beta$$ given $$\beta_{\text{prev}}$$ as the previous iteration’s value is using standard optimization over $$L(\pi, \beta)$$ with a KL term on $$\beta$$, or concretely: 

$$
\beta \gets \text{argmin}_{0 \leq \beta \leq 1} \textcolor{blue}{[\beta(1 - \alpha) \hat{u}_e + (1 - \beta)\alpha \hat{v}_e + \frac{1}{\eta} D_{\text{KL}}((\beta, 1 - \beta) \parallel (\beta_{\text{prev}}, 1 - \beta_{\text{prev}}))]}
$$  

where $$\eta > 0$$. To be clear, the KL divergence can be given as: 

$$ 
D_{\text{KL}}((\beta, 1 - \beta) \parallel (\beta_{\text{prev}}, 1 - \beta_{\text{prev}})) = \beta \log \frac{\beta}{\beta_{\text{prev}}} + (1 - \beta) \log \frac{1 - \beta}{1 - \beta_{\text{prev}}}
$$

Using a derivative w.r.t. $$\beta$$ to optimize the <span style="color: blue;">blue term</span> above we get: 

$$
(1 - \alpha) \hat{u}_e - \alpha \hat{v}_e + \frac{1}{\eta}[\log \frac{\beta}{1 - \beta} + \log \frac{1 - \beta_{\text{prev}}}{\beta_{\text{prev}}}] = 0
$$

which yields the update rule: 

$$
\log \frac{1 - \beta}{\beta} = \log \frac{1- \beta_{\text{prev}}}{\beta_{\text{prev}}} + \eta[(1 - \alpha)\hat{u}_e - \alpha \hat{v}_e]
$$

Observing that $$\lambda^{(e)} \propto \frac{1 - \beta}{\beta}$$, this implies the update rule: 

</details>

<div class="update-rule" markdown="1" style="border: 2px solid #c62828; border-radius: 6px; padding: 0 1rem; margin: 1rem auto; width: fit-content; max-width: 100%; box-sizing: border-box;">

$$
\log (\lambda^{(e)}) \gets \log(\lambda_{\text{prev}}^{(e)}) + \eta[(1 - \alpha)\hat{u}_e - \alpha \hat{v}_e]
$$

</div>

Note this extends cleanly to $$\alpha \in \{0, 1\}$$, and hence we are happy.

*Implementation Note*: Because we are choosing $$\alpha$$ and using it as a targeted direction for many sequential runs, $$s_{e, 0}$$and $$c_{e, 0}$$ are benchmarked against the *initial* policy $$\pi_0$$ as opposed to just the previous iteration.

## Toy-Setup: Multi-effort RL Task on 5 Trainable Parameters

We now explain our setup for testing this method on a toy RL-task supporting mutliple efforts and involving five trainable parameters total. A given rollout works as follows (where $0 \leq p_{\text{bad}} \leq p_{\text{good}} \leq 1$ are fixed): 

1. Sample a given task type $x$ uniformly from $$\{0, 1\}$$
2. Give the policy $\pi$ the task $x$ and an effort level $e$. Effort levels can vary from low/medum/hard. 
3. The policy will sample tool $$a \in \{0, 1\}$$ and a candidate count $K$
4. Generate $K$ candidates, where each candidate succeeds with probability: 

$$ 
p(x, a) = \begin{cases}
    p_{\text{good}} & a = x \\ 
    p_{\text{bad}} & a \neq x
\end{cases}
$$

5. Return success if $\geq 1$ candidate passes, and return the cost as $K$. Note that $K = 0$ means the success is zero.

And that's it! Our reward function in all scenarios will operate like $R = S - \lambda^{(e)} K$. 

**We now define the five-parameter policy.** To start, define logits $z_0, z_1$ where $\rho_x = \sigma(z_x) = \frac{1}{1 + \exp(-z_x)}$ and the policy samples $a \mid x \sim \text{Bern}(\rho_x)$. In other words, $\rho_x$ gives probability the policy chooses tool 1 on task type $x$. 

Note then we can give the probability of picking the correct tool (i.e. $a = x$) as $q = \frac{(1 - \rho_0) + \rho_1}{2}$ because both task types are equally likely. Because we initialize $z_0 = z_1 = 0$, $q_0 = \frac{1}{2}$. These two logits control the policy's ability to succeed. 

The remaining three parameters dictate how the policy (independently) decides the cost $K$. Specifically for each effort level $e$ (low/medium/high), we have trainable parameter $u_e$, where the total number of candidates is sampled like $K \sim \text{Geom}(\frac{1}{1 + \exp(u_e)}) - 1$ so the average cost incurred for effort level $e$ is $\mathbb{E}[K \mid e] = \exp(u_e)$.

So the five trainable parameters are $\theta = (z_0, z_1, u_{\text{low}}, u_{\text{medium}}, u_{\text{high}})$. Now an additional nice thing about this task is that it has a closed-form pareto curve $s(c) = \mathbb{P}[S = 1 \mid c]$, or in other words, average success can be written as a function of average cost. It is derived below. 

<details class="details-block derivation" markdown="1">
<summary>Pareto Curve Derivation</summary>

First suppose we are given a fixed task type $x$ with selected effort level $e$. We have chosen a tool with a $p$ chance of success on this task type, and moreover this effort level currently yields average cost $c := \mathbb{E}[K \mid e]$ through $u_e$'s value. Then observe that defining $r := \frac{1}{1 + c}$ (recall $K \sim \text{Geom}(r) - 1$): 

$$
\begin{aligned}
\mathbb{P}(S = 0 \mid p, c)
&= \sum_{k =0}^{\infty} \mathbb{P}(K = k) \cdot \mathbb{P}(S = 0\mid p, c, K = k) \\
&= \sum_{k =0}^{\infty} r(1 - r)^k \cdot (1 - p)^k \\
&= r \sum_{k =0}^{\infty} [(1 - r) \cdot (1 - p)]^k \\
&= r \cdot \frac{1}{1 - (1 - r) \cdot (1 - p)} \\
&= \frac{r}{r + p(1 - r)} \\
&= \frac{1/(1 + c)}{\frac{(1 + pc)}{(1 + c)}} \\
&= \frac{1}{1 + pc}
\end{aligned}
$$

and thus $\mathbb{P}(S = 1 \mid p, c) = \frac{pc}{1 + pc}$, hence meaning that if $q$ is the probability of picking the correct tool (determined by $z_x$ logits), then using LOTP: 

</details>

$$
s(c) = \mathbb{P}[S = 1 \mid c] =  q \frac{p_{\text{good}}c}{1 + p_{\text{good}}c} + (1 - q)\frac{p_{\text{bad}}c}{1 + p_{\text{bad}}c}
$$

To evaluate for any given effort level just pass in $c = \exp(u_e)$. The benefit of this is we can easily compute the derivative $s'(c)$ to find the slope of the pareto curve, which we use for initializing $\lambda^{(e)}$ in our experiments. 

## Toy Experiment Results 

We test across 100 different problems of the described task, where each problem has a different $(p_{\text{good}}, p_{\text{bad}})$ coordinate. For each problem, we run with three seeds for the policy RNG. We use $z_0 = z_1 = 0$ for initializing tool logits and $u_{\text{low}} = \log(2), u_{\text{medium}} = \log(6), u_{\text{high}} = \log(15)$. Moreover, for each of these methods we initialize $\lambda^{(e)}$ for effort level $e$ equal based on the pareto curve $s'(c)$ derived above[^1]. 

We additionally test against Cognition's method of fixing the slope (i.e. using the value we initialize with throughout RL). We run all experiments for 1,000 training steps with code at this [repo](https://github.com/anish-lakkapragada/swe-2-extended). 

I think the figures in the video visualize the results pretty well, but I'll reproduce them here because why not: 

{% include swe2-training-figures.html %}

## Acknowledgements 

Many thanks to early feedback and support from [Mars Xiang](https://marsxiang.com/). And of course, we thank Cognition for releasing their training details.

## Citations 

Please cite this work as follows: 

<pre class="citation"><code>@article{anishlk-swe2-extended-reward,
  author  = {Anish Lakkapragada},
  title   = {extending swe-2's reward function to steer the pareto frontier},
  journal = {Anish Lakkapragada's Blog},
  year    = {2026},
  note    = {https://anishlk.com/swe-2-extended},
}</code></pre>

As always if you have any questions, please feel [get in touch](mailto:anish.lakkapragada@yale.edu).


[^1]: In the real world, we will not always have such a clean pareto curve with a known derivative. I choose this sample RL task to not introduce the difficulty in high-fidelity derivative estimates in our assessment of this adaptive $\lambda^{(e)}$'s success.
