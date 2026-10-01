---
layout: post
title: extending swe-2's reward function to steer the pareto frontier
permalink: /swe-2-extended/
date: 2026-09-25 00:00:00 -0700
description: Experiments extending the reward function in Cognition's SWE-2 model release.
image:
  path: /assets/swe-2-extended/social-preview-v4.jpg
  width: 2400
  height: 1260
  alt: "Extending SWE-2's Reward Function to steer the Pareto Frontier: a compass of colored training trajectories, one heading per alpha."
math: true
wide: true
back: /
---

This is the full explanation of a project I worked on to extend SWE-2’s reward function. For a short summary, please see the video below!
{% include swe2-video.html %}

## What did Cognition do in SWE-2?

Earlier in September, Cognition released [SWE-2](https://cognition.com/blog/swe-2), which made great progress on the capability vs. cost Pareto curve, as measured by Cognition’s own benchmark, [FrontierCode 1.1](https://cognition.com/blog/frontier-code-1.1).

In its blog post, Cognition explains how SWE-2’s post-training reward function is explicitly constructed to improve the Pareto curve. Rigorously speaking, the reward function for a given effort level $$e$$ (i.e., medium, high, or max) is given by $$R(S, C) = S - \lambda^{(e)} C$$, where $$S$$ is the average solve rate, $$C$$ is the average cost, and $$\lambda^{(e)} > 0$$. What Cognition [elegantly derives](https://cognition.com/blog/swe-2#appendix-b) is that the slope $$\lambda^{(e)}$$ should be set to the slope of the Pareto curve of the base model at effort level $$e$$.

For a visual explanation, please see the <a href="/assets/swe-2-extended/approximating-pareto-curve-cog.png" target="_blank" rel="noopener noreferrer"><em>“Approximating the Pareto curve tangents of Kimi K3”</em></a> figure from the SWE-2 blog post.

## Problem Statement: Steering Pareto Frontier Improvements 

Overall, this methodology is great, as it yields improvements to the Pareto frontier throughout post-training. The key thing to note here is that while this “slope-matched penalty” will push the Pareto frontier, it is unopinionated about *where* the model should land on this frontier. This can be seen in the <a href="/assets/swe-2-extended/slope-matched-penalty.png" target="_blank" rel="noopener noreferrer">slope-matched penalty figure</a>, where any point along an iso-reward line gets the same reward.

**But what if we could steer the direction in which the Pareto frontier improved?** More explicitly, there are two clear basis vectors along which to improve, with a spectrum in between:

- Same performance, lower cost 
- Higher performance, same cost 

How do we target our preferred balance? 

## Solution: Update $$\lambda^{(e)}$$ throughout post-training based on the performance-cost tradeoff parameter $$\alpha \in [0, 1]$$

We now derive the update rule, given in the red box at the bottom of this section.

If we define $$e$$ to be some categorical variable representing the effort level (low/medium/high) and $$\pi$$ to be our policy, we can define the following two metrics: 

$$
s_e(\pi) = \mathbb{E}_{\pi}[S \mid e], \quad c_e(\pi) = \mathbb{E}_{\pi}[C \mid e] 
$$

These are the average solve rate and average cost, respectively, for effort level $$e$$. The base model's performance at this effort level can similarly be given by $$s_{e, 0} = \mathbb{E}_{\pi_{0}}[S \mid e]$$ and $$c_{e, 0} = \mathbb{E}_{\pi_{0}}[C \mid e]$$, where $$\pi_0$$ is the initial base model policy. Using this setup, the relative improvement of a new policy $$\pi$$ over $$\pi_0$$ can be quantified as follows:

$$
u_e(\pi) = \frac{s(\pi) - s_{e, 0}}{s_0}, \quad v_e(\pi) = \frac{c_{e, 0} - c_e(\pi)}{c_{e, 0}}
$$

Our goal then is to see how we can optimize $$\pi$$ to maximize improvements in success, $$u_e(\pi)$$, and cost, $$v_e(\pi)$$, while balancing the two. **This is done through a parameter $$\alpha \in [0, 1]$$, which we set.** This can be written as a constrained optimization problem over the real-valued quantity $$\tau > 0$$:

$$
\boxed{ \begin{aligned} \max_{\pi,\tau}\quad & \tau \\[4pt] \text{subject to}\quad & u_e(\pi) \ge \alpha\tau, \\ & v_e(\pi) \ge (1-\alpha)\tau. \end{aligned} }
$$

The endpoints have the following interpretations:

- $$\alpha = 0$$ means we demand $$u_e(\pi) \geq 0$$ and $$v_e(\pi) \geq \tau \implies$$ we only care about cost improvement while keeping success the same.
- $$\alpha = 1$$ conversely means we only care about success improvement while keeping cost the same.

Choosing $$0 < \alpha < 1$$ lets us optimize along the spectrum in between. 

We now derive the solution.

<details class="details-block derivation" markdown="1">
<summary>Solution Derivation</summary>

To start, we’ll first assume $$\alpha \in (0, 1)$$ and deal with the endpoints later. Under this assumption, the constraints give:

$$
\begin{aligned}
\tau &\leq \frac{u_e(\pi)}{\alpha}, \\[6pt]
\tau &\leq \frac{v_e(\pi)}{1 - \alpha}.
\end{aligned}
$$

The maximum value of $$\tau$$ is the smaller of these two bounds. Since $$\max_{\pi, \tau} \tau = \max_{\pi} (\max \tau)$$, our original constrained optimization problem can be re-expressed as:

$$
\max_{\pi} \min\left(\frac{u_e(\pi)}{\alpha}, \frac{v_e(\pi)}{1 - \alpha}\right).
$$

Since $$\alpha, 1 - \alpha \in \mathbb{R}_{> 0}$$, this is equivalent to:

$$
\max_{\pi} \min\bigl((1-\alpha) u_e(\pi), \alpha v_e(\pi)\bigr).
$$

Now, in standard math fashion, we will pull an identity out of our ass:

$$
\min(a, b) = \min_{\beta \in [0, 1]} [\beta a + (1 - \beta) b]
$$ 

Applying this identity, we can write our optimization problem as:

$$
\max_{\pi} \min_{\beta \in [0, 1]} \beta(1 - \alpha)u_e(\pi) + (1 - \beta)\alpha v_e(\pi)
$$

Expanding this expression, we get:

$$
\max_{\pi} \min_{\beta \in [0, 1]} \underbrace{\frac{\beta(1 - \alpha) s_e(\pi)}{s_{e, 0}} - \frac{(1 - \beta)\alpha c_e(\pi)}{c_{e, 0}} + \text{const. w.r.t. $\pi, \beta$}}_{L(\pi, \beta)}
$$

We call this function $$L(\pi, \beta)$$. Notice that if we *fix* $$\beta$$ and multiply $$L(\pi, \beta)$$ by the positive factor $$s_{e, 0}/[\beta(1 - \alpha)]$$, we get:

$$
\begin{aligned}
R_e(\pi, \beta) &:= s(\pi) - \lambda^{(e)} c(\pi), \\[6pt]
\lambda^{(e)} &:= \frac{s_{e, 0}}{c_{e, 0}}\,\frac{\alpha}{1 - \alpha}\,\frac{1 - \beta}{\beta}.
\end{aligned}
$$

This looks like the SWE-2 reward function, right? The main point here is that the reward function $$\min_{\beta \in [0, 1]} L(\pi, \beta)$$ we will be updating $$\pi$$ against is exactly the same as in SWE-2, just with $$\lambda$$ as a function of $$\beta$$.

The remaining question is how to solve $$\min_{\beta \in [0, 1]} L(\pi, \beta)$$ to obtain a reward function for $$\pi$$ (which can then be RL’d against with your favorite policy optimization algorithm). To start with, recall that $$u_e(\pi)$$ and $$v_e(\pi)$$ rely on expectations. For this iteration, let’s define our empirical estimates as:

$$
\begin{aligned}
\hat{u}_e &:= \frac{\hat{s_e}(\pi) - s_{e, 0}}{s_{e, 0}}, \\[6pt]
\hat{v}_e &:= \frac{c_{e, 0} - \hat{c_e}(\pi)}{c_{e, 0}}.
\end{aligned}
$$

Then one way we can update $$\beta$$, given $$\beta_{\text{prev}}$$ as the previous iteration’s value, is to use standard optimization over $$L(\pi, \beta)$$ with a KL term on $$\beta$$. Concretely:

$$
\beta \gets \operatorname*{arg\,min}_{0 \leq \beta \leq 1}
\textcolor{blue}{\left[
\begin{aligned}
&\beta(1 - \alpha) \hat{u}_e + (1 - \beta)\alpha \hat{v}_e \\[6pt]
&\quad + \frac{1}{\eta} D_{\text{KL}}\bigl((\beta, 1 - \beta) \parallel (\beta_{\text{prev}}, 1 - \beta_{\text{prev}})\bigr)
\end{aligned}
\right]}
$$  

where $$\eta > 0$$. To be clear, the KL divergence can be given as: 

$$ 
\begin{aligned}
&D_{\text{KL}}\bigl((\beta, 1 - \beta) \parallel (\beta_{\text{prev}}, 1 - \beta_{\text{prev}})\bigr) \\[6pt]
&\qquad = \beta \log \frac{\beta}{\beta_{\text{prev}}}
+ (1 - \beta) \log \frac{1 - \beta}{1 - \beta_{\text{prev}}}.
\end{aligned}
$$

Taking the derivative of the <span style="color: blue;">blue term</span> above w.r.t. $$\beta$$ and setting it to zero, we get:

$$
(1 - \alpha) \hat{u}_e - \alpha \hat{v}_e + \frac{1}{\eta}[\log \frac{\beta}{1 - \beta} + \log \frac{1 - \beta_{\text{prev}}}{\beta_{\text{prev}}}] = 0
$$

This yields the update rule:

$$
\log \frac{1 - \beta}{\beta} = \log \frac{1- \beta_{\text{prev}}}{\beta_{\text{prev}}} + \eta[(1 - \alpha)\hat{u}_e - \alpha \hat{v}_e]
$$

Since $$\lambda^{(e)} \propto (1 - \beta)/\beta$$, we obtain the update rule:

</details>

<div class="update-rule" markdown="1" style="border: 2px solid #c62828; border-radius: 6px; padding: 0 1rem; margin: 1rem auto; width: fit-content; max-width: 100%; box-sizing: border-box;">

$$
\log (\lambda^{(e)}) \gets \log(\lambda_{\text{prev}}^{(e)}) + \eta[(1 - \alpha)\hat{u}_e - \alpha \hat{v}_e]
$$

</div>

Note that this extends cleanly to $$\alpha \in \{0, 1\}$$, and hence we are happy.

*Implementation Note*: Because we are choosing $$\alpha$$ and using it as a targeted direction for many sequential runs, $$s_{e, 0}$$ and $$c_{e, 0}$$ are benchmarked against the *initial* policy $$\pi_0$$, as opposed to the policy from just the previous iteration.

## Controlled Setup: Multi-effort RL Task with 5 Trainable Parameters

We now explain our setup for testing this method on a toy RL task supporting multiple effort levels and involving a total of five trainable parameters. A given rollout works as follows, with fixed probabilities satisfying $0 \leq p_{\text{bad}} \leq p_{\text{good}} \leq 1$:

1. Sample a given task type $x$ uniformly from $$\{0, 1\}$$.
2. Give the policy $\pi$ the task $x$ and an effort level $e$. Effort levels can be low, medium, or high.
3. Have the policy sample a tool $$a \in \{0, 1\}$$ and a candidate count $K$.
4. Generate $K$ candidates, where each candidate succeeds with the following probability:

$$ 
p(x, a) = \begin{cases}
    p_{\text{good}} & a = x \\ 
    p_{\text{bad}} & a \neq x
\end{cases}
$$

5. Return success if at least one candidate passes, and return the cost as $K$. Note that $K = 0$ means the success rate is zero.

And that's it! Our reward function in all scenarios will be $R = S - \lambda^{(e)} K$.

**We now define the five-parameter policy.** To start, define logits $z_0, z_1$, where $\rho_x = \sigma(z_x) = \frac{1}{1 + \exp(-z_x)}$ and the policy samples $a \mid x \sim \text{Bern}(\rho_x)$. In other words, $\rho_x$ gives the probability that the policy chooses tool 1 on task type $x$.

Note that we can then give the probability of picking the correct tool (i.e., $a = x$) as $q = \frac{(1 - \rho_0) + \rho_1}{2}$ because both task types are equally likely. Because we initialize $z_0 = z_1 = 0$, $q_0 = \frac{1}{2}$. These two logits control the policy's ability to succeed.

The remaining three parameters dictate how the policy (independently) decides the cost $K$. Specifically, for each effort level $e$ (low/medium/high), we have a trainable parameter $u_e$, where the total number of candidates is sampled according to $K \sim \text{Geom}(\frac{1}{1 + \exp(u_e)}) - 1$, so the average cost incurred for effort level $e$ is $\mathbb{E}[K \mid e] = \exp(u_e)$.

So the five trainable parameters are $\theta = (z_0, z_1, u_{\text{low}}, u_{\text{medium}}, u_{\text{high}})$. Now, an additional nice thing about this task is that it has a closed-form Pareto curve $s(c) = \mathbb{P}[S = 1 \mid c]$. In other words, average success can be written as a function of average cost. It is derived below.

<details class="details-block derivation" markdown="1">
<summary>Pareto Curve Derivation</summary>

First, suppose we are given a fixed task type $x$ with a selected effort level $e$. We have chosen a tool with a probability $p$ of success on this task type. Moreover, this effort level currently yields an average cost $c := \mathbb{E}[K \mid e]$ through the value of $u_e$. Defining $r := 1/(1 + c)$ (recall that $K \sim \text{Geom}(r) - 1$), we get:

$$
\begin{aligned}
\mathbb{P}(S = 0 \mid p, c)
&= \sum_{k =0}^{\infty} \mathbb{P}(K = k) \cdot \mathbb{P}(S = 0\mid p, c, K = k) \\[6pt]
&= \sum_{k =0}^{\infty} r(1 - r)^k \cdot (1 - p)^k \\[6pt]
&= r \sum_{k =0}^{\infty} [(1 - r) \cdot (1 - p)]^k \\[6pt]
&= r \cdot \frac{1}{1 - (1 - r) \cdot (1 - p)} \\[6pt]
&= \frac{r}{r + p(1 - r)} \\[6pt]
&= \frac{1/(1 + c)}{\frac{(1 + pc)}{(1 + c)}} \\[6pt]
&= \frac{1}{1 + pc}
\end{aligned}
$$

Thus, $\mathbb{P}(S = 1 \mid p, c) = pc/(1 + pc)$. If $q$ is the probability of picking the correct tool (determined by the $z_x$ logits), then the law of total probability (LOTP) gives:

</details>

$$
s(c) = \mathbb{P}[S = 1 \mid c] =  q \frac{p_{\text{good}}c}{1 + p_{\text{good}}c} + (1 - q)\frac{p_{\text{bad}}c}{1 + p_{\text{bad}}c}
$$

To evaluate this function for any given effort level, just pass in $c = \exp(u_e)$. The benefit of this is that we can easily compute the derivative $s'(c)$ to find the slope of the Pareto curve, which we use to initialize $\lambda^{(e)}$ in our experiments.

## Controlled Experiment Results

We test 100 different instances of the described task, where each problem has a different pair of values $(p_{\text{good}}, p_{\text{bad}})$. For each problem, we run the experiment with three seeds for the policy RNG. We initialize the tool logits with $z_0 = z_1 = 0$ and the effort parameters as follows:

$$
u_{\text{low}} = \log(2), \quad u_{\text{medium}} = \log(6), \quad u_{\text{high}} = \log(15).
$$

Moreover, for each of these methods, we initialize $\lambda^{(e)}$ for effort level $e$ based on the slope $s'(c)$ of the Pareto curve derived above[^1].

We additionally test against Cognition's method of fixing the slope (i.e., using the initial value throughout RL). We run all experiments for 1,000 training steps using the code in this [repo](https://github.com/anish-lakkapragada/swe-2-extended).

See figures below. 

{% include swe2-training-figures.html %}

## Acknowledgements 

Many thanks to [Mars Xiang](https://marsxiang.com/), [Neil Kale](https://neilkale.github.io/), [Rex Liu](https://rexliu.com/), and [Marc Melikyan](https://wqi.wisc.edu/wqcc/staff/marc-melikyan/), for their early feedback and support. And of course, we thank Cognition for releasing its training details.

## Citations 

Please cite this work as follows: 

<pre class="citation"><code>@article{anishlk-swe2-extended-reward,
  author  = {Anish Lakkapragada},
  title   = {extending swe-2's reward function to steer the pareto frontier},
  journal = {Anish Lakkapragada's Blog},
  year    = {2026},
  note    = {https://anishlk.com/swe-2-extended},
}</code></pre>

As always, if you have any questions, please feel free to [get in touch](mailto:anish.lakkapragada@yale.edu).


[^1]: In the real world, we will not always have such a clean Pareto curve with a known derivative. I chose this sample RL task to avoid introducing the difficulty of obtaining high-fidelity derivative estimates into our assessment of the success of this adaptive $\lambda^{(e)}$.
