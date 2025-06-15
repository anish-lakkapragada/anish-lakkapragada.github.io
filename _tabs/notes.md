---
# the default layout is 'page'
icon: fas fa-info-circle
order: 2
math: true 

---

## college

- Yale S&DS 242: [Sandwich Asymptotic Variance](/notes/s&ds-242/Sandwich_Variance.pdf)

## handy derivations

Below is a list of quick derivations. I put them here for my own edification and public viewing if helpful.


<details>

<summary> Derivation for Two-Sided Ockham's Razor Bound </summary>

<em> Notation follows from these <a href="https://users.cs.duke.edu/~cynthia/CourseNotes/StatisticalLearningTheoryNotes.pdf">lecture notes</a>.</em> <br/>

Let us define our finite (binary) function class as $\mathcal{F} = \{f_1, \dots, f_M \}$ where each function in $\mathcal{F}$ predicts either $-1$ or $1$. Notation $\mathbf{Z} \sim D^n$ indicates that a probability is taken over the randomness of data draws $Z_1, \dots, Z_n \sim D$ where each $Z_i = (X_i, Y_i)$. Then $\forall \epsilon > 0$ we have: 

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \epsilon] \leq \sum_{j = 1}^M \mathbb{P}_{\mathbf{Z} \sim D^n}[| R^{\text{true}}(f_j) - R^{\text{emp}}(f_j)| > \epsilon]
$$

Applying the two-sided Hoeffding's Inequality, we know that $\mathbb{P}_{\mathbf{Z} \sim D^n}[\mid R^{\text{true}}(f_j) - R^{\text{emp}}(f_j) \mid > \epsilon] \leq 2\exp(-2n\epsilon^2)$ and so: 

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \epsilon] \leq 2M\exp(-2n\epsilon^2)
$$

To express this more nicely, let us define $\delta = 2M \exp(-2n\epsilon^2)$ so we can express $\epsilon = \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}$.

$$
\mathbb{P}_{\mathbf{Z} \sim D^n}[\exists \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| > \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}] \leq \delta
$$

$$
\implies \mathbb{P}_{\mathbf{Z} \sim D^n}[\forall \ f  \in \mathcal{F} : | R^{\text{true}}(f) - R^{\text{emp}}(f)| \leq \sqrt{\frac{\log M + \log \frac{2}{\delta}}{2n}}] \geq 1 - \delta
$$

which is exactly the two-sided Ockham's Razor Bound.

</details>

<details>

<summary> L2 Regularization: Gaussian Prior on Weights for Linear Regression </summary>

We first assume that $y = \mathbf{x}^T \mathbf{w} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Second, we assume that weights $ \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I}) $. Using our posterior distribution $ \mathbf{w} \mid \mathbf{y}, \mathbf{X}$, we can get an understanding of $\mathbf{w}_{\text{MAP}}$: 

$$ \ 
f(\mathbf{w} \mid \mathbf{y}, \mathbf{X}) \propto f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) f(\mathbf{w}) \implies \mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) f(\mathbf{w})]
$$

$$
\implies \mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [\log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) + \log f(\mathbf{w})] 
$$

Given that $ y_i \mid \mathbf{w}, \mathbf{x_i} \sim \mathcal{N}(\mathbf{x_i}^T \mathbf{w}, \sigma^2)$, we can give a nice understanding of $ \log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) $: 

$$
\log f(\mathbf{y} \mid \mathbf{w}, \mathbf{X}) = \sum_{i = 1}^n \log f(y_i \mid \mathbf{w}, \mathbf{x_i}) = \sum_{i = 1}^n - \log(\sqrt{2\pi \sigma^2}) - \frac{1}{2\sigma^2} (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2 = -\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2 + \text{const}
$$

where the constant is w.r.t to $\mathbf{w}$. For the log-density of our prior $\log f(\mathbf{w})$ we have: 

$$
\log f(\mathbf{w}) = \log[\exp(-\frac{1}{2} (\mathbf{w} - 0)^T (\tau^2 \mathbf{I})^{-1} (\mathbf{w} - 0) )] + \text{const} = -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}
$$

where $\mathbf{w}^T \mathbf{w}$ is just the square of the L2 norm. Putting this together we have: 

$$
\mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [-\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}] = \underset{\mathbf{w}}{\text{argmin}} \ [\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   + \frac{\sigma^2}{\tau^2} \mathbf{w}^T \mathbf{w}]
$$

Thus, we can conclude that MAP for weights under a Gaussian prior follows the objective as L2/Ridge Regression (albeit with a tuned $ \lambda $ resembling the $\frac{\sigma^2}{\tau^2}$) terms.

</details>

<details>
<summary> L1 Regularization: Laplace Prior on Weights for Linear Regression </summary>

For our weights $\mathbf{w} \in \mathbb{R}^d$, we assume each individual weight component is independent with prior $w_i \sim \text{Laplace}(0, b)$. So we get the following log-density for our weights: 

$$
\log f(\mathbf{w}) = \sum_{i = 1}^d f(w_i) = \sum_{i = 1}^d \frac{-|w_i - 0 |}{b} + \text{const.} = -\frac{1}{b} \sum_{i = 1}^d |w_i| + \text{const.}
$$

and so using identical work from the previous L2 derivation we get: 


$$
\mathbf{w}_{\text{MAP}} = \underset{\mathbf{w}}{\text{argmax}} \ [-\frac{1}{2\sigma^2}\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2  + \log f(\mathbf{w})] = \underset{\mathbf{w}}{\text{argmin}} \ [\sum_{i = 1}^n (\mathbf{y_i} - \mathbf{x_i}^T \mathbf{w})^2   + \frac{2\sigma^2}{b} \sum_{i = 1}^d |w_i|]
$$

Thus, we arrive at a similar conclusion that MAP for weights under a Laplace prior follows the same objective as L1 Regularization (where hyperparameter $\lambda$ is tuned to resemble the $\frac{2\sigma^2}{b}$ term.)
</details> 

<details>
<summary> Ridge Regression Closed Form Solution </summary>

While the closed-form Normal Equation solution & <a href="https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/"> derivation</a> is well-known, the closed-form solution for ridge regression is less so. Taken straight from the <a href="https://en.wikipedia.org/wiki/Ridge_regression"> Wikipedia page</a>, we provide our cost function below for data $\mathbf{X} \in \mathbb{R}^{n \times p}$, weights $\mathbf{\beta} \in \mathbb{R}^p$, and labels $\mathbf{y} \in \mathbb{R}^n$:

$$
J(\mathbf{\beta}) = (\mathbf{y} - \mathbf{X\beta})^T (\mathbf{y} - \mathbf{X\beta}) + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c) = (\mathbf{y}^T - \mathbf{\beta}^T\mathbf{X}^T)(\mathbf{y - X\beta}) + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c)
$$

$$
= \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X\beta} - \beta^T\mathbf{X}^T\mathbf{y} + \mathbf{\beta}^T\mathbf{X}^T\mathbf{X\beta} + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c)
$$

Note that $\mathbf{y}^T\mathbf{X\beta}$ and $\beta^T\mathbf{X}^T\mathbf{y}$ are both scalars and are transposes of each other. Thus, they are equal and so we can write: 

$$
J(\mathbf{\beta}) = \mathbf{y}^T\mathbf{y} - 2\beta^T\mathbf{X}^T\mathbf{y} + \mathbf{\beta}^T\mathbf{X}^T\mathbf{X\beta} + \lambda(\mathbf{\beta}^T\mathbf{\beta} - c) 
$$

We can think of this as essentially a Lagrange multiplier optimization problem, where our constraint is that $\mathbf{\beta}^T\mathbf{\beta} - c = 0$ for some $c \in \mathbb{R}$. We'll see that this choice of $c$ does not matter too much; it only matters that the constraint is there. We proceed: 

$$
0 = \frac{\partial J(\mathbf{\beta})}{\partial \mathbf{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X\beta} + \lambda(2\mathbf{\beta}) \implies \mathbf{X}^T\mathbf{y} = (\mathbf{X}^T\mathbf{X + \lambda I})\mathbf{\beta} 
$$

and so we get the final closed-form solution: 

$$
\mathbf{\beta} = (\mathbf{X}^T\mathbf{X + \lambda I})^{-1} \mathbf{X}^T\mathbf{y}
$$

</details> 

## just for fun

Across the spring term this year, I visited some friends at Berkeley and UC Davis multiple times! During my stay, I had the privilege of attending a good amount of classes. I took some notes[^note] for my fun and memories: 

[^note]: Whenever the professor didn't speak faster than my fingers could latex :)

- (Berkeley) _EECS 126: Probability & Random Processes_ ([my notes](/notes/berk/eecs126-reversiblemc-poissonprocess.pdf))
- (Berkeley) _EECS 127: Optimization Models in Engineering_ ([my notes](/notes/berk/eecs127-lineconvexity-convexduality-farkaslemma.pdf))
- (UC Davis)   *STA 141C: Big Data & High Performance Statistical Computing* ([my notes](/notes/davis/kmeans-hierarchical-clustering.pdf))

## high school

If you are searching for calculus-based notes to teach machine-learning at the high school or basic undergraduate level, please check out some [challenges](https://old-anish.lakkapragada.com/notes/) I (co)-wrote in high school.
