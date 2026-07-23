---
layout: plain
title: "Derivation of Black-Scholes-Merton PDE under Risk-Neutral Probability Measure"
kind: derivation
order: 1
math: true
wide: true
back: /notes/
---

*This Black-Scholes-Merton derivation is not the standard delta-hedging derivation. It is outlined [here](https://en.wikipedia.org/wiki/Black–Scholes_equation#Alternative_derivation) on the Wikipedia page for the Black-Scholes-Merton equation.* 

We first assume a risk-free interest rate of $$r$$ (i.e. it is guaranteed that $$\$x$$ today can be lent to receive $$\$ e^{r\tau}x$$ in $$\tau$$ days). Now suppose we have a stock $$S_t$$ along with its derivative $$V = V(S_t, t)$$. From the risk-neutral measure, we get (1) the SDE $$dS_t = r S_t dt + \sigma S_t dW_t$$ and (2) that the process $$U_t := f(S_t, t) := e^{-rt}V(S_t, t)$$ is a martingale. Applying Itô's Lemma, we get an understanding of $$dU_t$$: 

$$
dU_t = (\frac{\partial f}{\partial t} + rS_t\frac{\partial f}{\partial S_t} +\frac{\sigma^2 S_t^2}{2} \frac{\partial^2 f}{\partial S_t^2} ) dt + \sigma S_t \frac{\partial f}{\partial S_t} dW_t
$$

Note that because $$U_t$$ is a martingale under risk-neutral probability measure $$\implies U_t$$ has no drift (see informal proof below). We give all relevant partials in the drift term below: 

$$
\frac{\partial f}{\partial t} = e^{-rt}[\frac{\partial V}{\partial t}-rV(S_t, t)], \quad \frac{\partial f}{\partial S_t} = e^{-rt} \frac{\partial V}{\partial S_t}, \quad \frac{\partial^2 f}{\partial S_t^2} = e^{-rt} \frac{\partial^2 V}{\partial S_t^2}
$$

and so plugging them into the drift term and solving for zero we arrive with: 

$$
e^{-rt}[\frac{\partial V}{\partial t}-rV(S_t, t)] + rS_t e^{-rt} \frac{\partial V}{\partial S_t} + \frac{\sigma^2 S_t^2}{2}  e^{-rt} \frac{\partial^2 V}{\partial S_t^2} = 0
$$

Or after some simplification: 

$$
\boxed{\frac{\partial V}{\partial t} + rS_t \frac{\partial V}{\partial S_t} + \frac{\sigma^2 S_t^2}{2}  \cdot  \frac{\partial^2 V}{\partial S_t^2} = rV}
$$

which is the Black-Scholes-Merton PDE.
