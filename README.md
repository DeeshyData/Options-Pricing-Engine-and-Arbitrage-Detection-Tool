# Options Pricing and Arbitrage Detection Engine
A comprehensive Python toolkit for pricing options and detecting arbitrage opportunities in options markets.

## Features

### 1. Pricing Models

---

### Black-Scholes
The Black-Scholes model, developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, calculates the theoretical value of a European-style option. There are six parameters that affect option prices, which the model takes as inputs:
- $S$ = current stock price
- $K$ = strike price
- $T$ = time to maturity
- $\sigma$ = volatility
- $r$ = continuously compounded risk-free rate
- $q$ = continuously compounded dividend yield

### Assumptions of the Black-Scholes Model

#### Stock price dynamics

The stock price follows a geometric Brownian motion with constant drift and volatility

$$
dS_t = (r - q) S_t dt + \sigma S_t dW_t.
$$

This implies that:
- Log returns are normally distributed
- Volatility is constant
- Price paths are continuous

#### Constant risk-free rate
- The interest rate $r$ is constant and known
- Investors can borrow and lend unlimited amounts at the risk-free rate

#### No arbitrage
There are no risk-free profit opportunities in the market.

#### Frictionless markets
- No transaction costs
- No taxes
- No bid-ask spreads

### Option Prices
The price of a European call option $(C)$ and put option $(P)$ are:

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),
$$

$$
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),
$$

where:

$$
d_1 = \frac{\ln\left(\frac{S}{K}\right) + T \left(r - q + \frac{\sigma^2}{2}\right)}{\sigma \sqrt{T}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T}.
$$

---

### Monte Carlo Simulation
Monte Carlo simulation estimates the price of options by simulating future stock prices and calculating the payoffs at maturity. This method implements the same six parameters as the Black-Scholes model, with additional inputs `n_steps` and `n_paths` which are the number of time steps and number of simulation paths respectively. It uses a discretised geometric Brownian motion under the risk-neutral measure, where the stock follows:

$$
S_{t+\Delta t} = S_t \exp \left[\left(r - q - \frac{1}{2} \sigma^2 \right) \Delta t + \sigma \sqrt{\Delta t} Z \right],
$$

where:
- $S_t$ = the stock price at time $t$
- $\Delta t = T /$ `n_steps` = a small time step
- $Z \sim N(0, 1)$

---

### Binomial Trees
The Binomial tree model is a discrete-time method for pricing options. It approximates the continuous-time dynamics of the underlying asset by modelling price evolution over a finite number of steps. The Binomial models takes additional inputs `N` which is the number of time steps, and `option_style`, the style of the option (American or European). To calculate the price of an American option, the option price tree is calculated backwards by taking the maximum of the continuation price (discounted option price from one time step ahead) with the payoff for exercising at the given time step while also discounting to time $t$. For European options, the additional exercise steps are not calculated and only the continuation prices are evaluated to time 0.

#### Stock Price Dynamics
Over a small time step $\Delta t = T / N$, the stock price can move:
- Up by a factor $u$
- Down by a factor $d$

$$
S_{t + \Delta t} =
\begin{cases}
S_t u \\
S_t d
\end{cases}
$$

#### Cox-Ross-Rubinstein Model (CRR)
Under the CRR model

$$
u = e^{\sqrt{\Delta t}},
$$
$$
d = \frac{1}{u}.
$$

The risk-neutral probability is

$$
p = \frac{e^{\left(r - q\right) \Delta t} - d}{u - d},
$$

where $p$ is the probability of an up move, while $1 - p$ is the probability of a down move

#### Jarrow-Rudd Model (JR)
Under the JR model

$$
p = 0.5
$$
$$
u = \exp{\left[\left(r - q - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t}\right]},
$$
$$
d = \exp{\left[\left(r - q - \frac{\sigma^2}{2}\right)\Delta t - \sigma \sqrt{\Delta t}\right]}.
$$

#### Leisen-Reimer Model (LR)
Under the LR model

$$
u = e^{(r - q)\Delta T} \cdot \frac{p'}{p},
$$
$$
d = e^{(r - q)\Delta T} \cdot \frac{1 - p'}{1 - p}
$$

where:

$$
p = h^{-1}(d_2), \\
p' = h^{-1}(d_1)
$$

Here, $h^{-1}(\cdot)$ is the Peizer-Pratt Inversion function, given by:

$$
h^{-1}(z) = \frac{1}{2} + \frac{\text{sign}(z)}{2} \sqrt{1 - \exp{\left[-\left(\frac{z}{n + \frac{1}{3} + \frac{0.1}{n + 1}}\right)^2\left(n + \frac{1}{6}\right)\right]}}
$$

---

### 2. Greeks Calculations
#### Delta $\left(\Delta = \frac{\partial V}{\partial S}\right)$:
The closed-form is given by

$$
\Delta_C = e^{-qT} N(d_1),
$$
$$
\Delta_P = e^{-qT} \left[N(d_1) - 1\right].
$$

Represents the sensitivity of the option price to the price of the stock.

#### Gamma $\left(\Gamma = \frac{\partial^2 V}{\partial S^2}\right)$:

$$
\Gamma = \frac{e^{-qT}}{S \sigma \sqrt{T}} N'(d_1).
$$

Represents the rate of change of Delta.

#### Vega $\left(\nu = \frac{\partial V}{\partial \sigma}\right)$:

$$
\nu = S e^{-qT} N'(d_1) \sqrt{T}.
$$

Represents the sensitivity of the option price to volatility.

#### Theta $\left(\Theta = -\frac{\partial V}{\partial T}\right)$:

$$
\Theta_C = -\frac{S \sigma e^{-qT}}{2 \sqrt{T}} N'(d_1) - rK e^{-rT} N(d_2) + qS e^{-qT} N(d_1),
$$
$$
\Theta_P = -\frac{S \sigma e^{-qT}}{2 \sqrt{T}} N'(d_1) + rK e^{-rT} N(-d_2) - qS e^{-qT} N(-d_1).
$$

Represents the sensitivity of the option price to time to maturity.

#### Rho $\left(\rho = \frac{\partial V}{\partial r}\right)$:

$$
\rho_C = KT e^{-rT} N(d_2),
$$
$$
\rho_P = KT e^{-rT} N(-d_2).
$$

Represents the sensitivity of the option price to the risk-free rate of interest.

### 3. Implied Volatility
The implied volatility is the value of volatility that makes the model price match the observed market price.

It solves:

$$
V_{\text{model}}(S, K, T, \sigma, r, q) = V_{\text{market}}
$$

There is no closed-form solution for implied volatility, and so it must be computed numerically.

#### Newton-Raphson Method
The Newton-Raphson Method recursively solves for the implied volatility until the two prices are within a certain tolerance of each other. If it is unable to find a solution within the number of iterations, then it returns an error.

$$
\sigma_{n + 1} = \sigma_n - \frac{V(\sigma_n) - V_{\text{market}}}{\text{Vega}(\sigma_n)}
$$

#### Brent's Method
Brent's method is a robust, derivative-free algorithm for solving nonlinear equations of the form:

$$
f(\sigma) = 0
$$

The algorithm dynamically chooses the most efficient step while maintaining a bracketing interval that contains the root.

The formula is:

$$
f(\sigma) = V_{\text{model}}(\sigma) - V_{\text{market}}
$$

Two methods for determining the implied volatility of the option were implemented. The first being the Newton-Raphson method and the second being Brent's method. The calculated implied volatility was compared with the observed volatility 


### 4. Arbitrage
Identify multiple types of arbitrage opportunities using different strategies:
- Put-call parity
- Box spreads
- Market price vs theoretical price
- Options bounds

---

### 5. Integrating Market Data
- Fetch real-time options data using `yfinance`
- Calculate implied volatilities
- Scan for arbitrage opportunities

## Project Structure
```
options_pricing_engine/
│
├── src/
│   ├── analysis/
│   │   ├── fd_greeks.py
│   │   └── iv_solver.py
│   │
│   ├── arbitrage/
│   │   └── detector.py
│   │
│   ├── models/
│   │   ├── binomial.py
│   │   ├── black_scholes.py
│   │   └── monte_carlo.py
│   │
│   ├── fetch_data.py
│   ├── main.ipynb
│   └── option_data.py
│
├── README.md
└── requirements.txt
```

### References
- [Black-Scholes Model - Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp)
- [Black-Scholes Formulas - macroption](https://www.macroption.com/black-scholes-formula/)
- [Cox-Ross-Rubinstein Model Formulas - macroption](https://www.macroption.com/cox-ross-rubinstein-formulas/)
- [Jarrow-Rudd Model Formulas - macroption](https://www.macroption.com/jarrow-rudd-formulas/)
- [Leisen-Reimer Model Formulas - macroption](https://www.macroption.com/leisen-reimer-formulas/)
