import numpy as np
from ..option_class import Option
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholes:
    VALID_OPTION_TYPES = {'call', 'put'}

    def __init__(self, option: Option):
        """
        Initialise paramters for Black-Scholes model
 
        :param option: The data of the option (option_type, S, K, T, sigma, r, q)
        """
        self.option = option

    def _d1(self):
        """Calculate d1"""
        return (np.log(self.option.S / self.option.K) + (self.option.r - self.option.q + 0.5 * self.option.sigma**2) * self.option.T) / \
               (self.option.sigma * np.sqrt(self.option.T))

    def _d2(self):
        """Calculate d2"""
        return self._d1() - self.option.sigma * np.sqrt(self.option.T)

    def price(self):
        """Calculate the price of a European option"""
        if self.option.option_type == "call":
            if self.option.T == 0:
                return max(self.option.S - self.option.K, 0)
            return self.option.S * np.exp(-self.option.q * self.option.T) * norm.cdf(self._d1()) - \
                   self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(self._d2())
        else:   # self.option.option_type == "put"
            if self.option.T == 0:
                return max(self.option.K - self.option.S, 0)
            return self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(-self._d2()) - \
                   self.option.S * np.exp(-self.option.q * self.option.T) * norm.cdf(-self._d1())

    def delta(self):
        """Calculate the Delta of the option"""
        if self.option.option_type == "call":
            return np.exp(-self.option.q * self.option.T) * norm.cdf(self._d1())
        else:   # self.option.option_type == "put"
            return np.exp(-self.option.q * self.option.T) * (norm.cdf(self._d1()) - 1)

    def gamma(self):
        """Calculate the Gamma of the option"""
        return np.exp(-self.option.q * self.option.T) * norm.pdf(self._d1()) / (self.option.S * self.option.sigma * np.sqrt(self.option.T))

    def vega(self):
        """Calculate the Vega of the option"""
        return self.option.S * np.exp(-self.option.q * self.option.T) * norm.pdf(self._d1()) * np.sqrt(self.option.T)

    def theta(self):
        """Calculate the Theta of the option"""
        if self.option.option_type == "call":
            return -(self.option.S * np.exp(-self.option.q * self.option.T) * norm.pdf(self._d1()) * self.option.sigma) / (2 * np.sqrt(self.option.T)) - \
                     self.option.r * self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(self._d2()) + \
                     self.option.q * self.option.S * np.exp(-self.option.q * self.option.T) * norm.cdf(self._d1())
        else:   # self.option.option_type == "put"
            return -(self.option.S * np.exp(-self.option.q * self.option.T) * norm.pdf(self._d1()) * self.option.sigma) / (2 * np.sqrt(self.option.T)) + \
                     self.option.r * self.option.K * np.exp(-self.option.r * self.option.T) * norm.cdf(-self._d2()) - \
                     self.option.q * self.option.S * np.exp(-self.option.q * self.option.T) * norm.cdf(-self._d1())

    def rho(self):
        """Calculate the Rho of the option"""
        if self.option.option_type == "call":
            return self.option.K * self.option.T * np.exp(-self.option.r * self.option.T) * norm.cdf(self._d2())
        else:   # self.option.option_type == "put"
            return -self.option.K * self.option.T * np.exp(-self.option.r * self.option.T) * norm.cdf(-self._d2())

    def greeks(self):
        """Get a table of all the greeks"""
        return {
            'Delta': self.delta(),
            'Gamma': self.gamma(),
            'Vega': self.vega() / 100,
            'Theta': self.theta() / 365,
            'Rho': self.rho() / 100
        }

    def iv_newton_raphson(self, market_price, tolerance, max_iterations, sigma):
        option = Option(self.option.option_type, self.option.S, self.option.K, self.option.T, sigma, self.option.r, self.option.q)
        for _ in range(max_iterations):
            option = BlackScholes(self.option.option_type, self.option.S, self.option.K, self.option.T, self.option.r, sigma_imp, self.option.q)
            bs_price = option.price()

            res = bs_price - market_price
            if abs(res) < tolerance:
                return sigma_imp

            vega = option.vega()
            if vega < 1e-8:
                break

            sigma_imp -= res / vega
        raise RuntimeError("Implied volatility did not converge")

    def implied_volatility_brent(self, market_price: float, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Calculate the implied volatility of the option using Brent's method

        :param market_price: current price of the option
        :param max_iterations: number of iterations to itrerate through the optimisation function
        :param tolerance: Tolerance for the implied volatility
        :param T: Time to maturity in years
        :param r: Risk-free rate
        :param sigma: Volatility
        :param q: Dividend yield rate
        """
        sigma = 0.2
        bs_option = BlackScholes(self.option.option_type, self.option.S, self.option.K, self.option.T, self.option.r, sigma, self.option.q)

        def objective(sigma):
            price = bs_option.price()
            return price - market_price

        implied_vol = brentq(objective, 1e-6, 5, maxiter=max_iterations, xtol=tolerance)
        if implied_vol:
            self.option.sigma = implied_vol
            return implied_vol
        else:
            print("Could not find a valid implied volatility")
            return None

    def partial_differential_equation(self):
        return self.theta() + 0.5 * self.option.sigma**2 * self.option.S**2 * self.gamma() + self.option.r * self.option.S * self.delta() - self.option.r * self.price()