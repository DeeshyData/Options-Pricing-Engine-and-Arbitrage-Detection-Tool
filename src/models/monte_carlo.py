import numpy as np
from ..option_class import Option

class MonteCarlo:
    VALID_OPTION_TYPES = {'call', 'put'}

    def __init__(self, option: Option, n_steps: int = 252, n_paths: int = 100_000):
        """
        Initialise paramters for Monte Carlo simulation

        :param option: The data of the option (option_type, S, K, T, sigma, r, q)
        :param n_steps: The number of time steps to simulate
        :param n_paths: The number of simulation paths
        """
        self.option = option
        self.n_steps = n_steps
        self.n_paths = n_paths

        self._validate_inputs()

    def _validate_inputs(self):
        if self.n_steps <= 0:
            raise ValueError(f"Number of steps must be greater than 0")
        if self.n_paths <= 0:
            raise ValueError(f"Number of simulation paths must be greater than 0")

    def _simulate_paths(self):
        """Simulates geometric brownian motion paths for the stock price"""
        dt = self.option.T / self.n_steps
        Z = np.random.standard_normal((self.n_paths, self.n_steps))

        drift = (self.option.r - self.option.q - 0.5 * self.option.sigma**2) * dt
        diffusion = self.option.sigma * np.sqrt(dt)

        returns = drift + diffusion * Z
        log_prices = np.log(self.option.S) + np.cumsum(returns, axis=1)
        paths = np.exp(log_prices)
        
        return paths

    def price(self):
        """Calculates the price of a European-style option"""
        paths = self._simulate_paths()
        ST = paths[:, -1]

        if self.option.option_type == "call":
            payoffs = np.maximum(ST - self.option.K, 0)
        else:    # self.option.option_type == "put"
            payoffs = np.maximum(self.option.K - ST, 0)

        discounted_payoffs = np.exp(-self.option.r * self.option.T) * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_paths)

        return price, std_error
