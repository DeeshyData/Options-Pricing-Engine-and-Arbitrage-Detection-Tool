from scipy.optimize import brentq
from ..option_class import Option
from ..models.black_scholes import BlackScholes

class ImpliedVolatilitySolver:
    VALID_OPTION_TYPES = {'call', 'put'}
    VALID_OPTION_STYLES = {'american', 'european'}

    def __init__(
            self,
            market_price: float,
            option_type: str,
            S: float,
            K: float,
            T: float,
            r: float,
            q: float = 0,
            lower_vol: float = 1e-6,
            upper_vol: float = 5,
            max_iterations: int = 100,
            tolerance: float = 1e-6
        ):
        """
        Initialise the parameters for the implied volatility solver

        :param market_price: Current market price of the option
        :param option_type: 
        :param max_iterations: Number of iterations to itrerate through the optimisation function
        :param tolerance: Tolerance for the implied volatility
        :param T: Time to maturity in years
        :param r: Risk-free rate
        :param sigma: Volatility
        :param q: Dividend yield rate
        """
        self.market_price = market_price
        self.option_type = option_type
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.lower_vol = lower_vol
        self.upper_vol = upper_vol
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._validate_inputs()

    def _validate_inputs(self):
        """Checks if the provided option data is valid"""
        if self.market_price <= 0:
            raise ValueError("Market price of the option must be greater than 0")
        if self.option_type not in self.VALID_OPTION_TYPES:
            raise ValueError(f"Invalid option type '{self.option_type}'. Must be {self.VALID_OPTION_TYPES}")
        if self.S <= 0:
            raise ValueError("Stock price must be greater than 0")
        if self.K <= 0:
            raise ValueError("Strike price must be greater than 0")
        if self.T < 0:
            raise ValueError("Time to maturity must be positive")
        if self.lower_vol <= 0:
            raise ValueError("Lower volatility bound must be greater than 0")
        if self.upper_vol <= self.lower_vol:
            raise ValueError("Upper volatility bound must be greater than lower volatility bound")
        if self.max_iterations <= 0:
            raise ValueError("Maximum iterations must be greater than 0")
        if self.tolerance < 0:
            raise ValueError("Tolerance must be greater than or equal to 0")

    def iv_newton_raphson(self, sigma=0.2):
        """
        Calculate the implied volatility of the option using Newton-Raphson method
        
        :param sigma: Initial sigma with which to iterate through
        """
        option = Option(self.option_type, self.S, self.K, self.T, sigma, self.r, self.q)
        bs = BlackScholes(option)

        for _ in range(self.max_iterations):
            bs.option.sigma = sigma
            price = bs.price()
            vega = bs.vega()

            diff = price - self.market_price

            if abs(diff) < self.tolerance:
                self.sigma = sigma
                return sigma

            if abs(vega) < 1e-8:
                break

            sigma -= diff / vega
            sigma = min(max(sigma, self.lower_vol), self.upper_vol)

        print("Newton-Rapshon implied volatility did not converge")
        return None

    def iv_brent(self):
        """Calculate the implied volatility of the option using Brent's method"""
        option = Option(self.option_type, self.S, self.K, self.T, 0.1, self.r, self.q)
        bs = BlackScholes(option)

        def objective(sigma):
            bs.option.sigma = sigma
            return bs.price() - self.market_price

        lower = objective(self.lower_vol)
        upper = objective(self.upper_vol)

        if lower * upper > 0:
            print(f"No roots exist in volatility range: [{self.lower_vol}, {self.upper_vol}]")
            return None

        implied_vol = brentq(objective, self.lower_vol, self.upper_vol, maxiter=self.max_iterations, xtol=self.tolerance)

        self.sigma = implied_vol
        return implied_vol