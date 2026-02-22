import numpy as np
from ..option_class import Option
from .black_scholes import BlackScholes

class Binomial:
    VALID_MODELS = ['crr', 'jr', 'lr']
    VALID_OPTION_STYLES = ['american', 'european']

    def __init__(self, option: Option, option_style: str, N: int = 1_000):
        """
        Initialise parameters for Binomial Model
        
        :param option: The data of the option (option_type, S, K, T, sigma, r, q)
        :param option_style: American or European
        :param N: The number of time steps
        """
        self.option = option
        self.option_style = option_style.lower()
        self.N = N

        self.dt = self.option.T / self.N

        self._validate_inputs()

    def _validate_inputs(self):
        """Validates the inputs for the Binomial model"""
        if self.option_style not in self.VALID_OPTION_STYLES:
            raise ValueError(f"Invalid option style '{self.option_style}'. Must be {self.VALID_OPTION_STYLES}")
        if self.N <= 0:
            raise ValueError(f"Number of steps must be greater than 0")

    def _calculate_option_price(self, u, d, p):
        """
        Calculates the price of an option discounting backwards
        
        :param u: The up move of the stock price
        :param d: The down move of the stock price
        :param p: The risk-neutral probability
        """
        ST = self.option.S * u**(np.arange(0, self.N + 1, 1)) * d**(np.arange(self.N, -1, -1))

        if self.option.option_type == "call":
            payoffs = np.maximum(ST - self.option.K, 0)
        else:   # self.option.option_type == "put"
            payoffs = np.maximum(self.option.K - ST, 0)

        for _ in range(self.N - 1, -1, -1):
            continuation = np.exp(-self.option.r * self.dt) * (p * payoffs[1:] + (1 - p) * payoffs[:-1])

            if self.option_style == "american":
                ST = ST[:-1] / d    # Stock prices at time previous step

                if self.option.option_type == "call":
                    payoffs = np.maximum(ST - self.option.K, continuation)
                else:   # self.option.option_type == "put"
                    payoffs = np.maximum(self.option.K - ST, continuation)
            else:   # self.option_style == "european":
                payoffs = continuation

        return payoffs[0]

    def _crr_model(self):
        """Calculates the price of an option under the Cox-Ross-Rubinstein model"""
        u = np.exp(self.option.sigma * np.sqrt(self.dt))
        d = 1 / u
        p = (np.exp((self.option.r - self.option.q) * self.dt) - d) / (u - d)

        return self._calculate_option_price(u, d, p)

    def _jr_model(self):
        """Calculates the price of an option under the Jarrow-Rudd model"""
        p = 0.5
        u = np.exp((self.option.r - self.option.q - 0.5 * self.option.sigma**2) * self.dt + self.option.sigma * np.sqrt(self.dt))
        d = np.exp((self.option.r - self.option.q - 0.5 * self.option.sigma**2) * self.dt - self.option.sigma * np.sqrt(self.dt))

        return self._calculate_option_price(u, d, p)

    def _peizer_pratt_inversion(self, z):
        """Evaluates the Peizer-Pratt function for a given parameter z"""
        return 0.5 + 0.5 * np.sign(z) * np.sqrt(1 - np.exp(-((z / (self.N + 1/3 + 0.1/(self.N + 1)))**2 * (self.N + 1/6))))

    def _lr_model(self):
        """Calculates the price of an option under the Leisen-Reimer model"""
        bs = BlackScholes(self.option)

        d1 = bs._d1()
        d2 = bs._d2()

        p = self._peizer_pratt_inversion(d2)
        p_prime = self._peizer_pratt_inversion(d1)

        u = np.exp((self.option.r - self.option.q) * self.dt) * p_prime / p
        d = np.exp((self.option.r - self.option.q) * self.dt) * (1 - p_prime) / (1 - p)

        return self._calculate_option_price(u, d, p)

    def price(self, model: str):
        """Calculates the price of an option under a specified model"""
        self.model = model.lower()

        if self.model == "crr":
            return self._crr_model()
        elif self.model == "jr":
            return self._jr_model()
        elif self.model == "lr":
            return self._lr_model()
        else:
            raise ValueError(f"Invalid option style '{self.model}'. Must be {self.VALID_MODELS}")
