class Option:
    VALID_OPTION_TYPES = ['call', 'put']

    def __init__(self, option_type: str, S: float, K: float, T: float, sigma: float, r: float, q: float = 0):
        """
        Initialise paramters for an option
 
        :param option_type: Call or put
        :param S: Current stock price
        :param K: Strike price
        :param T: Time to maturity in years
        :param sigma: Volatility
        :param r: Risk-free rate
        :param q: Dividend yield rate
        """
        self.option_type = option_type
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

        self._validate_inputs()

    def _validate_inputs(self):
        """Checks if the provided option parameters are valid"""
        if self.option_type not in self.VALID_OPTION_TYPES:
            raise ValueError(f"Invalid option type '{self.option_type}'. Must be {self.VALID_OPTION_TYPES}")
        if self.S <= 0:
            raise ValueError("Stock price must be greater than 0")
        if self.K <= 0:
            raise ValueError("Strike price must be greater than 0")
        if self.T < 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be greater than 0")

    def data(self):
        return {
            'Value': {
                'Option Type': self.option_type,
                'Stock Price': self.S,
                'Strike Price': self.K,
                'Time to Maturity': self.T,
                'Volatility': self.sigma,
                'Risk-free Rate': self.r,
                'Dividend Yield': self.q
            }
        }
