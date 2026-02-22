from ..option_class import Option
from ..models.binomial import Binomial

class FiniteDifferenceGreeks:
    def __init__(self, option: Option, option_style: str, N: int = 1_000):
        self.option = option
        self.option_style = option_style
        self.N = N

    def _price(self, price_func, S=None, T=None, sigma=None, r=None):
        option = Option(
            self.option.option_type,
            S if S is not None else self.option.S,
            self.option.K,
            T if T is not None else self.option.T,
            sigma if sigma is not None else self.option.sigma,
            r if r is not None else self.option.r,
            self.option.q
        )

        model = Binomial(option, self.option_style, self.N)
        return price_func(model)

    def fd_delta(self, price_func, h):
        price_up = self._price(price_func, S=self.option.S + h)
        price_down = self._price(price_func, S=self.option.S - h)

        return (price_up - price_down) / (2 * h)

    def fd_gamma(self, price_func, h):
        price_up = self._price(price_func, S=self.option.S + h)
        price_mid = self._price(price_func, S=self.option.S)
        price_down = self._price(price_func, S=self.option.S - h)

        return (price_up - 2 * price_mid + price_down) / (h**2)

    def fd_vega(self, price_func, h):
        price_up = self._price(price_func, sigma=self.option.sigma + h)
        price_down = self._price(price_func, sigma=self.option.sigma - h)

        return (price_up - price_down) / (2 * h)

    def fd_theta(self, price_func, h):
        price_up = self._price(price_func, T=self.option.T + h)
        price_down = self._price(price_func, T=self.option.T - h)

        return (price_up - price_down) / (2 * h)

    def fd_rho(self, price_func, h):
        price_up = self._price(price_func, r=self.option.r + h)
        price_down = self._price(price_func, r=self.option.r - h)

        return (price_up - price_down) / (2 * h)
