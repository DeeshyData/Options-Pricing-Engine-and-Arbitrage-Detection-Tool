import numpy as np
from ..option_class import Option
from ..models.black_scholes import BlackScholes

class ArbitrageDetector:
    def __init__(self, transaction_cost=0.0, min_threshold=0.01):
        self.min_threshold = min_threshold
        self.transaction_cost = transaction_cost

    def put_call_parity(self, C, P, S, K, T, r):
        lhs = C + K * np.exp(-r * T)
        rhs = P + S

        diff = lhs - rhs
        transaction_cost = self.transaction_cost * (C + P + S + K * np.exp(-r * T))
        profit = abs(diff) - transaction_cost

        arbitrage = {
            'call price': float(C),
            'put price': float(P),
            'arbitrage exists': bool(profit >= self.min_threshold),
            'profit': float(profit),
        }

        if profit >= self.min_threshold and diff > 0:
            arbitrage['strategy'] = 'Conversion (call is overpriced)'
            arbitrage['details'] = {
                'action': [
                    'Buy 1 put option',
                    'Buy 1 share of stock',
                    'Sell 1 call option',
                    f'Lend ${K * np.exp(-r * T):.2f} at risk-free rate {r * 100:.2f}%'
                ],
                'Initial Inflow': float(diff)
            }
        elif profit >= self.min_threshold and diff < 0:
            arbitrage['strategy'] = 'Reverse conversion (put is overpriced)'
            arbitrage['details'] = {
                'action': [
                    'Buy 1 call option',
                    f'Borrow ${K * np.exp(-r * T):.2f} at risk-free rate {r * 100:.2f}%',
                    'Sell 1 put option',
                    'Sell 1 share of stock',
                ],
                'Initial Outflow': -float(diff)
            }

        return arbitrage

    def box_spread(self, C1, C2, P1, P2, K1, K2, T, r):
        if K1 >= K2:
            raise ValueError("K1 must be strictly less than K2")
        if T <= 0:
            raise ValueError("Time to maturity must be positive")

        initial_cost = (C1 - C2) + (P2 - P1)
        payoff_maturity = K2 - K1

        diff = np.exp(-r * T) * payoff_maturity - initial_cost
        transaction_cost = self.transaction_cost * (C1 + C2 + P1 + P2)
        profit = abs(diff) - transaction_cost

        arbitrage = {
            'arbitrage exists': bool(profit >= self.min_threshold),
            'profit (pv)': float(profit)
        }

        if profit >= self.min_threshold and diff > 0:
            arbitrage['strategy'] = 'Buy the box'
            arbitrage['details'] = {
                'action at time 0': [
                    f'Buy call at strike K1 = {K1}',
                    f'Sell call at strike K2 = {K2}',
                    f'Sell put at strike K1 = {K1}',
                    f'Buy put at strike K2 = {K2}',
                ],
                'Initial Outflow': float(initial_cost),
                'Payoff at maturity': float(payoff_maturity)
            }
        elif profit >= self.min_threshold and diff < 0:
            arbitrage['strategy'] = 'Sell the box'
            arbitrage['details'] = {
                'action at time 0': [
                    f'Sell call at strike K1 = {K1}',
                    f'Buy call at strike K2 = {K2}',
                    f'Buy put at strike K1 = {K1}',
                    f'Sell put at strike K2 = {K2}',
                ],
                'Initial Inflow': float(initial_cost),
                'Payoff at maturity': float(-payoff_maturity)
            }

        return arbitrage

    def market_price_vs_bs_price(self, market_price, option: Option):
        bs_option = BlackScholes(option)
        bs_price = bs_option.price()
        delta = bs_option.delta()

        diff = market_price - bs_price
        transaction_cost = self.transaction_cost * (market_price + abs(delta) * option.S)
        profit = abs(diff) - transaction_cost

        initial_cashflow = delta * option.S - market_price

        arbitrage = {
            'market price': float(market_price),
            'black-scholes price': float(bs_price),
            'delta': float(delta),
            'arbitrage exists': bool(profit >= self.min_threshold),
            'profit': float(profit),
        }

        if profit >= self.min_threshold and diff > 0:
            arbitrage['details'] = {
                'action': [
                    f'Sell 1 {option.option_type} option',
                    f'Buy {delta:.4f} shares of stock'
                ],
                'Initial Cashflow': float(-initial_cashflow)
            }
        elif profit >= self.min_threshold and diff < 0:
            arbitrage['details'] = {
                'action': [
                    f'Buy 1 {option.option_type} option',
                    f'Sell {delta:.4f} shares of stock'
                ],
                'Initial Cashflow': float(initial_cashflow)
            }

        return arbitrage

    def check_option_bounds(self, option_price, option_style, option_type: str, S, K, T, r, q=0):
        option_type = option_type.lower()

        transaction_cost = self.transaction_cost * option_price

        if option_type == "call":
            lower_bound = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            upper_bound = S * np.exp(-q * T)
        else:   # option_type = "put"
            if option_style == "european":
                lower_bound = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
                upper_bound = K * np.exp(-r * T)
            elif option_style == "american":
                lower_bound = max(0, K - S * np.exp(-q * T))
                upper_bound = K

        arbitrage = {
            'market price': float(option_price),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'arbitrage exists': False,
            'profit': 0,
        }

        if option_price < lower_bound - transaction_cost - self.min_threshold:
            profit = lower_bound - option_price - transaction_cost
            arbitrage['exists'] = True
            arbitrage['profit'] = float(profit)
            arbitrage['violation'] = 'option price is below lower bound'
            arbitrage['details'] = {
                'action': f'Buy {option_type} option'
            }
        elif option_price > upper_bound + transaction_cost + self.min_threshold:
            profit = option_price - upper_bound - transaction_cost
            arbitrage['exists'] = True
            arbitrage['profit'] = profit
            arbitrage['violation'] = 'option price is above upper bound'
            arbitrage['details'] = {
                'action': f'Sell {option_type} option'
            }

        return arbitrage