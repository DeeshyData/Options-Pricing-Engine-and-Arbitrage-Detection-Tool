import yfinance as yf
import pandas as pd
from datetime import datetime
from .option_class import Option

class StockOptionsData:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        self.data = self.stock.history(period="1d")

        self._validate_ticker()

    def _validate_ticker(self):
        if self.data.empty:
            raise ValueError(f"Invalid ticker: {self.ticker}")

    def _validate_options(self, expirations: list):
        if len(expirations) == 0:
            raise ValueError(f"No options available for {self.ticker}")

    def get_stock_price(self):
        return self.data["Close"].iloc[-1]

    def get_expirations(self):
        expirations = self.stock.options
        self._validate_options(expirations)
        return list(expirations)

    def _validate_expiration_date(self, expiration_date):
        expirations = self.get_expirations()

        if expiration_date is None:
            return expirations[0]
        elif expiration_date not in expirations:
            raise ValueError(f"Expiration date {expiration_date} not available. Available dates: {expirations}")
        else:
            return expiration_date

    def get_call_options(self, expiration_date=None):
        expiration_date = self._validate_expiration_date(expiration_date)
        calls = self.stock.option_chain(expiration_date).calls
        calls = calls.rename(columns={
            'lastPrice': 'price'
        })
        calls['price'] = (calls['bid'] + calls['ask']) / 2
        return calls

    def get_put_options(self, expiration_date=None):
        expiration_date = self._validate_expiration_date(expiration_date)
        puts = self.stock.option_chain(expiration_date).puts
        puts = puts.rename(columns={
            'lastPrice': 'price'
        })
        puts['price'] = (puts['bid'] + puts['ask']) / 2
        return puts

    def get_option_chain(self, expiration_date=None):
        calls = self.get_call_options(expiration_date)
        puts = self.get_put_options(expiration_date)

        option_chain = pd.merge(
            calls[['strike', 'price', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']],
            puts[['strike', 'price', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']],
            on='strike',
            how='outer',
            suffixes=('_call', '_put')
        )

        option_chain = option_chain.rename(columns={
            'price_call': 'callPrice',
            'price_put': 'putPrice',
            'bid_call': 'callBid',
            'bid_put': 'putBid',
            'ask_call': 'callAsk',
            'ask_put': 'putAsk',
            'volume_call': 'callVolume',
            'volume_put': 'putVolume',
            'openInterest_call': 'callOpenInterest',
            'openInterest_put': 'putOpenInterest',
            'impliedVolatility_call': 'callImpliedVolatility',
            'impliedVolatility_put': 'putImpliedVolatility',
        })

        option_chain = option_chain[[
            'callPrice', 'callBid', 'callAsk', 'callVolume', 'callOpenInterest', 'callImpliedVolatility', 'strike',
            'putPrice', 'putBid', 'putAsk', 'putVolume', 'putOpenInterest', 'putImpliedVolatility'
        ]]

        return option_chain

    def get_risk_free_rate(self):
        treasury = yf.Ticker("^TNX")
        data = treasury.history(period="5d")
        return data["Close"].iloc[-1] / 100

    def get_dividend_yield(self):
        info = self.stock.info

        if 'dividendYield' in info and info['dividendYield'] is not None:
            return info['dividendYield'] / 100

        return 0

    def calculate_time_to_maturity(self, expiration_date=None):
        expiration_date = self._validate_expiration_date(expiration_date)
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        days_to_expiry = (exp_date - today).days
        return max(days_to_expiry / 365, 1 / 365)

    def get_complete_options_data(self, expiration_date=None):
        S = self.get_stock_price()
        option_chain = self.get_option_chain(expiration_date)

        if expiration_date is None:
            expirations = self.get_expirations()
            expiration_date = expirations[0]

        T = self.calculate_time_to_maturity(expiration_date)
        r = self.get_risk_free_rate()
        q = self.get_dividend_yield()

        return {
            'Value': {
                'Ticker': self.ticker,
                'Stock Price': S,
                'Expiration Date': expiration_date,
                'Time to Maturity': T,
                'Risk Free Rate': r,
                'Dividend Yield': q
            }
        }, option_chain

    def _get_strike_prices(self, option_chain):
        return list(option_chain.strike)

    def get_relevant_options_data(self, option_type, strike=None, expiration_date=None):
        S = self.get_stock_price()

        if option_type.lower() == "call":
            option_chain = self.get_call_options(expiration_date)
        elif option_type.lower() == "put":
            option_chain = self.get_put_options(expiration_date)
        else:
            raise ValueError(f"Invalid option type {option_type}, must be 'call' or 'put'")

        strikes = self._get_strike_prices(option_chain)

        if strike is None:
            strike = strikes[0]
        elif strike not in strikes:
            raise ValueError(f"Invalid strike {strike}. Valid strikes: {strikes}")

        K = strike
        T = self.calculate_time_to_maturity(expiration_date)
        r = self.get_risk_free_rate()
        sigma = option_chain.loc[option_chain['strike'] == K, 'impliedVolatility'].iloc[0]
        q = self.get_dividend_yield()

        market_price = option_chain.loc[option_chain['strike'] == K, 'price'].iloc[0]
        return Option(option_type, S, K, T, sigma, r, q), market_price
