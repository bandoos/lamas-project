class ExpressionImplemError(Exception):
    pass


class MissingValuation(Exception):
    def __init__(self, symbol, val):
        super().__init__(f"Symbol `{symbol}` is not defined in valuation")
        self.symbol = symbol
        self.val = val


class UnsupportedExpressionType(Exception):
    def __init__(self, msg, expr):
        super().__init__(msg)
        self.expr = expr
