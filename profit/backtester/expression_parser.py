"""
A custom parser and evaluator for strategy feature expressions.

This module provides functionality to parse string expressions, such as
"sma(close, fast_ma) > sma(close, slow_ma)", into an Abstract Syntax Tree (AST)
and then evaluate them.
"""
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from profit.indicators import factory as indicator_factory


class Expression(ABC):
    """Base class for all expression nodes in the AST."""
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        """Evaluates the expression."""
        pass

class Literal(Expression):
    """Represents a literal value (e.g., a number, boolean)."""
    def __init__(self, value: Union[int, float, bool]):
        self.value = value

    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        return self.value

class Variable(Expression):
    """Represents a variable (e.g., 'close', 'fast_ma')."""
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        # Prioritize data columns, then parameters
        if self.name in data.columns:
            return data[self.name]
        if self.name in parameters:
            return parameters[self.name]

        # Return False for undefined conditions (e.g., c4)
        if re.fullmatch(r'c\d+', self.name):
            return False

        raise ValueError(f"Undefined variable or column: {self.name}")

class FunctionCall(Expression):
    """Represents a function call (e.g., 'sma(close, 20)')."""
    def __init__(self, func_name: str, args: List[Expression]):
        self.func_name = func_name
        self.args = args

    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        evaluated_args = [arg.evaluate(data, parameters) for arg in self.args]
        
        # Map function names to indicator_factory methods or other built-ins
        if hasattr(indicator_factory, self.func_name):
            func = getattr(indicator_factory, self.func_name)
            return func(*evaluated_args)
        else:
            raise ValueError(f"Unknown function: {self.func_name}")

class BinaryOperation(Expression):
    """Represents a binary operation (e.g., 'a > b', 'x + y')."""
    def __init__(self, operator: str, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        left_val = self.left.evaluate(data, parameters)
        right_val = self.right.evaluate(data, parameters)

        if self.operator == '>':
            return left_val > right_val
        elif self.operator == '<':
            return left_val < right_val
        elif self.operator == '>=':
            return left_val >= right_val
        elif self.operator == '<=':
            return left_val <= right_val
        elif self.operator == '==':
            return left_val == right_val
        elif self.operator == '!=':
            return left_val != right_val
        elif self.operator == '+':
            return left_val + right_val
        elif self.operator == '-':
            return left_val - right_val
        elif self.operator == '*':
            return left_val * right_val
        elif self.operator == '/':
            return left_val / right_val
        elif self.operator == 'and':
            return left_val & right_val # Use bitwise for pandas Series
        elif self.operator == 'or':
            return left_val | right_val # Use bitwise for pandas Series
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

class UnaryOperation(Expression):
    """Represents a unary operation (e.g., 'not x')."""
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand

    def evaluate(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Any:
        operand_val = self.operand.evaluate(data, parameters)
        if self.operator == 'not':
            return ~operand_val # Use bitwise for pandas Series
        else:
            raise ValueError(f"Unknown unary operator: {self.operator}")


class Token:
    def __init__(self, type: str, value: Any):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Tokenizer:
    TOKEN_SPECS = [
        ('SKIP',        r'[ \t\n]+'),        # Whitespace
        ('AND',         r'and'),             # Logical AND
        ('OR',          r'or'),              # Logical OR
        ('NOT',         r'not'),             # Logical NOT
        ('BOOLEAN',     r'True|False'),      # Boolean literals
        ('NUMBER',      r'\d+(\.\d*)?'),     # Integer or float
        ('IDENTIFIER',  r'[a-zA-Z_][a-zA-Z0-9_]*'), # Function names, variables
        ('EQ',          r'=='),
        ('NE',          r'!='),
        ('LE',          r'<='),
        ('GE',          r'>='),
        ('LT',          r'<'),
        ('GT',          r'>'),
        ('PLUS',        r'\+'),
        ('MINUS',       r'-'),
        ('ASTERISK',    r'\*'),
        ('SLASH',       r'/'),
        ('LPAREN',      r'\('),
        ('RPAREN',      r'\)'),
        ('COMMA',       r','),
    ]

    def __init__(self, expression_str: str):
        self.expression_str = expression_str
        self.tokens: list[Token] = []
        self._tokenize()
        self.current_token_index = 0

    def _tokenize(self):
        token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.TOKEN_SPECS)
        for match in re.finditer(token_regex, self.expression_str):
            token_type = match.lastgroup
            token_value = match.group(token_type)
            if token_type != 'SKIP':
                if token_type == 'NUMBER':
                    if '.' in token_value:
                        self.tokens.append(Token(token_type, float(token_value)))
                    else:
                        self.tokens.append(Token(token_type, int(token_value)))
                elif token_type == 'BOOLEAN':
                    self.tokens.append(Token(token_type, token_value == 'True'))
                else: # Covers IDENTIFIER, AND, OR, NOT, EQ, etc.
                    self.tokens.append(Token(token_type, token_value))
        self.tokens.append(Token('EOF', None)) # End of File token

    def peek(self) -> Token:
        return self.tokens[self.current_token_index]

    def consume(self, expected_type: Optional[str] = None) -> Token:
        token = self.tokens[self.current_token_index]
        if expected_type and token.type != expected_type:
            raise SyntaxError(f"Expected token type {expected_type} but got {token.type} ({token.value})")
        self.current_token_index += 1
        return token

    def has_more_tokens(self) -> bool:
        return self.peek().type != 'EOF'


class Parser:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def parse_expression(self) -> Expression:
        return self.parse_or()

    def parse_or(self) -> Expression:
        left = self.parse_and()
        while self.tokenizer.peek().type == 'OR':
            operator = self.tokenizer.consume().value
            right = self.parse_and()
            left = BinaryOperation(operator, left, right)
        return left

    def parse_and(self) -> Expression:
        left = self.parse_not()
        while self.tokenizer.peek().type == 'AND':
            operator = self.tokenizer.consume().value
            right = self.parse_not()
            left = BinaryOperation(operator, left, right)
        return left

    def parse_not(self) -> Expression:
        if self.tokenizer.peek().type == 'NOT':
            operator = self.tokenizer.consume().value
            operand = self.parse_comparison()
            return UnaryOperation(operator, operand)
        return self.parse_comparison()

    def parse_comparison(self) -> Expression:
        left = self.parse_additive()
        while self.tokenizer.peek().type in ['EQ', 'NE', 'LT', 'GT', 'LE', 'GE']:
            operator = self.tokenizer.consume().value
            right = self.parse_additive()
            left = BinaryOperation(operator, left, right)
        return left

    def parse_additive(self) -> Expression:
        left = self.parse_multiplicative()
        while self.tokenizer.peek().type in ['PLUS', 'MINUS']:
            operator = self.tokenizer.consume().value
            right = self.parse_multiplicative()
            left = BinaryOperation(operator, left, right)
        return left

    def parse_multiplicative(self) -> Expression:
        left = self.parse_unary()
        while self.tokenizer.peek().type in ['ASTERISK', 'SLASH']:
            operator = self.tokenizer.consume().value
            right = self.parse_unary()
            left = BinaryOperation(operator, left, right)
        return left

    def parse_unary(self) -> Expression:
        # Handle cases like `macd_line[-1]` which is not directly supported by current AST
        # For now, we'll assume it's handled as part of the variable name or not present
        return self.parse_primary()

    def parse_primary(self) -> Expression:
        token = self.tokenizer.peek()

        if token.type == 'NUMBER':
            return Literal(self.tokenizer.consume('NUMBER').value)
        elif token.type == 'BOOLEAN':
            return Literal(self.tokenizer.consume('BOOLEAN').value)
        elif token.type == 'IDENTIFIER':
            identifier_name = self.tokenizer.consume('IDENTIFIER').value
            if self.tokenizer.peek().type == 'LPAREN':
                self.tokenizer.consume('LPAREN')
                args = []
                if self.tokenizer.peek().type != 'RPAREN':
                    args.append(self.parse_expression())
                    while self.tokenizer.peek().type == 'COMMA':
                        self.tokenizer.consume('COMMA')
                        args.append(self.parse_expression())
                self.tokenizer.consume('RPAREN')
                return FunctionCall(identifier_name, args)
            return Variable(identifier_name)
        elif token.type == 'LPAREN':
            self.tokenizer.consume('LPAREN')
            expr = self.parse_expression()
            self.tokenizer.consume('RPAREN')
            return expr
        else:
            raise SyntaxError(f"Unexpected token: {token}")


def parse_expression(expression_str: str) -> Expression:
    """
    Parses a string expression into an Expression AST.

    Args:
        expression_str (str): The string expression to parse.

    Returns:
        Expression: The root node of the AST.
    """
    tokenizer = Tokenizer(expression_str)
    parser = Parser(tokenizer)
    return parser.parse_expression()
