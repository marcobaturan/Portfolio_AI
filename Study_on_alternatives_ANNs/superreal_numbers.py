"""
Superreal Numbers Implementation
===============================

Superreal numbers extend real numbers with infinitesimals and infinities
in a different way than hyperreals. They form a field that includes
all real numbers plus infinitesimal and infinite elements.

Representation: a + b*δ + c*Ω
where δ is infinitesimal and Ω is infinite.
"""

import numpy as np
from typing import Union
import math


class SuperReal:
    """Implementation of superreal numbers."""
    
    def __init__(self, real: float = 0.0, infinitesimal: float = 0.0, infinite: float = 0.0, name: str = None):
        """
        Initialize a superreal number.
        
        Args:
            real: Real part
            infinitesimal: Coefficient of infinitesimal δ
            infinite: Coefficient of infinite Ω
            name: Optional name for the number
        """
        self.real = float(real)
        self.infinitesimal = float(infinitesimal)
        self.infinite = float(infinite)
        self.name = name
    
    def __add__(self, other: 'SuperReal') -> 'SuperReal':
        """Addition of superreal numbers."""
        return SuperReal(
            self.real + other.real,
            self.infinitesimal + other.infinitesimal,
            self.infinite + other.infinite
        )
    
    def __sub__(self, other: 'SuperReal') -> 'SuperReal':
        """Subtraction of superreal numbers."""
        return SuperReal(
            self.real - other.real,
            self.infinitesimal - other.infinitesimal,
            self.infinite - other.infinite
        )
    
    def __mul__(self, other: 'SuperReal') -> 'SuperReal':
        """Multiplication of superreal numbers."""
        # (a + bδ + cΩ)(a' + b'δ + c'Ω)
        # = aa' + (ab' + ba')δ + (ac' + ca')Ω + bb'δ² + (bc' + cb')δΩ + cc'Ω²
        # Simplifying: δ² ≈ 0, δΩ ≈ 1, Ω² ≈ Ω
        
        new_real = (self.real * other.real + 
                   self.infinitesimal * other.infinite + 
                   self.infinite * other.infinitesimal)
        
        new_infinitesimal = (self.real * other.infinitesimal + 
                           self.infinitesimal * other.real)
        
        new_infinite = (self.real * other.infinite + 
                       self.infinite * other.real + 
                       self.infinite * other.infinite)
        
        return SuperReal(new_real, new_infinitesimal, new_infinite)
    
    def __truediv__(self, other: 'SuperReal') -> 'SuperReal':
        """Division of superreal numbers (simplified)."""
        if other.real == 0 and other.infinitesimal == 0 and other.infinite == 0:
            raise ValueError("Division by zero superreal")
        
        # Simple case: division by real number
        if other.infinitesimal == 0 and other.infinite == 0:
            return SuperReal(
                self.real / other.real,
                self.infinitesimal / other.real,
                self.infinite / other.real
            )
        
        # Division by infinite
        if other.infinite != 0:
            return SuperReal(
                self.infinitesimal / other.infinite,
                self.real / other.infinite,
                0.0
            )
        
        # General case (approximation)
        return SuperReal(
            self.real / other.real if other.real != 0 else 0,
            self.infinitesimal / other.real if other.real != 0 else 0,
            self.infinite
        )
    
    def __neg__(self) -> 'SuperReal':
        """Negation."""
        return SuperReal(-self.real, -self.infinitesimal, -self.infinite)
    
    def __eq__(self, other: 'SuperReal') -> bool:
        """Equality check."""
        return (abs(self.real - other.real) < 1e-10 and
                abs(self.infinitesimal - other.infinitesimal) < 1e-10 and
                abs(self.infinite - other.infinite) < 1e-10)
    
    def __lt__(self, other: 'SuperReal') -> bool:
        """Less than comparison."""
        if self.infinite != other.infinite:
            return self.infinite < other.infinite
        elif self.real != other.real:
            return self.real < other.real
        else:
            return self.infinitesimal < other.infinitesimal
    
    def approximate_value(self) -> float:
        """Get approximate floating point value."""
        if self.infinite != 0:
            return float('inf') if self.infinite > 0 else float('-inf')
        elif abs(self.real) > 1e-10:
            return self.real + self.infinitesimal * 1e-8
        else:
            return self.infinitesimal * 1e-8
    
    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return self.name
        
        parts = []
        if self.real != 0:
            parts.append(f"{self.real}")
        if self.infinitesimal != 0:
            if self.infinitesimal == 1:
                parts.append("δ")
            elif self.infinitesimal == -1:
                parts.append("-δ")
            else:
                parts.append(f"{self.infinitesimal}δ")
        if self.infinite != 0:
            if self.infinite == 1:
                parts.append("Ω")
            elif self.infinite == -1:
                parts.append("-Ω")
            else:
                parts.append(f"{self.infinite}Ω")
        
        if not parts:
            return "0"
        
        result = parts[0]
        for part in parts[1:]:
            if part.startswith('-'):
                result += " " + part
            else:
                result += " + " + part
        
        return result
    
    def __repr__(self) -> str:
        return self.__str__()


# Basic superreal constants
ZERO_S = SuperReal(0, 0, 0, "0")
ONE_S = SuperReal(1, 0, 0, "1")
MINUS_ONE_S = SuperReal(-1, 0, 0, "-1")
DELTA = SuperReal(0, 1, 0, "δ")  # Infinitesimal
OMEGA_S = SuperReal(0, 0, 1, "Ω")  # Infinite


class SuperRealGenerator:
    """Generator for superreal numbers."""
    
    @staticmethod
    def from_real(value: float) -> SuperReal:
        """Convert real number to superreal."""
        return SuperReal(value, 0, 0)
    
    @staticmethod
    def basic_numbers() -> list:
        """Get basic superreal numbers."""
        return [ZERO_S, ONE_S, MINUS_ONE_S, DELTA, OMEGA_S,
                SuperReal(0.5, 0, 0), SuperReal(0, 0.5, 0),
                SuperReal(1, 1, 0), SuperReal(0, 0, 0.5)]
    
    @staticmethod
    def random_superreal() -> SuperReal:
        """Generate random superreal number."""
        import random
        return random.choice(SuperRealGenerator.basic_numbers())


def superreal_exp(x: SuperReal) -> SuperReal:
    """Exponential function for superreal numbers."""
    if x.infinite > 0:
        return OMEGA_S
    elif x.infinite < 0:
        return DELTA
    
    exp_real = math.exp(x.real) if abs(x.real) < 100 else (math.exp(100) if x.real > 0 else math.exp(-100))
    
    # Taylor expansion: exp(a + bδ) ≈ exp(a)(1 + bδ)
    return SuperReal(exp_real, exp_real * x.infinitesimal, x.infinite)


def test_superreals():
    """Test superreal number operations."""
    print("🔢 SUPERREAL NUMBERS TEST")
    print("=" * 40)
    
    # Basic operations
    a = SuperReal(2, 1, 0)  # 2 + δ
    b = SuperReal(1, 0, 1)  # 1 + Ω
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a / ONE_S = {a / ONE_S}")
    
    # Comparisons
    print(f"\nComparisons:")
    print(f"DELTA < ONE_S: {DELTA < ONE_S}")
    print(f"ONE_S < OMEGA_S: {ONE_S < OMEGA_S}")
    
    # Approximate values
    print(f"\nApproximate values:")
    for num in [ZERO_S, ONE_S, DELTA, OMEGA_S, a, b]:
        print(f"{num} ≈ {num.approximate_value()}")


if __name__ == "__main__":
    test_superreals()
