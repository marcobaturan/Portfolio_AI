"""
Hypercomplex Numbers Implementation
==================================

Hypercomplex numbers extend complex numbers to higher dimensions.
We implement quaternions (4D hypercomplex numbers) as our hypercomplex system.

Quaternion: q = a + bi + cj + dk
where i² = j² = k² = ijk = -1
"""

import numpy as np
import math
from typing import Union


class Quaternion:
    """Implementation of quaternions (4D hypercomplex numbers)."""
    
    def __init__(self, w: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, name: str = None):
        """
        Initialize a quaternion.
        
        Args:
            w: Real part (scalar)
            x: i component
            y: j component  
            z: k component
            name: Optional name
        """
        self.w = float(w)  # Real part
        self.x = float(x)  # i component
        self.y = float(y)  # j component
        self.z = float(z)  # k component
        self.name = name
    
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Addition of quaternions."""
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Subtraction of quaternions."""
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Multiplication of quaternions (Hamilton product).
        (a + bi + cj + dk)(e + fi + gj + hk)
        """
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        
        return Quaternion(w, x, y, z)
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm_squared(self) -> float:
        """Squared norm of quaternion."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def norm(self) -> float:
        """Norm (magnitude) of quaternion."""
        return math.sqrt(self.norm_squared())
    
    def __truediv__(self, other: 'Quaternion') -> 'Quaternion':
        """Division of quaternions."""
        if other.norm_squared() == 0:
            raise ValueError("Division by zero quaternion")
        
        conj = other.conjugate()
        norm_sq = other.norm_squared()
        
        numerator = self * conj
        return Quaternion(
            numerator.w / norm_sq,
            numerator.x / norm_sq,
            numerator.y / norm_sq,
            numerator.z / norm_sq
        )
    
    def __neg__(self) -> 'Quaternion':
        """Negation."""
        return Quaternion(-self.w, -self.x, -self.y, -self.z)
    
    def __eq__(self, other: 'Quaternion') -> bool:
        """Equality check."""
        return (abs(self.w - other.w) < 1e-10 and
                abs(self.x - other.x) < 1e-10 and
                abs(self.y - other.y) < 1e-10 and
                abs(self.z - other.z) < 1e-10)
    
    def __lt__(self, other: 'Quaternion') -> bool:
        """Less than comparison (by norm)."""
        return self.norm() < other.norm()
    
    def scalar_multiply(self, scalar: float) -> 'Quaternion':
        """Multiply by scalar."""
        return Quaternion(self.w * scalar, self.x * scalar, self.y * scalar, self.z * scalar)
    
    def normalize(self) -> 'Quaternion':
        """Normalize quaternion to unit quaternion."""
        n = self.norm()
        if n == 0:
            return Quaternion(0, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def to_complex_pair(self) -> tuple:
        """Convert to pair of complex numbers for visualization."""
        z1 = complex(self.w, self.x)
        z2 = complex(self.y, self.z)
        return z1, z2
    
    def approximate_value(self) -> float:
        """Get approximate real value (using norm)."""
        return self.norm()
    
    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return self.name
        
        parts = []
        if self.w != 0:
            parts.append(f"{self.w}")
        if self.x != 0:
            if self.x == 1:
                parts.append("i")
            elif self.x == -1:
                parts.append("-i")
            else:
                parts.append(f"{self.x}i")
        if self.y != 0:
            if self.y == 1:
                parts.append("j")
            elif self.y == -1:
                parts.append("-j")
            else:
                parts.append(f"{self.y}j")
        if self.z != 0:
            if self.z == 1:
                parts.append("k")
            elif self.z == -1:
                parts.append("-k")
            else:
                parts.append(f"{self.z}k")
        
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


# Basic quaternion constants
ZERO_Q = Quaternion(0, 0, 0, 0, "0")
ONE_Q = Quaternion(1, 0, 0, 0, "1")
I_Q = Quaternion(0, 1, 0, 0, "i")
J_Q = Quaternion(0, 0, 1, 0, "j")
K_Q = Quaternion(0, 0, 0, 1, "k")


class QuaternionGenerator:
    """Generator for quaternions."""
    
    @staticmethod
    def from_real(value: float) -> Quaternion:
        """Convert real number to quaternion."""
        return Quaternion(value, 0, 0, 0)
    
    @staticmethod
    def from_complex(z: complex) -> Quaternion:
        """Convert complex number to quaternion."""
        return Quaternion(z.real, z.imag, 0, 0)
    
    @staticmethod
    def basic_quaternions() -> list:
        """Get basic quaternions."""
        return [ZERO_Q, ONE_Q, I_Q, J_Q, K_Q,
                Quaternion(0.5, 0, 0, 0), Quaternion(0, 0.5, 0, 0),
                Quaternion(1, 1, 0, 0), Quaternion(0, 0, 1, 1)]
    
    @staticmethod
    def random_quaternion() -> Quaternion:
        """Generate random quaternion."""
        import random
        return random.choice(QuaternionGenerator.basic_quaternions())
    
    @staticmethod
    def random_unit_quaternion() -> Quaternion:
        """Generate random unit quaternion."""
        # Generate random quaternion and normalize
        w, x, y, z = [random.uniform(-1, 1) for _ in range(4)]
        q = Quaternion(w, x, y, z)
        return q.normalize()


def quaternion_exp(q: Quaternion) -> Quaternion:
    """Exponential function for quaternions."""
    # exp(q) = exp(w) * (cos(|v|) + (v/|v|) * sin(|v|))
    # where q = w + v, v = xi + yj + zk
    
    w = q.w
    v_norm = math.sqrt(q.x**2 + q.y**2 + q.z**2)
    
    if v_norm == 0:
        return Quaternion(math.exp(w), 0, 0, 0)
    
    exp_w = math.exp(w)
    cos_v = math.cos(v_norm)
    sin_v = math.sin(v_norm)
    
    factor = exp_w * sin_v / v_norm
    
    return Quaternion(
        exp_w * cos_v,
        factor * q.x,
        factor * q.y,
        factor * q.z
    )


def test_quaternions():
    """Test quaternion operations."""
    print("🔢 QUATERNION (HYPERCOMPLEX) NUMBERS TEST")
    print("=" * 50)
    
    # Basic quaternions
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 1, 0, 1)
    
    print(f"q1 = {q1}")
    print(f"q2 = {q2}")
    print(f"q1 + q2 = {q1 + q2}")
    print(f"q1 * q2 = {q1 * q2}")
    print(f"q1.conjugate() = {q1.conjugate()}")
    print(f"q1.norm() = {q1.norm():.4f}")
    
    # Basic units
    print(f"\nBasic units:")
    print(f"i * j = {I_Q * J_Q}")
    print(f"j * k = {J_Q * K_Q}")
    print(f"k * i = {K_Q * I_Q}")
    print(f"i * i = {I_Q * I_Q}")
    
    # Exponential
    q3 = Quaternion(0, math.pi/2, 0, 0)
    print(f"\nexp({q3}) = {quaternion_exp(q3)}")


if __name__ == "__main__":
    import random
    test_quaternions()
