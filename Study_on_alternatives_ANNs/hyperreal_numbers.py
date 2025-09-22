"""
Implementación de Números Hiperreales
====================================

Los números hiperreales fueron desarrollados por Abraham Robinson en 1960
para el análisis no estándar. Extienden los números reales con:
- Infinitésimos (números más pequeños que cualquier real positivo pero > 0)
- Infinitos (números más grandes que cualquier real)
- Mantienen las propiedades algebraicas de los reales

Representación: Un hiperreal h se puede escribir como:
h = a₀ + a₁ε + a₂ε² + ... + aₙεⁿ + ... + b₁ω + b₂ω² + ...

Donde:
- aᵢ son números reales (coeficientes)
- ε es un infinitésimo (0 < ε < r para cualquier real positivo r)
- ω es un infinito (ω > r para cualquier real r)
"""

import numpy as np
from typing import Union, List, Tuple
import math


class NumeroHiperreal:
    """
    Implementación de números hiperreales usando representación dual:
    h = parte_real + coef_infinitesimo * ε + coef_infinito * ω
    
    Donde:
    - parte_real: componente estándar (número real)
    - coef_infinitesimo: coeficiente del infinitésimo ε
    - coef_infinito: coeficiente del infinito ω
    """
    
    def __init__(self, parte_real: float = 0.0, 
                 coef_infinitesimo: float = 0.0, 
                 coef_infinito: float = 0.0,
                 nombre: str = None):
        """
        Inicializa un número hiperreal.
        
        Args:
            parte_real: Parte estándar (número real)
            coef_infinitesimo: Coeficiente del infinitésimo ε
            coef_infinito: Coeficiente del infinito ω
            nombre: Nombre descriptivo (opcional)
        """
        self.real = float(parte_real)
        self.infinitesimo = float(coef_infinitesimo)
        self.infinito = float(coef_infinito)
        self.nombre = nombre
    
    def es_infinitesimo(self) -> bool:
        """Verifica si el número es infinitesimal."""
        return self.real == 0.0 and self.infinitesimo != 0.0 and self.infinito == 0.0
    
    def es_infinito(self) -> bool:
        """Verifica si el número es infinito."""
        return self.infinito != 0.0
    
    def es_finito(self) -> bool:
        """Verifica si el número es finito."""
        return self.infinito == 0.0
    
    def parte_estandar(self) -> float:
        """Retorna la parte estándar (real) del número hiperreal."""
        return self.real
    
    def __add__(self, otro: 'NumeroHiperreal') -> 'NumeroHiperreal':
        """
        Suma de números hiperreales:
        (a + bε + cω) + (a' + b'ε + c'ω) = (a+a') + (b+b')ε + (c+c')ω
        """
        return NumeroHiperreal(
            self.real + otro.real,
            self.infinitesimo + otro.infinitesimo,
            self.infinito + otro.infinito,
            f"({self} + {otro})"
        )
    
    def __sub__(self, otro: 'NumeroHiperreal') -> 'NumeroHiperreal':
        """Resta de números hiperreales."""
        return NumeroHiperreal(
            self.real - otro.real,
            self.infinitesimo - otro.infinitesimo,
            self.infinito - otro.infinito,
            f"({self} - {otro})"
        )
    
    def __mul__(self, otro: 'NumeroHiperreal') -> 'NumeroHiperreal':
        """
        Multiplicación de números hiperreales:
        (a + bε + cω)(a' + b'ε + c'ω)
        
        Reglas:
        - ε * ε = 0 (infinitésimo de orden superior, despreciable)
        - ω * ε = 1 (cancelación infinito-infinitésimo)
        - ω * ω = ω² (infinito de orden superior)
        """
        # Parte real: a*a'
        nueva_real = self.real * otro.real
        
        # Parte infinitesimal: a*b' + b*a' (términos lineales en ε)
        nuevo_infinitesimo = (self.real * otro.infinitesimo + 
                             self.infinitesimo * otro.real)
        
        # Parte infinita: a*c' + c*a' + b*b'*ω (ε*ε→0, pero consideramos términos mixtos)
        nuevo_infinito = (self.real * otro.infinito + 
                         self.infinito * otro.real +
                         self.infinito * otro.infinito)  # ω * ω
        
        # Término especial: infinitésimo * infinito ≈ constante
        if self.infinitesimo != 0 and otro.infinito != 0:
            nueva_real += self.infinitesimo * otro.infinito
        if self.infinito != 0 and otro.infinitesimo != 0:
            nueva_real += self.infinito * otro.infinitesimo
        
        return NumeroHiperreal(nueva_real, nuevo_infinitesimo, nuevo_infinito,
                              f"({self} * {otro})")
    
    def __truediv__(self, otro: 'NumeroHiperreal') -> 'NumeroHiperreal':
        """
        División de números hiperreales (aproximada).
        """
        if otro.real == 0 and otro.infinitesimo == 0 and otro.infinito == 0:
            raise ValueError("División por cero hiperreal")
        
        # Caso simple: división por número real
        if otro.infinitesimo == 0 and otro.infinito == 0:
            return NumeroHiperreal(
                self.real / otro.real,
                self.infinitesimo / otro.real,
                self.infinito / otro.real,
                f"({self} / {otro})"
            )
        
        # División por infinito
        if otro.es_infinito():
            return NumeroHiperreal(
                0.0,
                self.real / abs(otro.infinito) if otro.infinito != 0 else 0.0,
                0.0,
                f"({self} / {otro})"
            )
        
        # División por infinitésimo (resultado infinito)
        if otro.es_infinitesimo():
            return NumeroHiperreal(
                0.0,
                0.0,
                self.real / otro.infinitesimo if otro.infinitesimo != 0 else 0.0,
                f"({self} / {otro})"
            )
        
        # Caso general (aproximación)
        return NumeroHiperreal(
            self.real / otro.real,
            (self.infinitesimo * otro.real - self.real * otro.infinitesimo) / (otro.real ** 2),
            self.infinito / otro.real,
            f"({self} / {otro})"
        )
    
    def __neg__(self) -> 'NumeroHiperreal':
        """Negación de número hiperreal."""
        return NumeroHiperreal(-self.real, -self.infinitesimo, -self.infinito,
                              f"-{self}")
    
    def __eq__(self, otro: 'NumeroHiperreal') -> bool:
        """
        Igualdad hiperreal: dos números son iguales si todas sus componentes
        son iguales.
        """
        return (abs(self.real - otro.real) < 1e-10 and
                abs(self.infinitesimo - otro.infinitesimo) < 1e-10 and
                abs(self.infinito - otro.infinito) < 1e-10)
    
    def __lt__(self, otro: 'NumeroHiperreal') -> bool:
        """
        Orden hiperreal:
        1. Si infinitos diferentes, el mayor infinito gana
        2. Si infinitos iguales, comparar partes reales
        3. Si partes reales iguales, comparar infinitésimos
        """
        if self.infinito != otro.infinito:
            return self.infinito < otro.infinito
        elif self.real != otro.real:
            return self.real < otro.real
        else:
            return self.infinitesimo < otro.infinitesimo
    
    def __le__(self, otro: 'NumeroHiperreal') -> bool:
        return self < otro or self == otro
    
    def __gt__(self, otro: 'NumeroHiperreal') -> bool:
        return otro < self
    
    def __ge__(self, otro: 'NumeroHiperreal') -> bool:
        return otro <= self
    
    def valor_aproximado(self) -> float:
        """
        Calcula un valor aproximado como float para visualización.
        
        Regla:
        - Si es infinito: ±∞
        - Si es infinitésimo puro: valor muy pequeño
        - Caso general: parte real + aproximación de infinitésimo
        """
        if self.es_infinito():
            return float('inf') if self.infinito > 0 else float('-inf')
        elif self.es_infinitesimo():
            return self.infinitesimo * 1e-10  # Representar como muy pequeño
        else:
            return self.real + self.infinitesimo * 1e-6  # Aproximación
    
    def __str__(self) -> str:
        """Representación como string."""
        if self.nombre:
            return self.nombre
        
        partes = []
        
        if self.real != 0:
            partes.append(f"{self.real}")
        
        if self.infinitesimo != 0:
            if self.infinitesimo == 1:
                partes.append("ε")
            elif self.infinitesimo == -1:
                partes.append("-ε")
            else:
                partes.append(f"{self.infinitesimo}ε")
        
        if self.infinito != 0:
            if self.infinito == 1:
                partes.append("ω")
            elif self.infinito == -1:
                partes.append("-ω")
            else:
                partes.append(f"{self.infinito}ω")
        
        if not partes:
            return "0"
        
        resultado = partes[0]
        for parte in partes[1:]:
            if parte.startswith('-'):
                resultado += " " + parte
            else:
                resultado += " + " + parte
        
        return resultado
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        """Hash basado en las componentes."""
        return hash((round(self.real, 10), round(self.infinitesimo, 10), round(self.infinito, 10)))


# Constantes hiperreales básicas
CERO_H = NumeroHiperreal(0, 0, 0, "0")
UNO_H = NumeroHiperreal(1, 0, 0, "1")
MENOS_UNO_H = NumeroHiperreal(-1, 0, 0, "-1")
EPSILON = NumeroHiperreal(0, 1, 0, "ε")  # Infinitésimo básico
OMEGA = NumeroHiperreal(0, 0, 1, "ω")    # Infinito básico
MEDIO_H = NumeroHiperreal(0.5, 0, 0, "1/2")
DOS_H = NumeroHiperreal(2, 0, 0, "2")


class GeneradorHiperreales:
    """Generador de números hiperreales para entrenamiento."""
    
    @staticmethod
    def numeros_basicos() -> List[NumeroHiperreal]:
        """Retorna números hiperreales básicos."""
        return [
            CERO_H, UNO_H, MENOS_UNO_H, MEDIO_H, DOS_H,
            EPSILON, NumeroHiperreal(0, -1, 0, "-ε"),  # -ε
            NumeroHiperreal(0, 0.5, 0, "ε/2"),        # ε/2
            NumeroHiperreal(1, 1, 0, "1+ε"),          # 1+ε
            NumeroHiperreal(0, 0, 0.5, "ω/2"),        # ω/2
        ]
    
    @staticmethod
    def desde_real(valor: float) -> NumeroHiperreal:
        """Convierte un número real a hiperreal."""
        return NumeroHiperreal(valor, 0, 0, f"~{valor}")
    
    @staticmethod
    def aleatorio_simple() -> NumeroHiperreal:
        """Genera un número hiperreal aleatorio simple."""
        import random
        basicos = GeneradorHiperreales.numeros_basicos()
        return random.choice(basicos)
    
    @staticmethod
    def infinitesimo_aleatorio() -> NumeroHiperreal:
        """Genera un infinitésimo aleatorio."""
        import random
        coef = random.uniform(-1, 1)
        return NumeroHiperreal(0, coef, 0, f"{coef}ε")
    
    @staticmethod
    def infinito_aleatorio() -> NumeroHiperreal:
        """Genera un infinito aleatorio."""
        import random
        coef = random.uniform(0.1, 2.0) * random.choice([-1, 1])
        return NumeroHiperreal(0, 0, coef, f"{coef}ω")


def exp_hiperreal(x: NumeroHiperreal) -> NumeroHiperreal:
    """
    Función exponencial para números hiperreales.
    exp(a + bε + cω) ≈ exp(a) * (1 + bε) * exp(cω)
    """
    if x.es_infinito() and x.infinito > 0:
        return OMEGA  # exp(∞) = ∞
    elif x.es_infinito() and x.infinito < 0:
        return EPSILON  # exp(-∞) = ε
    
    exp_real = math.exp(x.real) if abs(x.real) < 700 else (math.exp(700) if x.real > 0 else math.exp(-700))
    
    # Expansión de Taylor: exp(a + bε) ≈ exp(a)(1 + bε)
    nueva_real = exp_real
    nuevo_infinitesimo = exp_real * x.infinitesimo
    nuevo_infinito = x.infinito  # exp(ω) ≈ ω para infinitos
    
    return NumeroHiperreal(nueva_real, nuevo_infinitesimo, nuevo_infinito, f"exp({x})")


def test_hiperreales():
    """Prueba básica de números hiperreales."""
    print("🧮 PRUEBA DE NÚMEROS HIPERREALES")
    print("=" * 50)
    
    # Números básicos
    print("📊 Números básicos:")
    print(f"Cero: {CERO_H}")
    print(f"Uno: {UNO_H}")
    print(f"Infinitésimo: {EPSILON}")
    print(f"Infinito: {OMEGA}")
    print(f"1 + ε: {UNO_H + EPSILON}")
    
    # Operaciones
    print(f"\n🔢 Operaciones:")
    print(f"ε + ε = {EPSILON + EPSILON}")
    print(f"1 + ε = {UNO_H + EPSILON}")
    print(f"ε * ω = {EPSILON * OMEGA}")
    print(f"2ε = {EPSILON + EPSILON}")
    print(f"ω - ω = {OMEGA - OMEGA}")
    
    # Comparaciones
    print(f"\n📈 Comparaciones:")
    print(f"0 < ε: {CERO_H < EPSILON}")
    print(f"ε < 1: {EPSILON < UNO_H}")
    print(f"1 < ω: {UNO_H < OMEGA}")
    print(f"ω > 1000: {OMEGA > GeneradorHiperreales.desde_real(1000)}")
    
    # Valores aproximados
    print(f"\n💻 Valores aproximados:")
    numeros = [CERO_H, UNO_H, EPSILON, OMEGA, UNO_H + EPSILON]
    for num in numeros:
        print(f"{num} ≈ {num.valor_aproximado()}")
    
    # Función exponencial
    print(f"\n📐 Función exponencial:")
    print(f"exp(0) = {exp_hiperreal(CERO_H)}")
    print(f"exp(ε) = {exp_hiperreal(EPSILON)}")
    print(f"exp(1) = {exp_hiperreal(UNO_H)}")


if __name__ == "__main__":
    test_hiperreales()
