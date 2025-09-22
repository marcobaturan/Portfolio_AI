"""
Implementación de Números Surreales
===================================

Los números surreales fueron creados por John Conway en 1970.
Cada número surreal se representa como {L | R} donde:
- L es un conjunto de números surreales menores
- R es un conjunto de números surreales mayores
- Ningún elemento de L es >= a ningún elemento de R

Ejemplos:
- 0 = {∅ | ∅}
- 1 = {0 | ∅}  
- -1 = {∅ | 0}
- 1/2 = {0 | 1}
- 2 = {1 | ∅}
- ω (infinito) = {0,1,2,3,... | ∅}
- ε (infinitésimo) = {0 | 1,1/2,1/4,1/8,...}
"""

from typing import Set, Union, List, Optional
import numpy as np
from fractions import Fraction


class NumeroSurreal:
    """Implementación básica de números surreales."""
    
    def __init__(self, izquierda: Set['NumeroSurreal'] = None, 
                 derecha: Set['NumeroSurreal'] = None, 
                 nombre: str = None):
        """
        Inicializa un número surreal.
        
        Args:
            izquierda: Conjunto de números surreales menores (L)
            derecha: Conjunto de números surreales mayores (R)  
            nombre: Nombre descriptivo del número (opcional)
        """
        self.L = izquierda if izquierda is not None else set()
        self.R = derecha if derecha is not None else set()
        self.nombre = nombre
        
        # Verificar validez (ningún elemento de L >= elemento de R)
        if not self._es_valido():
            raise ValueError(f"Número surreal inválido: {self}")
    
    def _es_valido(self) -> bool:
        """Verifica si el número surreal es válido."""
        for l in self.L:
            for r in self.R:
                if l >= r:
                    return False
        return True
    
    def __eq__(self, otro: 'NumeroSurreal') -> bool:
        """Igualdad: x = y si x <= y y y <= x"""
        return self <= otro and otro <= self
    
    def __le__(self, otro: 'NumeroSurreal') -> bool:
        """
        Orden: x <= y si:
        - No existe l en L_x tal que y <= l, Y
        - No existe r en R_y tal que r <= x
        """
        # No debe haber l en L_x tal que otro <= l
        for l in self.L:
            if otro <= l:
                return False
        
        # No debe haber r en R_otro tal que r <= self
        for r in otro.R:
            if r <= self:
                return False
        
        return True
    
    def __lt__(self, otro: 'NumeroSurreal') -> bool:
        """Menor estricto."""
        return self <= otro and not (otro <= self)
    
    def __ge__(self, otro: 'NumeroSurreal') -> bool:
        """Mayor o igual."""
        return otro <= self
    
    def __gt__(self, otro: 'NumeroSurreal') -> bool:
        """Mayor estricto."""
        return otro < self
    
    def __add__(self, otro: 'NumeroSurreal') -> 'NumeroSurreal':
        """
        Suma simplificada usando aproximaciones para evitar recursión infinita.
        """
        # Casos especiales
        if self == CERO:
            return otro
        if otro == CERO:
            return self
        
        # Para otros casos, usar aproximación numérica
        val_self = self.valor_aproximado()
        val_otro = otro.valor_aproximado()
        resultado_aprox = val_self + val_otro
        
        # Crear número surreal basado en el resultado aproximado
        return GeneradorSurreales.desde_real(resultado_aprox)
    
    def __neg__(self) -> 'NumeroSurreal':
        """Negación: -x = {-R_x | -L_x}"""
        L_neg = {-r for r in self.R}
        R_neg = {-l for l in self.L}
        return NumeroSurreal(L_neg, R_neg, f"-{self}")
    
    def __sub__(self, otro: 'NumeroSurreal') -> 'NumeroSurreal':
        """Resta: x - y = x + (-y)"""
        return self + (-otro)
    
    def __mul__(self, otro: 'NumeroSurreal') -> 'NumeroSurreal':
        """
        Multiplicación simplificada usando aproximaciones para evitar recursión infinita.
        """
        # Casos especiales para evitar recursión
        if self == CERO or otro == CERO:
            return CERO
        if self == UNO:
            return otro
        if otro == UNO:
            return self
        
        # Para otros casos, usar aproximación numérica
        val_self = self.valor_aproximado()
        val_otro = otro.valor_aproximado()
        resultado_aprox = val_self * val_otro
        
        # Crear número surreal basado en el resultado aproximado
        return GeneradorSurreales.desde_real(resultado_aprox)
    
    def valor_aproximado(self) -> float:
        """
        Calcula un valor aproximado del número surreal como float.
        Útil para visualización y cálculos aproximados.
        """
        if self.nombre and self.nombre in ["0", "cero"]:
            return 0.0
        elif self.nombre and self.nombre in ["1", "uno"]:
            return 1.0
        elif self.nombre and self.nombre in ["-1", "menos_uno"]:
            return -1.0
        elif self.nombre and "infinito" in self.nombre:
            return float('inf') if "positivo" in self.nombre else float('-inf')
        elif self.nombre and "infinitesimo" in self.nombre:
            return 1e-10 if "positivo" in self.nombre else -1e-10
        
        # Aproximación basada en los conjuntos L y R
        if not self.L and not self.R:
            return 0.0
        
        max_L = max([l.valor_aproximado() for l in self.L]) if self.L else float('-inf')
        min_R = min([r.valor_aproximado() for r in self.R]) if self.R else float('inf')
        
        if max_L == float('-inf') and min_R == float('inf'):
            return 0.0
        elif max_L == float('-inf'):
            return min_R - 1.0
        elif min_R == float('inf'):
            return max_L + 1.0
        else:
            return (max_L + min_R) / 2.0
    
    def __str__(self) -> str:
        """Representación como string."""
        if self.nombre:
            return self.nombre
        
        L_str = "{" + ", ".join(str(l) for l in sorted(self.L, key=lambda x: x.valor_aproximado())) + "}"
        R_str = "{" + ", ".join(str(r) for r in sorted(self.R, key=lambda x: x.valor_aproximado())) + "}"
        return f"{L_str}|{R_str}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        """Hash basado en el valor aproximado (para usar en sets)."""
        return hash(round(self.valor_aproximado(), 10))


# Definir números surreales básicos
CERO = NumeroSurreal(set(), set(), "0")
UNO = NumeroSurreal({CERO}, set(), "1")
MENOS_UNO = NumeroSurreal(set(), {CERO}, "-1")

# Crear más números básicos
def crear_numeros_basicos():
    """Crea números surreales básicos para usar en la red."""
    global DOS, MEDIO, UN_CUARTO, TRES_CUARTOS
    global INFINITO_POSITIVO, INFINITO_NEGATIVO
    global INFINITESIMO_POSITIVO, INFINITESIMO_NEGATIVO
    
    # Enteros básicos
    DOS = NumeroSurreal({UNO}, set(), "2")
    TRES = NumeroSurreal({DOS}, set(), "3")
    
    # Fracciones
    MEDIO = NumeroSurreal({CERO}, {UNO}, "1/2")
    UN_CUARTO = NumeroSurreal({CERO}, {MEDIO}, "1/4")
    TRES_CUARTOS = NumeroSurreal({MEDIO}, {UNO}, "3/4")
    
    # Infinitos (aproximación)
    INFINITO_POSITIVO = NumeroSurreal({CERO, UNO, DOS}, set(), "infinito_positivo")
    INFINITO_NEGATIVO = NumeroSurreal(set(), {CERO, MENOS_UNO}, "infinito_negativo")
    
    # Infinitésimos (aproximación)
    INFINITESIMO_POSITIVO = NumeroSurreal({CERO}, {UN_CUARTO, MEDIO, UNO}, "infinitesimo_positivo")
    INFINITESIMO_NEGATIVO = NumeroSurreal({MENOS_UNO}, {CERO}, "infinitesimo_negativo")

# Inicializar números básicos
crear_numeros_basicos()


class GeneradorSurreales:
    """Generador de números surreales para entrenamiento."""
    
    @staticmethod
    def numeros_basicos() -> List[NumeroSurreal]:
        """Retorna una lista de números surreales básicos."""
        return [
            CERO, UNO, MENOS_UNO, DOS, MEDIO, UN_CUARTO, TRES_CUARTOS,
            INFINITESIMO_POSITIVO, INFINITESIMO_NEGATIVO
        ]
    
    @staticmethod
    def aleatorio_simple() -> NumeroSurreal:
        """Genera un número surreal aleatorio simple."""
        import random
        basicos = GeneradorSurreales.numeros_basicos()
        return random.choice(basicos)
    
    @staticmethod
    def desde_real(valor: float) -> NumeroSurreal:
        """
        Convierte un número real a surreal (aproximación).
        """
        if valor == 0:
            return CERO
        elif valor == 1:
            return UNO
        elif valor == -1:
            return MENOS_UNO
        elif valor == 0.5:
            return MEDIO
        elif valor > 1000:
            return INFINITO_POSITIVO
        elif valor < -1000:
            return INFINITO_NEGATIVO
        elif 0 < valor < 0.001:
            return INFINITESIMO_POSITIVO
        elif -0.001 < valor < 0:
            return INFINITESIMO_NEGATIVO
        else:
            # Aproximación simple para otros valores
            if valor > 0:
                return NumeroSurreal({CERO}, {UNO}, f"~{valor}")
            else:
                return NumeroSurreal({MENOS_UNO}, {CERO}, f"~{valor}")


def test_numeros_surreales():
    """Prueba básica de los números surreales."""
    print("🧮 PRUEBA DE NÚMEROS SURREALES")
    print("=" * 40)
    
    # Prueba de igualdad y orden
    print(f"0 = 0: {CERO == CERO}")
    print(f"0 < 1: {CERO < UNO}")
    print(f"1 > 0: {UNO > CERO}")
    print(f"-1 < 0: {MENOS_UNO < CERO}")
    
    # Prueba de operaciones
    print(f"\n🔢 Operaciones:")
    print(f"0 + 1 ≈ {(CERO + UNO).valor_aproximado()}")
    print(f"1 + 1 ≈ {(UNO + UNO).valor_aproximado()}")
    print(f"1 - 1 ≈ {(UNO - UNO).valor_aproximado()}")
    print(f"-1 + 1 ≈ {(MENOS_UNO + UNO).valor_aproximado()}")
    
    # Valores aproximados
    print(f"\n📊 Valores aproximados:")
    numeros = [CERO, UNO, MENOS_UNO, MEDIO, INFINITESIMO_POSITIVO, INFINITO_POSITIVO]
    for num in numeros:
        print(f"{num}: {num.valor_aproximado()}")


if __name__ == "__main__":
    test_numeros_surreales()
