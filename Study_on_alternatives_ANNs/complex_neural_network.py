"""
Red Neuronal Multicapa con Números Complejos
=============================================

Una implementación de red neuronal de 4 capas (4-3-4-1) que opera con números imaginarios.
Utiliza función de activación sigmoide adaptada para números complejos.

Arquitectura:
- Capa de entrada: 4 neuronas
- Capa oculta 1: 3 neuronas  
- Capa oculta 2: 4 neuronas
- Capa de salida: 1 neurona
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd


class ComplexNeuralNetwork:
    """Red neuronal que opera con números complejos."""
    
    def __init__(self, architecture: List[int], learning_rate: float = 0.1):
        """
        Initialize complex neural network.
        
        Args:
            architecture: List with number of neurons per layer [4, 3, 4, 1]
            learning_rate: Learning rate for training process
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.num_layers = len(architecture)
        
        # Initialize weights and biases with random complex numbers
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Random complex weights with real and imaginary parts
            weight_real = np.random.randn(self.architecture[i+1], self.architecture[i]) * 0.5
            weight_imag = np.random.randn(self.architecture[i+1], self.architecture[i]) * 0.5
            complex_weight = weight_real + 1j * weight_imag
            self.weights.append(complex_weight)
            
            # Random complex biases
            bias_real = np.random.randn(self.architecture[i+1], 1) * 0.5
            bias_imag = np.random.randn(self.architecture[i+1], 1) * 0.5
            complex_bias = bias_real + 1j * bias_imag
            self.biases.append(complex_bias)
    
    def complex_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Complex sigmoid activation function.
        
        For complex number z = a + bi, applies:
        σ(z) = σ(a) * cos(b) + i * σ(a) * sin(b)
        where σ(x) = 1/(1 + exp(-x))
        
        Args:
            z: Array of complex numbers
            
        Returns:
            Array with complex sigmoid applied
        """
        # Separar parte real e imaginaria
        parte_real = np.real(z)
        parte_imag = np.imag(z)
        
        # Aplicar sigmoide a la parte real
        sigmoide_real = 1 / (1 + np.exp(-np.clip(parte_real, -500, 500)))
        
        # Crear la versión compleja
        resultado_real = sigmoide_real * np.cos(parte_imag)
        resultado_imag = sigmoide_real * np.sin(parte_imag)
        
        return resultado_real + 1j * resultado_imag
    
    def derivada_sigmoide_compleja(self, z: np.ndarray) -> np.ndarray:
        """
        Derivada de la función sigmoide compleja.
        
        Args:
            z: Array de números complejos
            
        Returns:
            Derivada de la sigmoide compleja
        """
        sig = self.sigmoide_compleja(z)
        # Para números complejos, la derivada es más compleja
        # Usamos una aproximación: σ'(z) ≈ σ(z) * (1 - |σ(z)|²)
        return sig * (1 - np.abs(sig)**2)
    
    def propagacion_adelante(self, entrada: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza la propagación hacia adelante.
        
        Args:
            entrada: Array de entrada con números complejos
            
        Returns:
            Tupla con (activaciones, z_valores) para cada capa
        """
        activaciones = [entrada]
        z_valores = []
        
        activacion_actual = entrada
        
        for i in range(self.numero_capas - 1):
            # Calcular z = W*a + b
            z = np.dot(self.pesos[i], activacion_actual) + self.sesgos[i]
            z_valores.append(z)
            
            # Aplicar función de activación
            activacion_actual = self.sigmoide_compleja(z)
            activaciones.append(activacion_actual)
        
        return activaciones, z_valores
    
    def retropropagacion(self, entrada: np.ndarray, salida_esperada: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza la retropropagación para calcular gradientes.
        
        Args:
            entrada: Datos de entrada
            salida_esperada: Salida esperada
            
        Returns:
            Gradientes para pesos y sesgos
        """
        m = entrada.shape[1]  # Número de ejemplos
        
        # Propagación hacia adelante
        activaciones, z_valores = self.propagacion_adelante(entrada)
        
        # Inicializar gradientes
        gradientes_pesos = [np.zeros_like(w) for w in self.pesos]
        gradientes_sesgos = [np.zeros_like(b) for b in self.sesgos]
        
        # Error en la capa de salida
        error = activaciones[-1] - salida_esperada
        
        # Retropropagación
        for i in range(self.numero_capas - 2, -1, -1):
            # Gradiente de la función de activación
            if i == self.numero_capas - 2:  # Capa de salida
                delta = error * self.derivada_sigmoide_compleja(z_valores[i])
            else:  # Capas ocultas
                delta = np.dot(self.pesos[i+1].T.conj(), delta) * self.derivada_sigmoide_compleja(z_valores[i])
            
            # Calcular gradientes
            gradientes_pesos[i] = np.dot(delta, activaciones[i].T.conj()) / m
            gradientes_sesgos[i] = np.sum(delta, axis=1, keepdims=True) / m
        
        return gradientes_pesos, gradientes_sesgos
    
    def entrenar(self, datos_entrada: np.ndarray, datos_salida: np.ndarray, epocas: int = 1000):
        """
        Entrena la red neuronal.
        
        Args:
            datos_entrada: Datos de entrenamiento de entrada
            datos_salida: Datos de entrenamiento de salida esperada
            epocas: Número de épocas de entrenamiento
        """
        historial_errores = []
        
        for epoca in range(epocas):
            # Calcular gradientes
            grad_pesos, grad_sesgos = self.retropropagacion(datos_entrada, datos_salida)
            
            # Actualizar pesos y sesgos
            for i in range(len(self.pesos)):
                self.pesos[i] -= self.tasa_aprendizaje * grad_pesos[i]
                self.sesgos[i] -= self.tasa_aprendizaje * grad_sesgos[i]
            
            # Calcular error para seguimiento
            if epoca % 100 == 0:
                predicciones = self.predecir(datos_entrada)
                error = np.mean(np.abs(predicciones - datos_salida)**2)
                historial_errores.append(error)
                print(f"Época {epoca}: Error = {error:.6f}")
        
        return historial_errores
    
    def predecir(self, entrada: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con la red entrenada.
        
        Args:
            entrada: Datos de entrada
            
        Returns:
            Predicciones de la red
        """
        activaciones, _ = self.propagacion_adelante(entrada)
        return activaciones[-1]


def generar_datos_complejos(num_muestras: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos de entrenamiento con números complejos.
    
    Args:
        num_muestras: Número de muestras a generar
        
    Returns:
        Tupla con (datos_entrada, datos_salida)
    """
    # Generar entradas complejas aleatorias
    entrada_real = np.random.uniform(-1, 1, (4, num_muestras))
    entrada_imag = np.random.uniform(-1, 1, (4, num_muestras))
    datos_entrada = entrada_real + 1j * entrada_imag
    
    # Crear una función objetivo compleja
    # Por ejemplo: f(z1, z2, z3, z4) = (z1 * z2 + z3 * z4) / 4
    salida_compleja = (datos_entrada[0] * datos_entrada[1] + 
                      datos_entrada[2] * datos_entrada[3]) / 4
    
    # Normalizar la salida
    salida_compleja = salida_compleja / (1 + np.abs(salida_compleja))
    datos_salida = salida_compleja.reshape(1, -1)
    
    return datos_entrada, datos_salida


def mostrar_tabla_comparacion(red: RedNeuronalCompleja, datos_entrada: np.ndarray, 
                             datos_salida: np.ndarray, num_ejemplos: int = 10):
    """
    Muestra una tabla comparando entradas, salidas esperadas y predicciones.
    
    Args:
        red: Red neuronal entrenada
        datos_entrada: Datos de entrada
        datos_salida: Salidas esperadas
        num_ejemplos: Número de ejemplos a mostrar
    """
    predicciones = red.predecir(datos_entrada)
    
    # Crear DataFrame para mejor visualización
    tabla_datos = []
    
    for i in range(min(num_ejemplos, datos_entrada.shape[1])):
        fila = {
            'Ejemplo': i + 1,
            'Entrada_1': f"{datos_entrada[0, i]:.3f}",
            'Entrada_2': f"{datos_entrada[1, i]:.3f}",
            'Entrada_3': f"{datos_entrada[2, i]:.3f}",
            'Entrada_4': f"{datos_entrada[3, i]:.3f}",
            'Salida_Esperada': f"{datos_salida[0, i]:.3f}",
            'Predicción': f"{predicciones[0, i]:.3f}",
            'Error': f"{abs(datos_salida[0, i] - predicciones[0, i]):.3f}"
        }
        tabla_datos.append(fila)
    
    df = pd.DataFrame(tabla_datos)
    print("\n" + "="*120)
    print("TABLA DE COMPARACIÓN - ENTRADAS vs PREDICCIONES")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)


def visualizar_resultados(red: RedNeuronalCompleja, datos_entrada: np.ndarray, datos_salida: np.ndarray):
    """
    Visualiza los resultados de la red neuronal.
    """
    predicciones = red.predecir(datos_entrada)
    
    # Crear subplots para parte real e imaginaria
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parte real
    ax1.scatter(np.real(datos_salida[0]), np.real(predicciones[0]), alpha=0.6, color='blue')
    ax1.plot([np.min(np.real(datos_salida[0])), np.max(np.real(datos_salida[0]))], 
             [np.min(np.real(datos_salida[0])), np.max(np.real(datos_salida[0]))], 'r--', lw=2)
    ax1.set_xlabel('Salida Esperada (Parte Real)')
    ax1.set_ylabel('Predicción (Parte Real)')
    ax1.set_title('Comparación - Parte Real')
    ax1.grid(True, alpha=0.3)
    
    # Parte imaginaria
    ax2.scatter(np.imag(datos_salida[0]), np.imag(predicciones[0]), alpha=0.6, color='green')
    ax2.plot([np.min(np.imag(datos_salida[0])), np.max(np.imag(datos_salida[0]))], 
             [np.min(np.imag(datos_salida[0])), np.max(np.imag(datos_salida[0]))], 'r--', lw=2)
    ax2.set_xlabel('Salida Esperada (Parte Imaginaria)')
    ax2.set_ylabel('Predicción (Parte Imaginaria)')
    ax2.set_title('Comparación - Parte Imaginaria')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_red_compleja.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Función principal que ejecuta el entrenamiento y evaluación de la red."""
    print("="*60)
    print("RED NEURONAL MULTICAPA CON NÚMEROS COMPLEJOS")
    print("="*60)
    print("Arquitectura: 4 → 3 → 4 → 1")
    print("Función de activación: Sigmoide compleja")
    print("="*60)
    
    # Crear la red neuronal
    arquitectura = [4, 3, 4, 1]
    red = RedNeuronalCompleja(arquitectura, tasa_aprendizaje=0.1)
    
    # Generar datos de entrenamiento
    print("\n📊 Generando datos de entrenamiento con números complejos...")
    datos_entrada, datos_salida = generar_datos_complejos(num_muestras=200)
    
    print(f"Forma de datos de entrada: {datos_entrada.shape}")
    print(f"Forma de datos de salida: {datos_salida.shape}")
    
    # Mostrar algunos ejemplos de datos
    print("\n🔍 Ejemplos de datos generados:")
    for i in range(3):
        print(f"Entrada {i+1}: [{datos_entrada[0,i]:.3f}, {datos_entrada[1,i]:.3f}, "
              f"{datos_entrada[2,i]:.3f}, {datos_entrada[3,i]:.3f}]")
        print(f"Salida esperada {i+1}: {datos_salida[0,i]:.3f}")
        print()
    
    # Entrenar la red
    print("🚀 Iniciando entrenamiento...")
    historial_errores = red.entrenar(datos_entrada, datos_salida, epocas=2000)
    
    # Evaluar la red
    print("\n📈 Evaluando la red entrenada...")
    predicciones = red.predecir(datos_entrada)
    error_final = np.mean(np.abs(predicciones - datos_salida)**2)
    print(f"Error final de entrenamiento: {error_final:.6f}")
    
    # Mostrar tabla de comparación
    mostrar_tabla_comparacion(red, datos_entrada, datos_salida, num_ejemplos=15)
    
    # Generar datos de prueba
    print("\n🧪 Generando datos de prueba...")
    datos_prueba_entrada, datos_prueba_salida = generar_datos_complejos(num_muestras=50)
    predicciones_prueba = red.predecir(datos_prueba_entrada)
    error_prueba = np.mean(np.abs(predicciones_prueba - datos_prueba_salida)**2)
    print(f"Error en datos de prueba: {error_prueba:.6f}")
    
    # Mostrar tabla de prueba
    print("\n📋 TABLA DE DATOS DE PRUEBA:")
    mostrar_tabla_comparacion(red, datos_prueba_entrada, datos_prueba_salida, num_ejemplos=10)
    
    # Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    try:
        visualizar_resultados(red, datos_entrada, datos_salida)
        print("✅ Gráficos guardados como 'resultados_red_compleja.png'")
    except Exception as e:
        print(f"⚠️  No se pudieron generar los gráficos: {e}")
    
    # Mostrar información de la red
    print(f"\n🧠 INFORMACIÓN DE LA RED:")
    print(f"   • Capas: {len(red.arquitectura)}")
    print(f"   • Neuronas por capa: {red.arquitectura}")
    print(f"   • Total de parámetros: {sum(w.size + b.size for w, b in zip(red.pesos, red.sesgos))}")
    print(f"   • Tasa de aprendizaje: {red.tasa_aprendizaje}")
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
