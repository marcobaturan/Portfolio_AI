"""
Red Neuronal Profunda con Números Complejos - 50 Capas x 50 Neuronas
====================================================================

Una implementación de red neuronal extremadamente profunda que opera con números imaginarios.
Arquitectura: 4 → [50 x 50 capas] → 1

Incluye técnicas avanzadas para entrenar redes profundas:
- Inicialización Xavier/He adaptada para números complejos
- Normalización de gradientes
- Monitoreo de gradientes que explotan/desaparecen
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
import time
from red_neuronal_compleja import generar_datos_complejos


class RedNeuronalProfundaCompleja:
    """Red neuronal profunda que opera con números complejos."""
    
    def __init__(self, capas_entrada: int = 4, capas_ocultas: int = 50, 
                 neuronas_por_capa: int = 50, capas_salida: int = 1, 
                 tasa_aprendizaje: float = 0.001):
        """
        Inicializa la red neuronal profunda con números complejos.
        
        Args:
            capas_entrada: Número de neuronas en la capa de entrada
            capas_ocultas: Número de capas ocultas
            neuronas_por_capa: Neuronas por cada capa oculta
            capas_salida: Número de neuronas en la capa de salida
            tasa_aprendizaje: Tasa de aprendizaje (más baja para redes profundas)
        """
        # Construir arquitectura
        self.arquitectura = [capas_entrada]
        for _ in range(capas_ocultas):
            self.arquitectura.append(neuronas_por_capa)
        self.arquitectura.append(capas_salida)
        
        self.tasa_aprendizaje = tasa_aprendizaje
        self.numero_capas = len(self.arquitectura)
        self.capas_ocultas = capas_ocultas
        
        print(f"🏗️  Creando red neuronal profunda:")
        print(f"   • Arquitectura: {capas_entrada} → [{capas_ocultas} capas × {neuronas_por_capa} neuronas] → {capas_salida}")
        print(f"   • Total de capas: {self.numero_capas}")
        print(f"   • Total de parámetros: {self._calcular_parametros()}")
        
        # Inicializar pesos y sesgos con técnica Xavier/He adaptada
        self.pesos = []
        self.sesgos = []
        self._inicializar_pesos_xavier()
        
        # Variables para monitoreo
        self.historial_gradientes = []
        self.historial_activaciones = []
    
    def _calcular_parametros(self) -> int:
        """Calcula el número total de parámetros."""
        total = 0
        for i in range(self.numero_capas - 1):
            # Pesos: entrada × salida, Sesgos: salida
            total += self.arquitectura[i] * self.arquitectura[i+1] + self.arquitectura[i+1]
        return total * 2  # x2 porque cada parámetro es complejo (real + imaginario)
    
    def _inicializar_pesos_xavier(self):
        """Inicialización Xavier/He adaptada para números complejos y redes profundas."""
        for i in range(self.numero_capas - 1):
            fan_in = self.arquitectura[i]
            fan_out = self.arquitectura[i+1]
            
            # Inicialización Xavier modificada para números complejos
            limite = np.sqrt(2.0 / (fan_in + fan_out))
            
            # Pesos complejos con inicialización cuidadosa
            peso_real = np.random.normal(0, limite, (fan_out, fan_in))
            peso_imag = np.random.normal(0, limite, (fan_out, fan_in))
            peso_complejo = peso_real + 1j * peso_imag
            self.pesos.append(peso_complejo)
            
            # Sesgos pequeños para redes profundas
            sesgo_real = np.random.normal(0, 0.01, (fan_out, 1))
            sesgo_imag = np.random.normal(0, 0.01, (fan_out, 1))
            sesgo_complejo = sesgo_real + 1j * sesgo_imag
            self.sesgos.append(sesgo_complejo)
    
    def sigmoide_compleja(self, z: np.ndarray) -> np.ndarray:
        """
        Función sigmoide adaptada para números complejos con estabilización.
        """
        # Separar parte real e imaginaria
        parte_real = np.real(z)
        parte_imag = np.imag(z)
        
        # Clipping más agresivo para redes profundas
        parte_real = np.clip(parte_real, -10, 10)
        parte_imag = np.clip(parte_imag, -10, 10)
        
        # Aplicar sigmoide a la parte real
        sigmoide_real = 1 / (1 + np.exp(-parte_real))
        
        # Crear la versión compleja con estabilización
        resultado_real = sigmoide_real * np.cos(parte_imag)
        resultado_imag = sigmoide_real * np.sin(parte_imag)
        
        return resultado_real + 1j * resultado_imag
    
    def derivada_sigmoide_compleja(self, z: np.ndarray) -> np.ndarray:
        """Derivada de la función sigmoide compleja con estabilización."""
        sig = self.sigmoide_compleja(z)
        # Evitar gradientes que desaparecen
        magnitud = np.abs(sig)
        factor_estabilizacion = np.maximum(magnitud, 0.01)  # Evitar división por cero
        return sig * (1 - magnitud**2) * factor_estabilizacion
    
    def propagacion_adelante(self, entrada: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza la propagación hacia adelante con monitoreo de activaciones.
        """
        activaciones = [entrada]
        z_valores = []
        
        activacion_actual = entrada
        magnitudes_activacion = []
        
        for i in range(self.numero_capas - 1):
            # Calcular z = W*a + b
            z = np.dot(self.pesos[i], activacion_actual) + self.sesgos[i]
            z_valores.append(z)
            
            # Aplicar función de activación
            activacion_actual = self.sigmoide_compleja(z)
            activaciones.append(activacion_actual)
            
            # Monitorear magnitudes para detectar gradientes que explotan/desaparecen
            magnitud_promedio = np.mean(np.abs(activacion_actual))
            magnitudes_activacion.append(magnitud_promedio)
        
        self.historial_activaciones.append(magnitudes_activacion)
        return activaciones, z_valores
    
    def retropropagacion(self, entrada: np.ndarray, salida_esperada: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Retropropagación con monitoreo de gradientes y clipping.
        """
        m = entrada.shape[1]
        
        # Propagación hacia adelante
        activaciones, z_valores = self.propagacion_adelante(entrada)
        
        # Inicializar gradientes
        gradientes_pesos = [np.zeros_like(w) for w in self.pesos]
        gradientes_sesgos = [np.zeros_like(b) for b in self.sesgos]
        
        # Error en la capa de salida
        error = activaciones[-1] - salida_esperada
        
        # Retropropagación con monitoreo
        magnitudes_gradientes = []
        
        for i in range(self.numero_capas - 2, -1, -1):
            # Gradiente de la función de activación
            if i == self.numero_capas - 2:  # Capa de salida
                delta = error * self.derivada_sigmoide_compleja(z_valores[i])
            else:  # Capas ocultas
                delta = np.dot(self.pesos[i+1].T.conj(), delta) * self.derivada_sigmoide_compleja(z_valores[i])
            
            # Clipping de gradientes para evitar explosión
            magnitud_delta = np.abs(delta)
            if np.max(magnitud_delta) > 5.0:  # Threshold para clipping
                delta = delta * (5.0 / np.max(magnitud_delta))
            
            # Calcular gradientes
            gradientes_pesos[i] = np.dot(delta, activaciones[i].T.conj()) / m
            gradientes_sesgos[i] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Monitorear magnitudes de gradientes
            magnitud_grad = np.mean(np.abs(gradientes_pesos[i]))
            magnitudes_gradientes.append(magnitud_grad)
        
        self.historial_gradientes.append(magnitudes_gradientes)
        return gradientes_pesos, gradientes_sesgos
    
    def entrenar(self, datos_entrada: np.ndarray, datos_salida: np.ndarray, epocas: int = 500):
        """
        Entrena la red neuronal profunda con monitoreo avanzado.
        """
        print(f"\n🚀 Iniciando entrenamiento de red profunda ({epocas} épocas)...")
        historial_errores = []
        tiempo_inicio = time.time()
        
        for epoca in range(epocas):
            # Calcular gradientes
            grad_pesos, grad_sesgos = self.retropropagacion(datos_entrada, datos_salida)
            
            # Actualizar pesos y sesgos con tasa de aprendizaje adaptativa
            tasa_actual = self.tasa_aprendizaje
            if epoca > 200:  # Reducir tasa después de 200 épocas
                tasa_actual *= 0.5
            
            for i in range(len(self.pesos)):
                self.pesos[i] -= tasa_actual * grad_pesos[i]
                self.sesgos[i] -= tasa_actual * grad_sesgos[i]
            
            # Calcular error y mostrar progreso
            if epoca % 50 == 0:
                predicciones = self.predecir(datos_entrada)
                error = np.mean(np.abs(predicciones - datos_salida)**2)
                historial_errores.append(error)
                
                tiempo_transcurrido = time.time() - tiempo_inicio
                print(f"Época {epoca:4d}: Error = {error:.6f} | Tiempo: {tiempo_transcurrido:.1f}s")
                
                # Verificar convergencia
                if error < 0.001:
                    print(f"✅ Convergencia alcanzada en época {epoca}")
                    break
        
        tiempo_total = time.time() - tiempo_inicio
        print(f"⏱️  Tiempo total de entrenamiento: {tiempo_total:.2f} segundos")
        return historial_errores
    
    def predecir(self, entrada: np.ndarray) -> np.ndarray:
        """Realiza predicciones con la red entrenada."""
        activaciones, _ = self.propagacion_adelante(entrada)
        return activaciones[-1]
    
    def diagnosticar_red(self):
        """Diagnóstica problemas comunes en redes profundas."""
        print("\n🔍 DIAGNÓSTICO DE LA RED PROFUNDA:")
        print("=" * 50)
        
        if self.historial_gradientes:
            gradientes_recientes = self.historial_gradientes[-1]
            grad_min = min(gradientes_recientes)
            grad_max = max(gradientes_recientes)
            
            print(f"📊 Gradientes:")
            print(f"   • Mínimo: {grad_min:.2e}")
            print(f"   • Máximo: {grad_max:.2e}")
            
            if grad_max > 1.0:
                print("   ⚠️  Posible explosión de gradientes")
            elif grad_max < 1e-6:
                print("   ⚠️  Posible desvanecimiento de gradientes")
            else:
                print("   ✅ Gradientes en rango saludable")
        
        if self.historial_activaciones:
            activaciones_recientes = self.historial_activaciones[-1]
            act_min = min(activaciones_recientes)
            act_max = max(activaciones_recientes)
            
            print(f"📈 Activaciones:")
            print(f"   • Mínimo: {act_min:.6f}")
            print(f"   • Máximo: {act_max:.6f}")
            
            if act_max < 0.01:
                print("   ⚠️  Activaciones muy pequeñas (neurona muerta)")
            elif act_max > 10:
                print("   ⚠️  Activaciones muy grandes")
            else:
                print("   ✅ Activaciones en rango saludable")


def comparar_redes():
    """Compara la red simple (4-3-4-1) con la red profunda (4-[50x50]-1)."""
    print("="*80)
    print("COMPARACIÓN: RED SIMPLE vs RED PROFUNDA")
    print("="*80)
    
    # Importar la red simple
    from red_neuronal_compleja import RedNeuronalCompleja
    
    # Generar los mismos datos para ambas redes
    print("📊 Generando datos de prueba...")
    np.random.seed(42)  # Para reproducibilidad
    datos_entrada, datos_salida = generar_datos_complejos(num_muestras=100)
    datos_test_entrada, datos_test_salida = generar_datos_complejos(num_muestras=30)
    
    # Crear y entrenar red simple
    print("\n🏗️  RED SIMPLE (4-3-4-1):")
    red_simple = RedNeuronalCompleja([4, 3, 4, 1], tasa_aprendizaje=0.1)
    tiempo_inicio_simple = time.time()
    historial_simple = red_simple.entrenar(datos_entrada, datos_salida, epocas=1000)
    tiempo_simple = time.time() - tiempo_inicio_simple
    
    # Evaluar red simple
    pred_simple = red_simple.predecir(datos_test_entrada)
    error_simple = np.mean(np.abs(pred_simple - datos_test_salida)**2)
    
    # Crear y entrenar red profunda
    print(f"\n🏗️  RED PROFUNDA (4-[50×50]-1):")
    red_profunda = RedNeuronalProfundaCompleja(
        capas_entrada=4, 
        capas_ocultas=50, 
        neuronas_por_capa=50, 
        capas_salida=1,
        tasa_aprendizaje=0.001
    )
    
    tiempo_inicio_profunda = time.time()
    historial_profunda = red_profunda.entrenar(datos_entrada, datos_salida, epocas=300)
    tiempo_profunda = time.time() - tiempo_inicio_profunda
    
    # Evaluar red profunda
    pred_profunda = red_profunda.predecir(datos_test_entrada)
    error_profunda = np.mean(np.abs(pred_profunda - datos_test_salida)**2)
    
    # Diagnóstico de red profunda
    red_profunda.diagnosticar_red()
    
    # Crear tabla comparativa
    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("="*80)
    
    comparacion = pd.DataFrame({
        'Métrica': [
            'Arquitectura',
            'Total Parámetros',
            'Tiempo Entrenamiento (s)',
            'Error Final Test',
            'Épocas Entrenadas',
            'Velocidad (épocas/s)'
        ],
        'Red Simple': [
            '4-3-4-1',
            f'{sum(w.size + b.size for w, b in zip(red_simple.pesos, red_simple.sesgos))}',
            f'{tiempo_simple:.2f}',
            f'{error_simple:.6f}',
            '1000',
            f'{1000/tiempo_simple:.1f}'
        ],
        'Red Profunda': [
            '4-[50×50]-1',
            f'{red_profunda._calcular_parametros()}',
            f'{tiempo_profunda:.2f}',
            f'{error_profunda:.6f}',
            f'{len(historial_profunda)*50}',
            f'{len(historial_profunda)*50/tiempo_profunda:.1f}'
        ]
    })
    
    print(comparacion.to_string(index=False))
    
    # Comparar predicciones en ejemplos específicos
    print(f"\n📋 COMPARACIÓN DE PREDICCIONES (primeros 10 ejemplos):")
    print("="*100)
    
    tabla_pred = []
    for i in range(min(10, datos_test_entrada.shape[1])):
        fila = {
            'Ejemplo': i + 1,
            'Esperado': f"{datos_test_salida[0, i]:.4f}",
            'Red Simple': f"{pred_simple[0, i]:.4f}",
            'Red Profunda': f"{pred_profunda[0, i]:.4f}",
            'Error Simple': f"{abs(datos_test_salida[0, i] - pred_simple[0, i]):.4f}",
            'Error Profunda': f"{abs(datos_test_salida[0, i] - pred_profunda[0, i]):.4f}"
        }
        tabla_pred.append(fila)
    
    df_pred = pd.DataFrame(tabla_pred)
    print(df_pred.to_string(index=False))
    
    # Conclusiones
    print(f"\n🎯 CONCLUSIONES:")
    print("="*50)
    if error_simple < error_profunda:
        print("✅ La red SIMPLE obtuvo mejor precisión")
        print(f"   • Diferencia de error: {error_profunda - error_simple:.6f}")
    else:
        print("✅ La red PROFUNDA obtuvo mejor precisión")
        print(f"   • Diferencia de error: {error_simple - error_profunda:.6f}")
    
    if tiempo_simple < tiempo_profunda:
        print(f"⚡ La red SIMPLE fue {tiempo_profunda/tiempo_simple:.1f}x más rápida")
    else:
        print(f"⚡ La red PROFUNDA fue {tiempo_simple/tiempo_profunda:.1f}x más rápida")
    
    eficiencia_simple = 1 / (error_simple * tiempo_simple)
    eficiencia_profunda = 1 / (error_profunda * tiempo_profunda)
    
    if eficiencia_simple > eficiencia_profunda:
        print(f"🏆 La red SIMPLE es más eficiente (precisión/tiempo)")
    else:
        print(f"🏆 La red PROFUNDA es más eficiente (precisión/tiempo)")
    
    return {
        'red_simple': red_simple,
        'red_profunda': red_profunda,
        'error_simple': error_simple,
        'error_profunda': error_profunda,
        'tiempo_simple': tiempo_simple,
        'tiempo_profunda': tiempo_profunda
    }


if __name__ == "__main__":
    resultados = comparar_redes()
