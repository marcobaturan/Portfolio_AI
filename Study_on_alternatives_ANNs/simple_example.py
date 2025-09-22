"""
Ejemplo Simple de Red Neuronal con Números Complejos
===================================================

Un ejemplo rápido y directo para probar la funcionalidad básica.
"""

from red_neuronal_compleja import RedNeuronalCompleja, generar_datos_complejos, mostrar_tabla_comparacion
import numpy as np


def ejemplo_rapido():
    """Ejecuta un ejemplo rápido de la red neuronal."""
    print("🔥 EJEMPLO RÁPIDO - RED NEURONAL COMPLEJA")
    print("="*50)
    
    # Crear red con arquitectura 4-3-4-1
    red = RedNeuronalCompleja([4, 3, 4, 1], tasa_aprendizaje=0.2)
    
    # Generar pocos datos para ejemplo rápido
    entrada, salida = generar_datos_complejos(num_muestras=20)
    
    print("📊 Datos de ejemplo generados:")
    print(f"   • Entradas: {entrada.shape}")
    print(f"   • Salidas: {salida.shape}")
    
    # Entrenar por pocas épocas
    print("\n🚀 Entrenando (500 épocas)...")
    red.entrenar(entrada, salida, epocas=500)
    
    # Mostrar resultados
    print("\n📋 RESULTADOS:")
    mostrar_tabla_comparacion(red, entrada, salida, num_ejemplos=5)
    
    # Probar con datos nuevos
    entrada_test, salida_test = generar_datos_complejos(num_muestras=5)
    predicciones = red.predecir(entrada_test)
    
    print("\n🧪 DATOS DE PRUEBA:")
    for i in range(5):
        print(f"Entrada {i+1}: {entrada_test[:, i]}")
        print(f"Esperado: {salida_test[0, i]:.4f}")
        print(f"Predicho: {predicciones[0, i]:.4f}")
        print(f"Error: {abs(salida_test[0, i] - predicciones[0, i]):.4f}")
        print("-" * 30)


if __name__ == "__main__":
    ejemplo_rapido()
