from colonia_hormigas import ColoniaHormigas
from utilidades import (
    generar_ciudades_aleatorias,
    calcular_matriz_distancias,
    visualizar_tour,
    visualizar_convergencia,
    visualizar_matriz_feromonas,
    validar_tour
)


def ejemplo_1_basico():
    print("\n" + "="*70)
    print("EJEMPLO 1: IMPLEMENTACIÓN BÁSICA DEL ACO")
    print("="*70)

    # PASO 1: Generar problema TSP
    print("\n[1/5] Generando problema TSP...")
    num_ciudades = 15
    ciudades = generar_ciudades_aleatorias(
        num_ciudades=num_ciudades,
        rango_x=(0, 100),
        rango_y=(0, 100),
        semilla=42
    )

    # Calcular matriz de distancias
    matriz_distancias = calcular_matriz_distancias(ciudades)
    print(f" Problema TSP generado: {num_ciudades} ciudades")

    # PASO 2: Configurar e inicializar ACO
    print("\n[2/5] Inicializando Colonia de Hormigas...")

    # CONFIGURACIÓN DE PARÁMETROS
    aco = ColoniaHormigas(
        matriz_distancias=matriz_distancias,
        num_hormigas=20,          # m ≈ n (número de nodos)
        alpha=1.0,                # Peso de feromona (exploración)
        beta=2.0,                 # Peso heurístico (codicia)
        rho=0.5,                  # Evaporación moderada
        Q=100.0,                  # Constante de depósito
        iteraciones_max=100,      # Suficiente para convergencia
        semilla_aleatoria=42
    )
    print("Colonia inicializada")

    # PASO 3: Ejecutar optimización
    print("\n[3/5] Ejecutando algoritmo ACO...")
    mejor_camino, mejor_distancia = aco.optimizar(verbose=True)

    # PASO 4: Validar solución
    print("\n[4/5] Validando solución...")
    es_valido = validar_tour(mejor_camino, num_ciudades)
    print(f"Tour válido: {es_valido}")

    if es_valido:
        print(f" Mejor distancia: {mejor_distancia:.2f}")
        print(f" Tour: {' -> '.join(map(str, mejor_camino[:5]))} ... "
              f"{' -> '.join(map(str, mejor_camino[-3:]))}")

    # PASO 5: Visualizaciones
    print("\n[5/5] Generando visualizaciones...")

    # Obtener estadísticas completas
    estadisticas = aco.obtener_estadisticas()

    # Visualizar tour óptimo
    visualizar_tour(
        ciudades,
        mejor_camino,
        titulo=f"Mejor Tour ACO (Distancia: {mejor_distancia:.2f})",
    )

    # Visualizar convergencia
    visualizar_convergencia(
        estadisticas['historial_mejor'],
        estadisticas['historial_promedio'],
        titulo="Convergencia del Algoritmo ACO - Ejemplo 1",
    )

    # Visualizar feromonas finales
    visualizar_matriz_feromonas(
        aco.feromonas,
        titulo="Distribución de Feromonas al Final - Ejemplo 1",
    )

def ejemplo_2_comparacion_parametros():
    """
    EJEMPLO 2: Análisis de Sensibilidad de Parámetros
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: ANÁLISIS DE SENSIBILIDAD DE PARÁMETROS")
    print("="*70)

    # Generar problema común
    print("\n[1/3] Generando problema TSP...")
    num_ciudades = 20
    ciudades = generar_ciudades_aleatorias(
        num_ciudades=num_ciudades,
        semilla=123
    )
    matriz_distancias = calcular_matriz_distancias(ciudades)
    print(f"Problema generado: {num_ciudades} ciudades")

    # Configuraciones a probar
    configuraciones = {
        'Alpha_alto': {'alpha': 2.0, 'beta': 2.0, 'rho': 0.5},
        'Alpha_bajo': {'alpha': 0.5, 'beta': 2.0, 'rho': 0.5},
        'Beta_alto': {'alpha': 1.0, 'beta': 5.0, 'rho': 0.5},
        'Beta_bajo': {'alpha': 1.0, 'beta': 0.5, 'rho': 0.5},
        'Rho_alto': {'alpha': 1.0, 'beta': 2.0, 'rho': 0.9},
        'Rho_bajo': {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1},
        'Balanceado': {'alpha': 1.0, 'beta': 2.0, 'rho': 0.5},
    }

    print(f"\n[2/3] Probando {len(configuraciones)} configuraciones...")
    resultados = {}

    for nombre, params in configuraciones.items():
        print(f"\n  Ejecutando: {nombre}")
        print(f"    α={params['alpha']}, β={params['beta']}, ρ={params['rho']}")

        aco = ColoniaHormigas(
            matriz_distancias=matriz_distancias,
            num_hormigas=20,
            alpha=params['alpha'],
            beta=params['beta'],
            rho=params['rho'],
            Q=100.0,
            iteraciones_max=50,
            semilla_aleatoria=456
        )

        aco.optimizar(verbose=False)
        resultados[nombre] = aco.obtener_estadisticas()

        print(f"    Mejor distancia: {resultados[nombre]['mejor_distancia']:.2f}")

    # Análisis comparativo
    print("\n[3/3] Generando análisis comparativo...")

       # Resumen de resultados
    print("\nRESUMEN DE COMPARACIÓN:")
    print("-" * 70)
    for nombre, resultado in resultados.items():
        print(f"{nombre:15s}: Distancia = {resultado['mejor_distancia']:7.2f}, "
              f"Tiempo = {resultado['tiempo_ejecucion']:5.2f}s")

    # Identificar mejor configuración
    mejor_config = min(resultados.items(), key=lambda x: x[1]['mejor_distancia'])
    print(f"\nMejor configuración: {mejor_config[0]}")
    print(f"   Distancia: {mejor_config[1]['mejor_distancia']:.2f}")
    print("\nAnálisis comparativo guardado")
    print("="*70)

    return resultados


def ejemplo_3_problema_grande():
    """
    EJEMPLO 3: Problema TSP de Mayor Escala
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: PROBLEMA TSP DE MAYOR ESCALA")
    print("="*70)

    # Generar problema grande
    print("\n[1/4] Generando problema TSP grande...")
    num_ciudades = 30
    ciudades = generar_ciudades_aleatorias(
        num_ciudades=num_ciudades,
        semilla=789
    )
    matriz_distancias = calcular_matriz_distancias(ciudades)
    import math
    print(f" Problema generado: {num_ciudades} ciudades")
    tours_posibles = math.factorial(num_ciudades-1)//2
    print(f"  Tours posibles: {tours_posibles:.2e}")

    # Configurar ACO para problema grande
    print("\n[2/4] Configurando ACO optimizado...")

    # AJUSTES PARA PROBLEMAS GRANDES:
    # - Más hormigas para mejor exploración
    # - Más iteraciones para convergencia
    # - Parámetros ajustados

    aco = ColoniaHormigas(
        matriz_distancias=matriz_distancias,
        num_hormigas=30,           # Incrementado
        alpha=1.0,
        beta=3.0,                  # Mayor peso heurístico
        rho=0.3,                   # Evaporación más lenta
        Q=100.0,
        iteraciones_max=150,       # Más iteraciones
        semilla_aleatoria=789
    )

    print("Colonia configurada para problema de escala")

    # Ejecutar optimización
    print("\n[3/4] Ejecutando optimización (esto tomará más tiempo)...")
    mejor_camino, mejor_distancia = aco.optimizar(verbose=True)

    estadisticas = aco.obtener_estadisticas()

    # Análisis de resultados
    print("\n[4/4] Analizando resultados...")

    # Calcular mejora
    dist_inicial = estadisticas['historial_promedio'][0]
    dist_final = mejor_distancia
    mejora = (dist_inicial - dist_final) / dist_inicial * 100

    print(f"\nRESULTADOS FINALES:")
    print("-" * 70)
    print(f"  Mejor distancia encontrada: {mejor_distancia:.2f}")
    print(f"  Distancia promedio inicial: {dist_inicial:.2f}")
    print(f"  Mejora total: {mejora:.2f}%")
    print(f"  Tiempo de ejecución: {estadisticas['tiempo_ejecucion']:.2f}s")
    print(f"  Iteraciones: {estadisticas['num_iteraciones']}")

    # Visualizaciones
    print("\nGenerando visualizaciones...")

    visualizar_tour(
        ciudades,
        mejor_camino,
        titulo=f"Tour Óptimo - 30 Ciudades (Dist: {mejor_distancia:.2f})",
    )

    visualizar_convergencia(
        estadisticas['historial_mejor'],
        estadisticas['historial_promedio'],
        titulo="Convergencia - Problema de 30 Ciudades",
    )

def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("\nEste programa demuestra la implementación completa del algoritmo ACO")
    print("aplicado al problema del Vendedor Viajero (TSP).")
    print("\nSe ejecutarán 3 ejemplos diferentes:")
    print("  1. Implementación básica (15 ciudades)")
    print("  2. Análisis de parámetros (20 ciudades)")
    print("  3. Problema de mayor escala (30 ciudades)")

    input("\nPresiona ENTER para comenzar...")

    try:
        # Ejecutar ejemplos
        print("INICIANDO EJEMPLOS")
        # Ejemplo 1
        ejemplo_1_basico()
        input("\nPresiona ENTER para continuar al Ejemplo 2...")

        # Ejemplo 2
        ejemplo_2_comparacion_parametros()
        input("\nPresiona ENTER para continuar al Ejemplo 3...")
        # Ejemplo 3
        ejemplo_3_problema_grande()
        print("\n Programa finalizado correctamente")

    except Exception as e:
        print(f"\n Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()