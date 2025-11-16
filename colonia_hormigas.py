import numpy as np
from typing import List, Tuple, Optional
from hormiga import Hormiga
import time

class ColoniaHormigas:
    """
    Implementa el algoritmo ACO (Ant Colony Optimization) para resolver TSP.

    PARÁMETROS CRÍTICOS:
    - α (alpha): Control de exploración vs explotación
      * α alto → más peso a la feromona (explotación)
      * α bajo → más aleatorio (exploración)

    - β (beta): Importancia de información heurística
      * β alto → más codicia (favorece aristas cortas)
      * β bajo → menos codicia

    - ρ (rho): Tasa de evaporación [0,1]
      * ρ alto → olvido rápido (más exploración)
      * ρ bajo → memoria persistente (más explotación)

    - Q: Constante de depósito de feromona
      * Escala la cantidad de feromona depositada
    """

    def __init__(
            self,
            matriz_distancias: np.ndarray,
            num_hormigas: int = 20,
            alpha: float = 1.0,
            beta: float = 2.0,
            rho: float = 0.5,
            Q: float = 100.0,
            iteraciones_max: int = 100,
            semilla_aleatoria: Optional[int] = None
    ):

        # Validación de parámetros
        assert 0 <= rho <= 1, "rho debe estar en [0,1]"
        assert alpha >= 0, "alpha debe ser no negativo"
        assert beta >= 0, "beta debe ser no negativo"
        assert num_hormigas > 0, "Debe haber al menos una hormiga"

        # Configuración del problema
        self.matriz_distancias = matriz_distancias
        self.num_nodos = len(matriz_distancias)

        # Parámetros del algoritmo
        self.num_hormigas = num_hormigas
        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # Tasa de evaporación
        self.Q = Q      # Constante de feromona
        self.iteraciones_max = iteraciones_max

        # Control de aleatoriedad para reproducibilidad
        if semilla_aleatoria is not None:
            np.random.seed(semilla_aleatoria)

        # CONCEPTO: Todas las aristas inician con la misma cantidad de feromona
        # RAZÓN: Evitar sesgos iniciales, permitir exploración uniforme
        # FÓRMULA: τ₀ = 1/(n * L_nn) donde L_nn es longitud del tour greedy
        tour_greedy_length = self._calcular_tour_greedy()
        tau_inicial = 1.0 / (self.num_nodos * tour_greedy_length)
        self.feromonas = np.ones((self.num_nodos, self.num_nodos)) * tau_inicial

        # Variables de seguimiento
        self.mejor_camino: Optional[List[int]] = None
        self.mejor_distancia: float = float('inf')
        self.historial_mejor_distancia: List[float] = []
        self.historial_distancia_promedio: List[float] = []
        self.tiempo_ejecucion: float = 0.0

        # Crear población de hormigas
        self.hormigas: List[Hormiga] = [
            Hormiga(self.num_nodos, nodo_inicio=i % self.num_nodos)
            for i in range(num_hormigas)
        ]

    def _calcular_tour_greedy(self) -> float:
        """
        Calcula un tour usando heurística del vecino más cercano.

        Obtener una aproximación inicial para:
        1. Inicializar feromonas con valor razonable
        2. Tener una cota superior del problema

        ALGORITMO GREEDY (Nearest Neighbor):
        1. Iniciar en nodo 0
        2. Mientras haya nodos no visitados:
           - Ir al nodo no visitado más cercano
        3. Regresar al inicio

        Returns:
            Longitud del tour greedy
        """
        visitados = {0}
        nodo_actual = 0
        distancia_total = 0.0

        while len(visitados) < self.num_nodos:
            # Encontrar nodo más cercano no visitado
            distancias = [
                (nodo, self.matriz_distancias[nodo_actual][nodo])
                for nodo in range(self.num_nodos)
                if nodo not in visitados
            ]
            nodo_mas_cercano = min(distancias, key=lambda x: x[1])[0]

            distancia_total += self.matriz_distancias[nodo_actual][nodo_mas_cercano]
            visitados.add(nodo_mas_cercano)
            nodo_actual = nodo_mas_cercano

        # Cerrar tour
        distancia_total += self.matriz_distancias[nodo_actual][0]
        return distancia_total

    def _evaporar_feromonas(self):
        """
        Aplica evaporación a todas las feromonas.
        """
        self.feromonas *= (1 - self.rho)

    def _depositar_feromonas(self):
        """
        Deposita feromonas basándose en las soluciones de las hormigas.
        """
        # Matriz temporal para acumular depósitos
        delta_feromonas = np.zeros_like(self.feromonas)

        for hormiga in self.hormigas:
            camino, distancia = hormiga.obtener_solucion()
            # REFUERZO POSITIVO: Inversamente proporcional a distancia
            # Tours más cortos depositan más feromona
            deposito = self.Q / distancia

            # Depositar en cada arista del tour
            for i in range(len(camino) - 1):
                nodo_desde = camino[i]
                nodo_hasta = camino[i + 1]
                # Depositar en ambas direcciones (grafo no dirigido)
                delta_feromonas[nodo_desde][nodo_hasta] += deposito
                delta_feromonas[nodo_hasta][nodo_desde] += deposito
        # ACTUALIZACIÓN FINAL: Combinar evaporación + depósito
        self.feromonas += delta_feromonas

    def _actualizar_mejor_solucion(self):
        """
        Actualiza la mejor solución encontrada hasta el momento.
        """
        for hormiga in self.hormigas:
            camino, distancia = hormiga.obtener_solucion()

            if distancia < self.mejor_distancia:
                self.mejor_distancia = distancia
                self.mejor_camino = camino.copy()

    def _calcular_estadisticas_iteracion(self) -> Tuple[float, float]:
        """
        Calcula estadísticas de la iteración actual.

        Returns:
            Tupla (mejor_distancia_iteracion, distancia_promedio)
        """
        distancias = [hormiga.obtener_solucion()[1] for hormiga in self.hormigas]
        return min(distancias), np.mean(distancias)

    def optimizar(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Ejecuta el algoritmo ACO completo.
        Args:
            verbose: Si True, muestra progreso durante ejecución
        Returns:
            Tupla (mejor_camino, mejor_distancia)
        """
        inicio = time.time()

        if verbose:
            print("="*70)
            print("INICIANDO OPTIMIZACIÓN POR COLONIA DE HORMIGAS (ACO)")
            print("="*70)
            print(f"Configuración:")
            print(f"  - Nodos: {self.num_nodos}")
            print(f"  - Hormigas: {self.num_hormigas}")
            print(f"  - Alpha (α): {self.alpha} (influencia feromona)")
            print(f"  - Beta (β): {self.beta} (influencia heurística)")
            print(f"  - Rho (ρ): {self.rho} (evaporación)")
            print(f"  - Q: {self.Q} (constante feromona)")
            print(f"  - Iteraciones: {self.iteraciones_max}")
            print("="*70)

        # CICLO PRINCIPAL DEL ALGORITMO ACO
        for iteracion in range(self.iteraciones_max):

            # Cada hormiga construye una solución completa
            for hormiga in self.hormigas:
                hormiga.reiniciar(nodo_inicio=np.random.randint(0, self.num_nodos))
                hormiga.construir_solucion(
                    self.matriz_distancias,
                    self.feromonas,
                    self.alpha,
                    self.beta
                )


            # 2.1: Evaporación (olvido)
            self._evaporar_feromonas()

            # 2.2: Depósito (refuerzo)
            self._depositar_feromonas()

            # FASE 3: ACTUALIZACIÓN DE MEJOR SOLUCIÓN
            self._actualizar_mejor_solucion()

            # FASE 4: ESTADÍSTICAS Y SEGUIMIENTO
            mejor_iter, promedio_iter = self._calcular_estadisticas_iteracion()
            self.historial_mejor_distancia.append(self.mejor_distancia)
            self.historial_distancia_promedio.append(promedio_iter)

            # Mostrar progreso
            if verbose and (iteracion % 10 == 0 or iteracion == self.iteraciones_max - 1):
                print(f"Iter {iteracion+1:3d} | "
                      f"Mejor global: {self.mejor_distancia:.2f} | "
                      f"Mejor iter: {mejor_iter:.2f} | "
                      f"Promedio: {promedio_iter:.2f}")

        self.tiempo_ejecucion = time.time() - inicio

        if verbose:
            print("="*70)
            print("OPTIMIZACIÓN COMPLETADA")
            print(f"Tiempo de ejecución: {self.tiempo_ejecucion:.2f} segundos")
            print(f"Mejor distancia encontrada: {self.mejor_distancia:.2f}")
            print("="*70)

        return self.mejor_camino, self.mejor_distancia

    def obtener_estadisticas(self) -> dict:
        """
        Retorna estadísticas completas de la ejecución.
        Returns:
            Diccionario con estadísticas del algoritmo
        """
        return {
            'mejor_camino': self.mejor_camino,
            'mejor_distancia': self.mejor_distancia,
            'historial_mejor': self.historial_mejor_distancia,
            'historial_promedio': self.historial_distancia_promedio,
            'tiempo_ejecucion': self.tiempo_ejecucion,
            'num_iteraciones': self.iteraciones_max,
            'parametros': {
                'num_hormigas': self.num_hormigas,
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'Q': self.Q
            }
        }
