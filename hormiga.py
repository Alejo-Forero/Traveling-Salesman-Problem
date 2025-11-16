import numpy as np
from typing import List, Set


class Hormiga:
    """
    Representa una hormiga artificial en el algoritmo ACO.
    """

    def __init__(self, num_nodos: int, nodo_inicio: int = 0):
        """
        Inicializa una hormiga en el sistema.
        Args:
            num_nodos: Cantidad total de nodos en el grafo
            nodo_inicio: Nodo desde donde inicia la hormiga
        """
        self.num_nodos = num_nodos
        self.nodo_actual = nodo_inicio
        self.camino: List[int] = [nodo_inicio]
        self.lista_tabu: Set[int] = {nodo_inicio}
        self.distancia_total = 0.0

    def seleccionar_siguiente_nodo(
            self,
            feromonas: np.ndarray,
            heuristica: np.ndarray,
            alpha: float,
            beta: float
    ) -> int:
        """
         Args:
            feromonas: Matriz de feromonas τ(i,j)
            heuristica: Matriz de información heurística η(i,j)
            alpha: Parámetro α (influencia de feromona)
            beta: Parámetro β (influencia de heurística)
        Returns:
            Índice del siguiente nodo seleccionado
        """
        # Obtener nodos no visitados (factibles)
        nodos_no_visitados = [n for n in range(self.num_nodos)
                              if n not in self.lista_tabu]

        if not nodos_no_visitados:
            # Caso especial: regresar al inicio (completar tour en TSP)
            return self.camino[0]

        # Extraer valores de feromona para aristas factibles
        feromonas_factibles = np.array([
            feromonas[self.nodo_actual][j] for j in nodos_no_visitados
        ])

        # Extraer valores heurísticos para aristas factibles
        heuristica_factibles = np.array([
            heuristica[self.nodo_actual][j] for j in nodos_no_visitados
        ])

        # Aplicar exponentes α y β (importancia relativa)
        # tau^alpha representa la MEMORIA COLECTIVA
        tau_alpha = np.power(feromonas_factibles + 1e-10, alpha)

        # eta^beta representa la CODICIA (información heurística)
        eta_beta = np.power(heuristica_factibles + 1e-10, beta)

        # Producto: combina ambas fuentes de información
        probabilidades = tau_alpha * eta_beta

        # Normalizar para obtener distribución de probabilidad válida
        suma_probabilidades = np.sum(probabilidades)

        if suma_probabilidades == 0:
            # Caso especial: distribución uniforme si no hay información
            probabilidades = np.ones(len(nodos_no_visitados))

        probabilidades = probabilidades / np.sum(probabilidades)

        nodo_seleccionado_idx = np.random.choice(
            len(nodos_no_visitados),
            p=probabilidades
        )

        return nodos_no_visitados[nodo_seleccionado_idx]

    def visitar_nodo(self, nodo: int, distancia: float):
        """
        Registra la visita a un nuevo nodo.
        Args:
            nodo: Nodo a visitar
            distancia: Distancia desde el nodo actual al nuevo nodo
        """
        self.camino.append(nodo)
        self.lista_tabu.add(nodo)
        self.distancia_total += distancia
        self.nodo_actual = nodo

    def construir_solucion(
            self,
            matriz_distancias: np.ndarray,
            feromonas: np.ndarray,
            alpha: float,
            beta: float
    ):
        """
        Construye una solución completa (tour) para el TSP.
        Args:
            matriz_distancias: Matriz de distancias entre nodos
            feromonas: Matriz actual de feromonas
            alpha: Parámetro de influencia de feromona
            beta: Parámetro de influencia heurística
        """
        heuristica = 1.0 / (matriz_distancias + 1e-10)

        # Construir tour visitando todos los nodos
        while len(self.camino) < self.num_nodos:
            siguiente_nodo = self.seleccionar_siguiente_nodo(
                feromonas, heuristica, alpha, beta
            )

            distancia = matriz_distancias[self.nodo_actual][siguiente_nodo]
            self.visitar_nodo(siguiente_nodo, distancia)

        # Cerrar el tour: regresar al nodo inicial
        distancia_retorno = matriz_distancias[self.nodo_actual][self.camino[0]]
        self.distancia_total += distancia_retorno
        self.camino.append(self.camino[0])

    def obtener_solucion(self) -> tuple:
        """
        Retorna la solución construida por la hormiga.
        Returns:
            Tupla (camino, distancia_total)
        """
        return self.camino, self.distancia_total

    def reiniciar(self, nodo_inicio: int = 0):
        """
        Reinicia la hormiga para una nueva iteración.
        Args:
            nodo_inicio: Nodo desde donde reiniciar
        """
        self.nodo_actual = nodo_inicio
        self.camino = [nodo_inicio]
        self.lista_tabu = {nodo_inicio}
        self.distancia_total = 0.0
