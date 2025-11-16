from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np

def generar_ciudades_aleatorias(
        num_ciudades: int,
        rango_x: Tuple[float, float] = (0, 100),
        rango_y: Tuple[float, float] = (0, 100),
        semilla: int = None
) -> np.ndarray:
    """
    Genera coordenadas aleatorias de ciudades en un plano 2D.
    Args:
        num_ciudades: Número de ciudades a generar
        rango_x: Rango de coordenadas X (min, max)
        rango_y: Rango de coordenadas Y (min, max)
        semilla: Semilla para reproducibilidad
    Returns:
        Array de forma (num_ciudades, 2) con coordenadas (x, y)
    """
    if semilla is not None:
        np.random.seed(semilla)

    ciudades = np.random.uniform(
        low=[rango_x[0], rango_y[0]],
        high=[rango_x[1], rango_y[1]],
        size=(num_ciudades, 2)
    )

    return ciudades

def calcular_matriz_distancias(ciudades: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de distancias euclidianas entre ciudades.
    Args:
        ciudades: Array de coordenadas de ciudades

    Returns:
        Matriz NxN de distancias euclidianas
    """
    num_ciudades = len(ciudades)
    matriz = np.zeros((num_ciudades, num_ciudades))

    for i in range(num_ciudades):
        for j in range(num_ciudades):
            if i != j:
                dx = ciudades[i][0] - ciudades[j][0]
                dy = ciudades[i][1] - ciudades[j][1]
                matriz[i][j] = np.sqrt(dx**2 + dy**2)

    return matriz

def validar_tour(tour: List[int], num_nodos: int) -> bool:
    """
    Valida que un tour sea una solución válida del TSP.
    Args:
        tour: Secuencia de nodos visitados
        num_nodos: Número total de nodos

    Returns:
        True si el tour es válido
    """
    if len(tour) != num_nodos + 1:
        return False

    if tour[0] != tour[-1]:
        return False

    nodos_unicos = set(tour[:-1])
    if len(nodos_unicos) != num_nodos:
        return False

    return True

def visualizar_tour(
        ciudades: np.ndarray,
        tour: List[int],
        titulo: str = "Tour ACO",
        guardar_como: str = None
):
    """
    Visualiza un tour en el plano 2D.
    Args:
        ciudades: Coordenadas de ciudades
        tour: Secuencia de nodos del tour
        titulo: Título del gráfico
        guardar_como: Ruta para guardar imagen (opcional)
    """
    plt.figure(figsize=(12, 8))

    # Dibujar camino
    for i in range(len(tour) - 1):
        ciudad_actual = ciudades[tour[i]]
        ciudad_siguiente = ciudades[tour[i+1]]
        plt.plot(
            [ciudad_actual[0], ciudad_siguiente[0]],
            [ciudad_actual[1], ciudad_siguiente[1]],
            'b-', linewidth=1.5, alpha=0.7
        )

    # Dibujar ciudades
    plt.scatter(
        ciudades[:, 0],
        ciudades[:, 1],
        c='red',
        s=200,
        zorder=5,
        edgecolors='black',
        linewidth=2
    )

    # Destacar ciudad de inicio
    plt.scatter(
        ciudades[tour[0], 0],
        ciudades[tour[0], 1],
        c='green',
        s=300,
        zorder=6,
        marker='*',
        edgecolors='black',
        linewidth=2,
        label='Inicio'
    )

    # Etiquetar ciudades
    for i, ciudad in enumerate(ciudades):
        plt.annotate(
            str(i),
            (ciudad[0], ciudad[1]),
            fontsize=10,
            ha='center',
            va='center',
            color='white',
            weight='bold'
        )

    plt.title(titulo, fontsize=16, weight='bold')
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')

    plt.show()

def visualizar_convergencia(
        historial_mejor: List[float],
        historial_promedio: List[float],
        titulo: str = "Convergencia ACO",
        guardar_como: str = None
):
    """
    Visualiza la convergencia del algoritmo ACO.
    Args:
        historial_mejor: Lista con mejor distancia por iteración
        historial_promedio: Lista con distancia promedio por iteración
        titulo: Título del gráfico
        guardar_como: Ruta para guardar imagen (opcional)
    """
    plt.figure(figsize=(12, 6))

    iteraciones = range(1, len(historial_mejor) + 1)

    # Graficar mejor solución
    plt.plot(
        iteraciones,
        historial_mejor,
        'g-',
        linewidth=2,
        label='Mejor solución global',
        marker='o',
        markersize=3
    )

    # Graficar promedio
    plt.plot(
        iteraciones,
        historial_promedio,
        'b--',
        linewidth=2,
        label='Distancia promedio',
        marker='s',
        markersize=3,
        alpha=0.7
    )

    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Distancia del tour', fontsize=12)
    plt.title(titulo, fontsize=16, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Añadir anotación de mejora
    mejora_porcentual = (
            (historial_promedio[0] - historial_mejor[-1]) / historial_promedio[0] * 100
    )
    plt.text(
        0.02, 0.98,
        f'Mejora: {mejora_porcentual:.1f}%',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')

    plt.show()


def visualizar_matriz_feromonas(
        feromonas: np.ndarray,
        titulo: str = "Matriz de Feromonas",
        guardar_como: str = None
):
    """
    Visualiza la distribución de feromonas como mapa de calor.
    Args:
        feromonas: Matriz de feromonas
        titulo: Título del gráfico
        guardar_como: Ruta para guardar imagen (opcional)
    """
    plt.figure(figsize=(10, 8))

    im = plt.imshow(feromonas, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, label='Intensidad de feromona')

    plt.title(titulo, fontsize=16, weight='bold')
    plt.xlabel('Nodo destino', fontsize=12)
    plt.ylabel('Nodo origen', fontsize=12)

    # Añadir grid
    ax = plt.gca()
    ax.set_xticks(np.arange(len(feromonas)))
    ax.set_yticks(np.arange(len(feromonas)))
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')

    plt.show()
