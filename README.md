# Optimización por Colonia de Hormigas (ACO)
Este proyecto implementa el algoritmo de **Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO)** aplicado al problema del **Vendedor Viajero (Traveling Salesman Problem - TSP)**.
##  Estructura del Proyecto

```
Traveling-Salesman-Problem/
│
├── hormiga.py                    # Clase Hormiga (agente individual)
├── colonia_hormigas.py           # Algoritmo ACO principal
├── utilidades.py                 # Funciones auxiliares y visualización
├── main.py                       # Programa principal con ejemplos
└── README.md                     # Este archivo
```

## Requisitos
### Librerías Python Necesarias:
```bash
pip install numpy matplotlib
```
### Versiones Recomendadas:
- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

---

## Uso del Programa

### **Ejecutar Todos los Ejemplos**

```bash
python main.py
```
Este comando ejecuta 3 ejemplos completos:
1. **Ejemplo Básico**: TSP con 15 ciudades
2. **Análisis de Parámetros**: Comparación de diferentes configuraciones (20 ciudades)
3. **Problema Grande**: TSP con 30 ciudades

##Parámetros del Algoritmo

### **Alpha (α)** - Influencia de Feromona
- **Rango típico**: [0.5, 2.0]
- **Efecto**: Controla cuánto peso se da a la memoria colectiva
  - α bajo → Más exploración
  - α alto → Más explotación

### **Beta (β)** - Influencia Heurística
- **Rango típico**: [2.0, 5.0]
- **Efecto**: Controla codicia del algoritmo
  - β bajo → Menos greedy
  - β alto → Más greedy (favorece aristas cortas)

### **Rho (ρ)** - Tasa de Evaporación
- **Rango**: [0, 1]
- **Efecto**: Controla "olvido" de información
  - ρ bajo → Memoria persistente
  - ρ alto → Olvido rápido, más exploración

### **Número de Hormigas (m)**
- **Rango típico**: [n, 2n] donde n = número de nodos
- **Efecto**: Más hormigas = mejor exploración pero más costo computacional

### **Iteraciones**
- **Rango típico**: [50, 300] para problemas medianos
- **Efecto**: Más iteraciones = mejor convergencia pero más tiempo
---
