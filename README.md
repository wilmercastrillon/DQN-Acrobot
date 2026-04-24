# Acrobot

Proyecto práctico de un agente **Deep Q-Network (DQN)** en el entorno [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) de Gymnasium, con una implementación en PyTorch.

## Entorno Acrobot-v1

[Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) es un ambiente de un péndulo doble, es decir, dos pendulos simples unidos en serie (el segundo pendulo esta unido al final del primer pendulo). En este ambiente solo el nodo intermedio está accionado, el objetivo es balancear el extremo del segundo pendulo por encima de una altura fija, aplicando torque en ese nodo.

### Observaciones

El espacio de observación incluye el coseno y el seno de ambos ángulos de brazo y las dos velocidades angulares, obteniendo un estado continuo de 6 dimensiones.


| Índice | Variable | Descripción | Rango |
|:---:|---|---|---|
| 0 | cos(θ₁) | Coseno del ángulo del primer eslabón (base) | −1 a 1 |
| 1 | sin(θ₁) | Seno del ángulo del primer eslabón | −1 a 1 |
| 2 | cos(θ₂) | Coseno del ángulo del segundo eslabón (relativo) | −1 a 1 |
| 3 | sin(θ₂) | Seno del ángulo del segundo eslabón | −1 a 1 |
| 4 | θ̇₁ | Velocidad angular del enlace 1 | (−12.567 * -4π, 12.567 * 4π) |
| 5 | θ̇₂ | Velocidad angular del enlace 2 | (−28.274 * -9π, 28.274 * 9π) |

### Acciones

| Valor | Acción |
|:---:|---|
| 0 | Aplicar −1 de torque al nodo accionado |
| 1 | Aplicar 0 (sin torque) |
| 2 | Aplicar +1 de torque al nodo accionado |

### Recompensas y fin de episodio

| Evento | Recompensa |
|---|---|
| Cada paso mientras **no** se alcanza el objetivo | **−1** |
| Se alcanza la altura objetivo (éxito) | **0** (termina el episodio) |
| Tras **500** pasos sin alcanzar el objetivo | Truncado (termina el episodio) |

En Gymnasium, un entorno se considera **resuelto** con una recompensa media de **−100** en 100 episodios consecutivos.

> Ejecuta `acrobot inspect` para ver en vivo los espacios de observación/acción y algunas transiciones aleatorias con el entorno por defecto.

## Entrenamiento de Acrobot con DQN

El DQN aproxima la función de valor acción óptima Q∗(s,a) mediante una red neuronal entrenada con descenso de gradiente, el entrenamiento se basa en la ecuación de Bellman, donde el valor objetivo se calcula como la recompensa inmediata más el valor máximo estimado del siguiente estado usando una red objetivo (target network).

Para estabilizar el aprendizaje, se emplea un buffer de repetición de experiencias que almacena transiciones y permite obtenerlas aleatoriamente, además, se utiliza una red objetivo cuyos parámetros se actualizan periódicamente a partir de la red principal, evitando que los valores objetivo cambien demasiado rápido.

La política de exploración es ε-greedy sobre la red en línea: inicialmente se prioriza la exploración con valores altos de ε, y a medida que avanza el entrenamiento, ε disminuye para favorecer la explotación de la política aprendida.

A continuación se detalla el flujo tal como está implementado en el método `train` de `src/acrobot/agents/dqn.py`.

### Inicialización al llamar a `train`

1. Se crea el entorno de `Acrobot-v1` con `gym.make(...)`.
2. Se inicializa una lista `rewards_history` para registrar la recompensa acumulada de cada episodio.

### Bucle exterior: episodio a episodio

Para cada episodio (el limite es definido por `total_episodes`, por defecto es 10_000) se realizan las siguientes acciones:

3. Se reinicia el entorno con `env.reset()`, se ponen a cero el retorno acumulado del episodio, la bandera `done` y el contador de pasos `steps`.

### Bucle interior: pasos hasta fin de episodio o límite de pasos (por defecto 500)

Mientras el episodio no haya terminado y el número de pasos sea menor que el limite de pasos:

4. Con probabilidad `epsilon` se elige una acción al azar entre las tres disponibles, en caso contrario, se pasan las observaciones por la red y se toma el índice de la acción con **mayor** \(Q\).

5. Se da el siguiente paso con `env.step(action)` para obtener 4 valores: el siguiente estado, la recompensa, terminated y truncated, el ambiente no termina hasta que no se consiga la altura objetivo o se llegue al limite de intentos, la recompensa por cada paso es −1.

6. Se guarda la transición `(obs, acción, recompensa, next_obs, done)` en el buffer, se valida si el buffer tiene menos transiciones que `batch_size`, en ese caso se devuelve 0.0, si no, se obtiene una muestra aleatoria del buffer, despues se calcula el Q actual, el target (Bellman), el error y se actualizan los pesos.

7. Se obtiene la siguiente observacion, se acumula la recompensa en el total del episodio e incrementa `steps`, de esta forma en cada paso de interacción (una vez rellenado el buffer lo suficiente) se puede hacer una actualización de gradiente.

### Tras terminar un episodio (salir del `while`)

8. Se actualiza la exploración `epsilon = max(epsilon_end, epsilon * epsilon_decay)` al terminar cada episodio.

9. Si el número de episodio es múltiplo de 10 se copian los pesos de la red principal a la red objetivo.

Al agotar todos los episodios, se cierra el entorno y se devuelve la lista con la recompensa de cada episodio.

## Arquitectura de la red neuronal

MLP completamente conectado con arquitectura 6 → H → H → H → 3 (H = 512 por defecto), utilizando activaciones ReLU entre capas lineales, la red recibe el estado como entrada y produce un vector Q(s,⋅) con un valor por cada acción discreta (3 en este entorno).

Se emplea una red objetivo con la misma arquitectura, cuyos parámetros se actualizan periódicamente copiando los de la red principal, en lugar de ser entrenada directamente por gradiente, con el fin de estabilizar el aprendizaje.

## Resultados entrenamiento

El entrenamiento se realizo con 1000 episodios y de forma resumida se obtuvieron estos resultados:
```python
Episode   20/1000 | Avg Reward: -494.8 | Best: -396.0 | Epsilon: 0.9417 | Buffer: 9897
Episode   40/1000 | Avg Reward: -500.0 | Best: -500.0 | Epsilon: 0.8868 | Buffer: 19897
Episode   60/1000 | Avg Reward: -498.7 | Best: -474.0 | Epsilon: 0.8350 | Buffer: 29872
Episode   80/1000 | Avg Reward: -482.9 | Best: -355.0 | Epsilon: 0.7863 | Buffer: 39533
Episode  100/1000 | Avg Reward: -490.0 | Best: -383.0 | Epsilon: 0.7405 | Buffer: 49336
...
Episode  440/1000 | Avg Reward: -145.0 | Best:  -98.0 | Epsilon: 0.2666 | Buffer: 100000
Episode  460/1000 | Avg Reward: -150.1 | Best: -102.0 | Epsilon: 0.2511 | Buffer: 100000
Episode  480/1000 | Avg Reward: -159.3 | Best:  -92.0 | Epsilon: 0.2364 | Buffer: 100000
Episode  500/1000 | Avg Reward: -134.6 | Best:  -88.0 | Epsilon: 0.2226 | Buffer: 100000
Episode  520/1000 | Avg Reward: -130.7 | Best:  -92.0 | Epsilon: 0.2096 | Buffer: 100000
Episode  540/1000 | Avg Reward: -118.2 | Best:  -84.0 | Epsilon: 0.1974 | Buffer: 100000
...
Episode  900/1000 | Avg Reward:  -80.5 | Best:  -63.0 | Epsilon: 0.0669 | Buffer: 100000
Episode  920/1000 | Avg Reward:  -79.3 | Best:  -63.0 | Epsilon: 0.0630 | Buffer: 100000
Episode  940/1000 | Avg Reward:  -84.5 | Best:  -62.0 | Epsilon: 0.0594 | Buffer: 100000
Episode  960/1000 | Avg Reward:  -81.9 | Best:  -66.0 | Epsilon: 0.0559 | Buffer: 100000
Episode  980/1000 | Avg Reward:  -80.2 | Best:  -63.0 | Epsilon: 0.0526 | Buffer: 100000
Episode 1000/1000 | Avg Reward:  -76.3 | Best:  -63.0 | Epsilon: 0.0496 | Buffer: 100000
```

## Reflexión del ejercicio

### Reflexión de los resultados
El agente muestra un proceso de aprendizaje progresivo pasando de recompensas iniciales de -500 por truncamiento a valores de -76, lo que indica que logra resolver el entorno en menos de los 100 pasos requeridos, la mejora es progresiva hasta cerca del episodio 800 donde se presentan oscilaciones finales que reflejan cierta inestabilidad en la convergencia a partir de que el agente se aproxima a recompensas mas bajas.

### Retos del proyecto

Uno de los principales retos durante el desarrollo del proyecto fue lograr la estabilidad en el entrenamiento del agente con DQN, pues al tratarse de un ejercicio con un sistema caotico la elección de hiperparámetros debia ser la adecuada, encontrar valores como la tasa de aprendizaje, el factor de descuento, frecuencia de actualización, etc,  requirió múltiples experimentos y análisis de resultados.

## Configuración (setup)

```bash
uv sync
source .venv/bin/activate   # Linux y macOS
.venv\Scripts\activate      # Windows
```

## Uso de la CLI

El programa de entrada es **`acrobot`**. Todos los subcomandos trabajan sobre **Acrobot-v1** y el guardado DQN descrito. Resumen de subcomandos:

| Comando | Finalidad |
|---:|---|
| `version` | Muestra la versión del paquete |
| `inspect` | Muestra espacios y transiciones al azar (opcionales `--env`, `--steps`) |
| `init` | Crea un DQN **nuevo** con pesos iniciales aleatorios y lo guarda en `saves/dqn_Acrobot-v1.pt` (falla si el fichero ya existe; antes usa `delete`) |
| `train` | Entrenamiento: si hay guardado, carga; si no, inicia de cero, y luego sustituye el guardado |
| `load` | Carga el guardado, imprime `info` (con `--eval`, 10 episodios de prueba) |
| `sim` | Simulación en texto con un guardado entrenado (`--episodes`, y opcionales `--steps`, `--verbose`) |
| `render` | Ejecución con ventana gráfica (`--episodes`) |
| `delete` | Borra `saves/dqn_Acrobot-v1.pt` |

### Comando `version`

```bash
acrobot version
```

### Inspección del entorno

```bash
acrobot inspect                    # por defecto: Acrobot-v1, 5 pasos de muestra
acrobot inspect --steps 10
acrobot inspect --env Acrobot-v1
```

### Inicializar un agente nuevo (sin entrenar)

```bash
acrobot init
```

### Entrenar

Por defecto, **10_000** episodios; cámbialo con `--episodes`.

```bash
acrobot train
acrobot train --episodes 5000
```

### Cargar y (opcional) evaluar

```bash
acrobot load
acrobot load --eval
```

### Simulación en consola (requiere guardado entrenado)

```bash
acrobot sim
acrobot sim --episodes 3
acrobot sim --episodes 2 --verbose
acrobot sim --episodes 3 --steps 20
```

Con `--steps`, solo se imprimen con detalle los *primeros N* pasos de cada episodio; el episodio sigue hasta el final para el resumen.

### Renderizado con ventana (requiere guardado entrenado)

```bash
acrobot render
acrobot render --episodes 3
```

### Borrar el fichero de guardado

```bash
acrobot delete
```

## Estructura del proyecto

```
src/acrobot/
├── cli.py           # Punto de entrada de la CLI (comando `acrobot`)
└── agents/
    └── dqn.py      # DQN: QNetwork, buffer de repetición, bucle de entrenamiento, guardar/cargar
```

Los guardados se escriben en el directorio `saves/` bajo el directorio de trabajo actual (`dqn_Acrobot-v1.pt`).

## Requisitos

- Python 3.11 (el rango exacto figura en `pyproject.toml`)
- Dependencias principales: Gymnasium, NumPy, PyTorch — consulta `pyproject.toml` e instala con `uv sync`
