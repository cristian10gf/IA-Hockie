# Air Hockey RL - Juego de Hockey Aéreo con IA

Un juego de hockey aéreo donde puedes entrenar y jugar contra una IA que aprende usando Deep Reinforcement Learning (DQN).

![Preview del juego](preview.png)

## 🎮 Características

- Juego de hockey aéreo estilo arcade con físicas realistas
- IA entrenada mediante Deep Q-Learning
- Modo de entrenamiento para la IA
- Modo de juego contra la IA
- Interfaz neon retro
- Puntuaciones en tiempo real

## 🛠️ Requisitos

- Python 3.8+
- PyTorch
- Pygame
- Gymnasium
- NumPy
- tqdm

## 🚀 Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/cristian10gf/IA-hockie.git
cd IA-hockie
```

2. Instala las dependencias:

```bash
pip install torch pygame gymnasium numpy tqdm
```

## 📝 Cómo Usar

1. **Iniciar el juego:**

```bash
python v1-gemini.py
```

2. **Menú Principal:**

- `Train AI`: Entrena una nueva IA
- `Play against AI`: Juega contra la IA entrenada
- `Quit`: Salir del juego

## 🎯 Controles

### Menú

- `↑/↓`: Navegar opciones
- `Enter`: Seleccionar
- `Esc`: Volver/Salir

### Juego

- `Mouse`: Controlar paleta del jugador
- `P`: Pausar juego
- `R`: Reiniciar partida
- `Space`: Reanudar juego
- `Esc`: Volver al menú

## 🧠 Entrenamiento de la IA

La IA utiliza:

- Deep Q-Network (DQN)
- Experience Replay
- Target Network
- Epsilon-greedy exploration
- Reward shaping adaptativo

## 🌟 Características de la IA

- Aprendizaje progresivo
- Comportamiento defensivo y ofensivo
- Adaptación al estilo del oponente
- Predicción de trayectorias
- Toma de decisiones en tiempo real

## 🔧 Tecnologías

- PyTorch para Deep Learning
- Pygame para gráficos y físicas
- Gymnasium para el entorno de RL
- NumPy para cálculos numéricos

## 📈 Resultados

La IA mejora progresivamente:

- Aprende estrategias defensivas
- Desarrolla tácticas ofensivas
- Se adapta a diferentes situaciones de juego
- Mejora su precisión y timing

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Puedes:

- Reportar bugs
- Sugerir mejoras
- Enviar pull requests

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
