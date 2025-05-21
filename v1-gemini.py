import concurrent
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
import os
import datetime
import time
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BORDER_THICKNESS = 10

# Colors (Neon Arcade Theme)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (230, 230, 230)
COLOR_NEON_BLUE = (0, 220, 255)        # Player 1 (Human or Rule-based)
COLOR_NEON_YELLOW = (255, 220, 0)      # Player 2 (AI Agent)
COLOR_NEON_PINK = (255, 0, 150)        # Puck
COLOR_NEON_GREEN = (50, 255, 50)       # Goal lines / Accents
COLOR_GRAY = (100, 100, 100)
COLOR_DARK_GRAY = (50,50,50)
COLOR_RED = (255, 50, 50) # For messages or warnings

# Game physics and properties
PADDLE_RADIUS = 20
PUCK_RADIUS = 10
PADDLE_SPEED = 7  # For AI discrete moves
HUMAN_PADDLE_SPEED_FACTOR = 1.0 # How closely human paddle follows mouse (1.0 = instant)
AI_PADDLE_SPEED_TRAINING_OPPONENT = 30 # Speed for the rule-based opponent during training
PUCK_MAX_SPEED = 15
DIAGONAL_SPEED_FACTOR = 0.7071 # sqrt(2)/2 for diagonal movement (45 degrees)
PUCK_FRICTION = 0.9999  # Velocity multiplier per frame (closer to 1 means less friction)
GOAL_WIDTH = BORDER_THICKNESS + 5 # Slightly wider than border
GOAL_HEIGHT = int(SCREEN_HEIGHT * 0.4)
GOAL_Y_START = (SCREEN_HEIGHT - GOAL_HEIGHT) // 2
GOAL_Y_END = GOAL_Y_START + GOAL_HEIGHT

CENTER_LINE_X = SCREEN_WIDTH // 2
AI_PLAYER_SIDE_MIN_X = CENTER_LINE_X + PADDLE_RADIUS
AI_PLAYER_SIDE_MAX_X = SCREEN_WIDTH - PADDLE_RADIUS - BORDER_THICKNESS
HUMAN_PLAYER_SIDE_MIN_X = BORDER_THICKNESS + PADDLE_RADIUS
HUMAN_PLAYER_SIDE_MAX_X = CENTER_LINE_X - PADDLE_RADIUS
PADDLE_SMOOTHING = 0.5 # Smoothing factor for paddle movement (0.0 = no smoothing, 1.0 = instant)


# RL Agent parameters
STATE_SIZE = 10  # Puck (x,y,vx,vy), AI_paddle (x,y), Opponent_paddle (x,y), score_ai, score_opponent
ACTION_SIZE = 9  # 0-3: Up,Down,Left,Right, 4-7: Diagonales, 8: Stay Still
LEARNING_RATE = 0.005 # Adjusted learning rate
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.999995 # Slower decay: 0.9995, Faster: 0.995
BATCH_SIZE = 128 # Increased batch size
MEMORY_SIZE = 50000 # Increased memory size
TARGET_UPDATE_FREQ = 20 # Update target network every 20 episodes
MIN_REPLAY_SIZE_TO_TRAIN = 1000 # Start training only after this many samples in memory
AI_VELOCITY = 4.5 # AI paddle velocity for training opponent

# Paths
MODEL_DIR = "trained_models"
LOG_DIR = "training_logs"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Pygame font
pygame.font.init()
FONT_SCORE = pygame.font.Font(None, 74)
FONT_MENU = pygame.font.Font(None, 50)
FONT_INFO = pygame.font.Font(None, 30)
FONT_SMALL_INFO = pygame.font.Font(None, 24)


# Device configuration for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Transition tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

REWARD_PROXIMITY = 0.1  # Recompensa por proximidad a la pelota
REWARD_ALIGNMENT = 0.03  # Recompensa por alineación con el objetivo
INACTION_PENALTY = -0.1  # Penalización por quedarse quieto
MAX_REPEATED_ACTIONS = 10  # Máximo de acciones iguales consecutivas permitidas

# --- Helper Functions ---
def normalize_state(state_array, screen_width, screen_height, max_speed):
    """Normalizes state values to be roughly between -1 and 1."""
    norm_state = np.zeros_like(state_array, dtype=float)
    # Normalizar posiciones y velocidades como antes
    norm_state[0] = (state_array[0] - screen_width / 2) / (screen_width / 2)  # Puck X
    norm_state[1] = (state_array[1] - screen_height / 2) / (screen_height / 2) # Puck Y
    norm_state[2] = state_array[2] / max_speed  # Puck Vx
    norm_state[3] = state_array[3] / max_speed  # Puck Vy
    norm_state[4] = (state_array[4] - (screen_width * 0.75)) / (screen_width * 0.25) # AI Paddle X
    norm_state[5] = (state_array[5] - screen_height / 2) / (screen_height / 2)     # AI Paddle Y
    norm_state[6] = (state_array[6] - (screen_width * 0.25)) / (screen_width * 0.25) # Opp Paddle X
    norm_state[7] = (state_array[7] - screen_height / 2) / (screen_height / 2)     # Opp Paddle Y
    # Normalizar puntajes (asumiendo máximo 5 puntos)
    norm_state[8] = state_array[8] / 5.0   # AI Score
    norm_state[9] = state_array[9] / 5.0   # Opponent Score
    return norm_state

def check_collision(obj1_pos, obj1_radius, obj2_pos, obj2_radius):
    """
    Checks collision between two circular objects using precise distance calculation.
    Args:
        obj1_pos: [x,y] position of first object
        obj1_radius: radius of first object
        obj2_pos: [x,y] position of second object
        obj2_radius: radius of second object
    Returns:
        bool: True if objects are colliding, False otherwise
    """
    # Calculate exact distance between centers using sqrt
    dx = obj1_pos[0] - obj2_pos[0]
    dy = obj1_pos[1] - obj2_pos[1]
    actual_distance = math.sqrt(dx*dx + dy*dy)

    # Compare with sum of radii for precise collision detection
    # Add small epsilon to handle floating point imprecision
    return actual_distance <= (obj1_radius + obj2_radius + 1e-10)

def resolve_paddle_puck_collision(paddle_pos, paddle_vel, puck_pos, puck_vel, paddle_radius, puck_radius, puck_max_speed):
    """Resolves collision between a paddle and the puck. Modifies puck_pos and returns new puck_vel."""
    dx = puck_pos[0] - paddle_pos[0]
    dy = puck_pos[1] - paddle_pos[1]
    distance = math.sqrt(dx**2 + dy**2)

    if distance == 0: distance = 0.001 # avoid division by zero

    # Normal vector from paddle center to puck center
    nx = dx / distance
    ny = dy / distance

    # Relative velocity
    rvx = puck_vel[0] - paddle_vel[0]
    rvy = puck_vel[1] - paddle_vel[1]

    # Velocity component along the normal
    vel_along_normal = rvx * nx + rvy * ny

    # Do not resolve if velocities are separating (puck moving away from paddle)
    if vel_along_normal > 0:
        return list(puck_vel) # No change

    # Elasticity (coefficient of restitution) - 1.0 for perfect elasticity
    e = 0.8 # Make it a bit less bouncy for more control

    # Calculate impulse scalar (simplified for equal masses, or paddle mass >> puck mass)
    j = -(1 + e) * vel_along_normal
    # More accurate would be: j = -(1 + e) * vel_along_normal / (1/m_puck + 1/m_paddle_effective)
    # For simplicity, assume paddle is very heavy or fixed in collision frame

    # Apply impulse to puck's velocity
    new_puck_vx = puck_vel[0] + j * nx
    new_puck_vy = puck_vel[1] + j * ny
    
    # Add a bit of paddle's own velocity component to the puck
    # This gives the player more "control" to direct the puck
    new_puck_vx += paddle_vel[0] * 0.2 
    new_puck_vy += paddle_vel[1] * 0.2

    # Cap puck speed
    speed = math.sqrt(new_puck_vx**2 + new_puck_vy**2)
    if speed > puck_max_speed:
        new_puck_vx = (new_puck_vx / speed) * puck_max_speed
        new_puck_vy = (new_puck_vy / speed) * puck_max_speed
        
    # Ensure puck is moved out of collision to prevent sticking
    overlap = paddle_radius + puck_radius - distance
    if overlap > 0:
        puck_pos[0] += nx * (overlap + 0.1) # Move slightly more than overlap
        puck_pos[1] += ny * (overlap + 0.1)

    return [new_puck_vx, new_puck_vy]


# --- Reinforcement Learning Environment (Gymnasium) ---
class AirHockeyEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60} # Increased FPS

    def __init__(self, render_mode=None, training_opponent_level=2, human_opponent=False):
        super(AirHockeyEnv, self).__init__()

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.paddle_radius = PADDLE_RADIUS
        self.puck_radius = PUCK_RADIUS
        self.puck_max_speed = PUCK_MAX_SPEED
        self.puck_friction = PUCK_FRICTION
        self.ai_paddle_speed = PADDLE_SPEED
        self.training_opponent_level = training_opponent_level
        self.human_opponent = human_opponent # If true, opponent paddle is controlled by mouse in play_game

        self.action_space = spaces.Discrete(ACTION_SIZE)
        low = np.array([-1.0] * STATE_SIZE, dtype=np.float32)
        high = np.array([1.0] * STATE_SIZE, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.max_episode_steps = 1500 # Increased max steps
        self.current_step = 0
        self.score_ai = 0
        self.score_opponent = 0

        self.ai_last_pos = [self.screen_width * 0.75, self.screen_height / 2]  # Para suavizado
        self.score_surface = None  # Para mantener superficies de puntaje
        self.last_scores = None  # Para verificar cambios en puntajes

        self.last_action = None
        self.repeated_action_count = 0
        self._reset_positions()

    def _reset_positions(self, new_episode=True):
        # AI Paddle (Yellow) siempre a la derecha
        self.ai_paddle_pos = [self.screen_width * 0.75, self.screen_height / 2]
        # Opponent/Human Paddle (Blue) siempre a la izquierda
        self.opponent_paddle_pos = [self.screen_width * 0.25, self.screen_height / 2]
        
        # Randomize puck start slightly
        puck_start_x = self.screen_width / 2 + random.uniform(-30, 30)
        puck_start_y = self.screen_height / 2 + random.uniform(-30, 30)
        self.puck_pos = [puck_start_x, puck_start_y]
        
        # Start puck moving towards a random player, or stationary
        rand_choice = random.random()
        if rand_choice < 0.4: # Towards AI
            self.puck_vel = [random.uniform(1, 3), random.uniform(-2, 2)]
        elif rand_choice < 0.8: # Towards Opponent
            self.puck_vel = [random.uniform(-3, -1), random.uniform(-2, 2)]
        else: # Stationary or slow
            self.puck_vel = [random.uniform(-1,1), random.uniform(-1,1)]
        
        self.ai_paddle_vel = [0,0] 
        self.opponent_paddle_vel = [0,0]

        if new_episode: # Reset scores only for a new episode, not just a round reset
            self.current_step = 0
            # Scores are reset outside if it's a full game reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions(new_episode=True) # Full reset for a new episode
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def _get_obs(self):
        state_array = np.array([
            self.puck_pos[0], self.puck_pos[1], self.puck_vel[0], self.puck_vel[1],
            self.ai_paddle_pos[0], self.ai_paddle_pos[1],
            self.opponent_paddle_pos[0], self.opponent_paddle_pos[1],
            float(self.score_ai), float(self.score_opponent)  # Agregar puntajes al estado
        ], dtype=np.float32)
        return normalize_state(state_array, self.screen_width, self.screen_height, self.puck_max_speed)

    def _get_info(self):
        return {
            "puck_pos": list(self.puck_pos), 
            "ai_paddle_pos": list(self.ai_paddle_pos), 
            "opponent_paddle_pos": list(self.opponent_paddle_pos),
            "score_ai": self.score_ai,
            "score_opponent": self.score_opponent
        }

    def _move_training_opponent(self):
        """Enhanced rule-based opponent with improved speed, precision and agility."""
        opponent_speed = AI_PADDLE_SPEED_TRAINING_OPPONENT * 2  # Base speed increased
        
        # Better position prediction with acceleration consideration
        predicted_y = self.puck_pos[1]
        time_to_intercept = 0
        
        if abs(self.puck_vel[0]) > 0.1:
            # Consider acceleration in prediction
            time_to_intercept = (self.opponent_paddle_pos[0] - self.puck_pos[0]) / (self.puck_vel[0] + 0.1)
            predicted_y = self.puck_pos[1] + (self.puck_vel[1] * time_to_intercept) + (0.5 * self.puck_friction * time_to_intercept ** 2)

        # Dynamic positioning based on game state
        puck_speed = math.sqrt(self.puck_vel[0]**2 + self.puck_vel[1]**2)
        goal_center_y = (GOAL_Y_START + GOAL_Y_END) / 2
        
        # Aggressive offensive behavior
        if self.puck_pos[0] < CENTER_LINE_X:
            if puck_speed < 2:  # Slow moving or stationary puck
                # Direct approach for quick shot
                target_x = self.puck_pos[0] - self.paddle_radius * 0.6
                target_y = self.puck_pos[1] + (goal_center_y - self.puck_pos[1]) * 0.7
                opponent_speed *= 1.5  # Faster approach for opportunity
            else:
                # Intercept with angle calculation
                angle_to_goal = math.atan2(goal_center_y - self.puck_pos[1], self.screen_width - self.puck_pos[0])
                target_x = self.puck_pos[0] - self.paddle_radius * 0.8
                target_y = self.puck_pos[1] + math.sin(angle_to_goal) * self.paddle_radius * 2
                opponent_speed *= 1.3
        else:
            # Enhanced defensive positioning
            if abs(self.puck_vel[0]) > 5:  # Fast moving puck
                # Predictive defense
                target_x = self.screen_width * 0.25
                target_y = predicted_y
                opponent_speed *= 1.8  # Faster defensive response
            else:
                # Dynamic defense position
                center_offset = (self.puck_pos[1] - goal_center_y) * 0.6
                target_x = self.screen_width * 0.3
                target_y = goal_center_y + center_offset
                opponent_speed *= 1.4

        # Difficulty level adjustments
        if self.training_opponent_level == 2:
            opponent_speed *= 1.5
            # More precise prediction and positioning
            if time_to_intercept > 0:
                predicted_y = self.puck_pos[1] + (self.puck_vel[1] * time_to_intercept * 1.2)
                if abs(self.puck_vel[0]) > 3:
                    target_y = predicted_y
                    opponent_speed *= 1.3

        # Store previous position for velocity calculation
        prev_x = self.opponent_paddle_pos[0]
        prev_y = self.opponent_paddle_pos[1]
        
        # Smooth movement with acceleration
        dx = target_x - self.opponent_paddle_pos[0]
        dy = target_y - self.opponent_paddle_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            # Normalized direction with speed adjustment
            move_x = (dx / distance) * opponent_speed
            move_y = (dy / distance) * opponent_speed
            
            # Apply movement with smoothing
            self.opponent_paddle_pos[0] += move_x
            self.opponent_paddle_pos[1] += move_y

        # Enforce boundaries with smoother clamping
        self.opponent_paddle_pos[0] = np.clip(self.opponent_paddle_pos[0],
                                            HUMAN_PLAYER_SIDE_MIN_X,
                                            HUMAN_PLAYER_SIDE_MAX_X)
        self.opponent_paddle_pos[1] = np.clip(self.opponent_paddle_pos[1],
                                            self.paddle_radius + BORDER_THICKNESS,
                                            self.screen_height - self.paddle_radius - BORDER_THICKNESS)
        
        # Update paddle velocity for collision physics
        self.opponent_paddle_vel = [
            self.opponent_paddle_pos[0] - prev_x,
            self.opponent_paddle_pos[1] - prev_y
        ]
        # Actualizar velocidad del paddle para colisiones más realistas
        self.opponent_paddle_vel = [
            self.opponent_paddle_pos[0] - prev_x,
            self.opponent_paddle_pos[1] - prev_y
        ]

    def update_human_opponent_paddle(self, mouse_pos):
        """Updates human opponent paddle based on mouse position."""
        if mouse_pos is None: return

        target_x, target_y = mouse_pos
        
        # Smooth movement (optional, can be direct)
        # self.opponent_paddle_pos[0] += (target_x - self.opponent_paddle_pos[0]) * HUMAN_PADDLE_SPEED_FACTOR
        # self.opponent_paddle_pos[1] += (target_y - self.opponent_paddle_pos[1]) * HUMAN_PADDLE_SPEED_FACTOR
        
        prev_opp_pos = list(self.opponent_paddle_pos)

        self.opponent_paddle_pos[0] = target_x
        self.opponent_paddle_pos[1] = target_y

        # Keep human paddle within its half and bounds
        self.opponent_paddle_pos[0] = np.clip(self.opponent_paddle_pos[0],
                                              HUMAN_PLAYER_SIDE_MIN_X,
                                              HUMAN_PLAYER_SIDE_MAX_X)
        self.opponent_paddle_pos[1] = np.clip(self.opponent_paddle_pos[1],
                                              self.paddle_radius + BORDER_THICKNESS,
                                              self.screen_height - self.paddle_radius - BORDER_THICKNESS)
        self.opponent_paddle_vel = [self.opponent_paddle_pos[0] - prev_opp_pos[0], self.opponent_paddle_pos[1] - prev_opp_pos[1]]


    def step(self, action, human_mouse_pos=None): # Added human_mouse_pos for play mode
        self.current_step += 1
        reward = 0.0 # Use float for rewards
        terminated = False 
        truncated = False 

        prev_ai_paddle_pos = list(self.ai_paddle_pos)
        puck_hit_by_ai_this_step = False

        # --- 1. AI Agent Action ---
        dx, dy = 0, 0
        if action < 4:  # Movimientos cardinales
            if action == 0: dy = -self.ai_paddle_speed*AI_VELOCITY   # Up
            elif action == 1: dy = self.ai_paddle_speed*AI_VELOCITY  # Down
            elif action == 2: dx = -self.ai_paddle_speed*AI_VELOCITY  # Left
            elif action == 3: dx = self.ai_paddle_speed*AI_VELOCITY  # Right
        elif action < 8:  # Movimientos diagonales
            if action == 4:  # Up-Left
                dx = -self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
                dy = -self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
            elif action == 5:  # Up-Right
                dx = self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
                dy = -self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
            elif action == 6:  # Down-Left
                dx = -self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
                dy = self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
            elif action == 7:  # Down-Right
                dx = self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
                dy = self.ai_paddle_speed * DIAGONAL_SPEED_FACTOR
        # action == 8 es Stay Still (dx = dy = 0)

        # Modificar la actualización de la posición de la IA para suavizar el movimiento
        target_x = self.ai_paddle_pos[0] + dx
        target_y = self.ai_paddle_pos[1] + dy

        # Aplicar suavizado al movimiento
        self.ai_paddle_pos[0] = self.ai_last_pos[0] + (target_x - self.ai_last_pos[0]) * PADDLE_SMOOTHING
        self.ai_paddle_pos[1] = self.ai_last_pos[1] + (target_y - self.ai_last_pos[1]) * PADDLE_SMOOTHING
        
        # Guardar la posición actual como última posición para el siguiente frame
        self.ai_last_pos = list(self.ai_paddle_pos)

        # Aplicar límites después del suavizado
        self.ai_paddle_pos[0] = np.clip(self.ai_paddle_pos[0], AI_PLAYER_SIDE_MIN_X, AI_PLAYER_SIDE_MAX_X)
        self.ai_paddle_pos[1] = np.clip(self.ai_paddle_pos[1],
                                        self.paddle_radius + BORDER_THICKNESS,
                                        self.screen_height - self.paddle_radius - BORDER_THICKNESS)
        self.ai_paddle_vel = [self.ai_paddle_pos[0] - prev_ai_paddle_pos[0], self.ai_paddle_pos[1] - prev_ai_paddle_pos[1]]

        # --- 2. Opponent Action ---
        if self.human_opponent:
            self.update_human_opponent_paddle(human_mouse_pos)
        else: # Training opponent
            self._move_training_opponent()

        # --- 3. Puck Movement & Physics ---
        self.puck_pos[0] += self.puck_vel[0]
        self.puck_pos[1] += self.puck_vel[1]
        self.puck_vel[0] *= self.puck_friction
        self.puck_vel[1] *= self.puck_friction
        
        # Limit puck speed if it somehow exceeds max (e.g. due to multiple small friction steps)
        current_puck_speed = math.sqrt(self.puck_vel[0]**2 + self.puck_vel[1]**2)
        if current_puck_speed > self.puck_max_speed:
            self.puck_vel[0] = (self.puck_vel[0] / current_puck_speed) * self.puck_max_speed
            self.puck_vel[1] = (self.puck_vel[1] / current_puck_speed) * self.puck_max_speed


        # Puck wall collisions
        if self.puck_pos[1] - self.puck_radius <= BORDER_THICKNESS:
            self.puck_vel[1] *= -1
            self.puck_pos[1] = BORDER_THICKNESS + self.puck_radius
            reward -= 0.01 # Small penalty for hitting side walls
        elif self.puck_pos[1] + self.puck_radius >= self.screen_height - BORDER_THICKNESS:
            self.puck_vel[1] *= -1
            self.puck_pos[1] = self.screen_height - BORDER_THICKNESS - self.puck_radius
            reward -= 0.01

        # Puck goal collisions / scoring
        goal_scored_this_step = False
        
        # AI scores (Puck hits left goal)
        if self.puck_pos[0] - self.puck_radius <= BORDER_THICKNESS + GOAL_WIDTH:
            if GOAL_Y_START < self.puck_pos[1] < GOAL_Y_END:
                if self.human_opponent:  
                    reward = 50.0  # La IA anota en modo juego
                    self.score_ai += 1  # Punto para la IA
                else:  
                    reward = 50.0  # La IA anota en modo entrenamiento
                    self.score_ai += 1
                terminated = True
                goal_scored_this_step = True
                self._reset_puck_to_center()  # Reset puck después del gol
                pygame.time.wait(300)  # Pequeña pausa para mostrar el gol
            else:
                self.puck_vel[0] *= -1
                self.puck_pos[0] = BORDER_THICKNESS + GOAL_WIDTH + self.puck_radius
                reward -= 0.5

        # Opponent/Human scores (Puck hits right goal)
        elif self.puck_pos[0] + self.puck_radius >= self.screen_width - BORDER_THICKNESS - GOAL_WIDTH:
            if GOAL_Y_START < self.puck_pos[1] < GOAL_Y_END:
                if self.human_opponent:
                    reward = -50.0  # La IA pierde (modo juego)
                    self.score_opponent += 1  # Punto para el humano
                else:
                    reward = -50.0  # La IA pierde (modo entrenamiento)
                    self.score_opponent += 1
                terminated = True
                goal_scored_this_step = True
                self._reset_puck_to_center()  # Reset puck después del gol
                pygame.time.wait(300)  # Pequeña pausa para mostrar el gol
            else:
                self.puck_vel[0] *= -1
                self.puck_pos[0] = self.screen_width - BORDER_THICKNESS - GOAL_WIDTH - self.puck_radius
                reward -= 0.5

        # Puck-paddle collisions
        if not goal_scored_this_step: # Only check paddle collisions if no goal was scored
            puck_vel_before_ai_hit = list(self.puck_vel) # Store puck velocity before potential AI hit
            if check_collision(self.ai_paddle_pos, self.paddle_radius, self.puck_pos, self.puck_radius):
                self.puck_vel = resolve_paddle_puck_collision(self.ai_paddle_pos, self.ai_paddle_vel, self.puck_pos, self.puck_vel, self.paddle_radius, self.puck_radius, self.puck_max_speed)
                reward += 1.0 # Base reward for hitting puck
                puck_hit_by_ai_this_step = True
                
                # Reward for hitting puck towards opponent goal (puck moving left)
                if self.puck_vel[0] < -1.0: # Moving significantly left
                    reward += 10.0 
                    # Bonus if it's a "shot on goal" (aligned with goal Y)
                    if GOAL_Y_START < self.puck_pos[1] < GOAL_Y_END:
                        reward += 5.0 
                
                # Penalty for hitting puck towards own goal (puck moving right)
                # Only penalize if it wasn't a defensive block of a puck already coming towards AI goal
                if self.puck_vel[0] > 1.0 and self.puck_pos[0] > CENTER_LINE_X + self.screen_width * 0.1:
                    if puck_vel_before_ai_hit[0] > 0.5: # Puck was already moving towards AI goal
                        reward += 15.0 # Successful block reward
                    else: # AI actively hit it towards its own goal
                        reward -= 10.0 # Stronger penalty for this mistake

            if check_collision(self.opponent_paddle_pos, self.paddle_radius, self.puck_pos, self.puck_radius):
                self.puck_vel = resolve_paddle_puck_collision(self.opponent_paddle_pos, self.opponent_paddle_vel, self.puck_pos, self.puck_vel, self.paddle_radius, self.puck_radius, self.puck_max_speed)
                # If AI just hit it and opponent immediately returns, maybe small penalty or neutral
                if puck_hit_by_ai_this_step:
                    reward -= 0.5 # AI's good hit was immediately countered


        # --- 4. Additional Rewards/Penalties (if no goal scored yet) ---
        if not goal_scored_this_step:
            # Penalización progresiva por distancia al puck
            dist_to_puck = math.sqrt((self.ai_paddle_pos[0]-self.puck_pos[0])**2 + 
                                   (self.ai_paddle_pos[1]-self.puck_pos[1])**2)
            
            # Penalización base por distancia
            max_allowed_distance = self.screen_width * 0.8  # Distancia máxima permitida
            if dist_to_puck > max_allowed_distance and self.puck_pos[0] < CENTER_LINE_X:
                # Penalización exponencial por distancia excesiva
                excess_distance = dist_to_puck - max_allowed_distance
                distance_penalty = -(0.4 * (excess_distance / 50.0))
                reward += distance_penalty

            # Penalización extra cuando el puck está en el lado de la IA
            if self.puck_pos[0] > CENTER_LINE_X:
                if dist_to_puck > self.paddle_radius * 2:
                    # Penalización más fuerte cuando el puck está en nuestro lado
                    defensive_penalty = -(0.2 * (dist_to_puck / 100.0))
                    reward += defensive_penalty
                else:
                    # Recompensa por estar cerca del puck en defensa
                    reward += 0.5

            # Resto de las recompensas/penalizaciones
            if self.puck_pos[0] < CENTER_LINE_X and puck_hit_by_ai_this_step:
                reward += 15.0
            
            # Penalty for puck in own defensive zone (near goal) - continuous
            if self.puck_pos[0] > self.screen_width * 0.80 and \
               GOAL_Y_START - self.paddle_radius < self.puck_pos[1] < GOAL_Y_END + self.paddle_radius:
                reward -= 0.2 # Small continuous penalty for puck danger

            # Penalty for AI paddle being too far from its goal when puck is on AI's side and moving towards goal
            if self.puck_pos[0] > CENTER_LINE_X and self.puck_vel[0] > 0.1: # Puck on AI side, moving towards AI goal
                dist_to_center_goal_y = abs(self.ai_paddle_pos[1] - self.screen_height/2)
                if dist_to_center_goal_y > self.screen_height * 0.25: # If paddle is far from center of Y
                    reward -= 0.05 * (dist_to_center_goal_y / (self.screen_height * 0.25)) # Scaled penalty

            # Small reward for keeping puck on opponent's side
            if self.puck_pos[0] < CENTER_LINE_X:
                reward += 4.0
            
            # Small reward for AI paddle being close to the puck when puck is on AI's side
            if self.puck_pos[0] > CENTER_LINE_X:
                dist_to_puck = math.sqrt((self.ai_paddle_pos[0]-self.puck_pos[0])**2 + (self.ai_paddle_pos[1]-self.puck_pos[1])**2)
                if dist_to_puck < self.paddle_radius * 3 : # If close
                    reward += 0.02
                else: # If far from puck on its own side
                    reward -= 0.01


        if self.current_step >= self.max_episode_steps and not goal_scored_this_step:
            terminated = True # End episode if too long
            reward -= 10.0 # Penalty for not finishing (increased)
            # print(f"Episode timed out. Steps: {self.current_step}")


        observation = self._get_obs()
        
        if self.render_mode == "human":
            self._render_frame()
            
        # In training, a round ends when a goal is scored or max steps.
        # In play_game mode, terminated will reset the puck position but continue the game.
        return observation, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_array=True)
        # Human rendering is handled in step or by calling _render_frame directly

    def _render_frame(self, to_array=False, scores=None): # scores is a tuple (opp_score, ai_score)
        if self.screen is None and self.render_mode == "human":
            pygame.init() # Ensure pygame is initialized if not already
            pygame.display.set_caption("Air Hockey Neon Arena")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(COLOR_BLACK)

        # Draw borders
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (0, 0, self.screen_width, BORDER_THICKNESS)) # Top
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (0, self.screen_height - BORDER_THICKNESS, self.screen_width, BORDER_THICKNESS)) # Bottom
        
        # Left wall (excluding goal)
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (0, BORDER_THICKNESS, BORDER_THICKNESS + GOAL_WIDTH, GOAL_Y_START - BORDER_THICKNESS))
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (0, GOAL_Y_END, BORDER_THICKNESS + GOAL_WIDTH, self.screen_height - GOAL_Y_END - BORDER_THICKNESS))
        
        # Right wall (excluding goal)
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (self.screen_width - BORDER_THICKNESS - GOAL_WIDTH, BORDER_THICKNESS, BORDER_THICKNESS + GOAL_WIDTH, GOAL_Y_START - BORDER_THICKNESS))
        pygame.draw.rect(canvas, COLOR_DARK_GRAY, (self.screen_width - BORDER_THICKNESS - GOAL_WIDTH, GOAL_Y_END, BORDER_THICKNESS + GOAL_WIDTH, self.screen_height - GOAL_Y_END - BORDER_THICKNESS))


        # Draw goals (actual scoring area)
        pygame.draw.rect(canvas, COLOR_NEON_GREEN, (BORDER_THICKNESS, GOAL_Y_START, GOAL_WIDTH, GOAL_HEIGHT)) # Left goal
        pygame.draw.rect(canvas, COLOR_NEON_GREEN, (self.screen_width - BORDER_THICKNESS - GOAL_WIDTH, GOAL_Y_START, GOAL_WIDTH, GOAL_HEIGHT)) # Right goal    
        
        # Draw center line and circle
        pygame.draw.line(canvas, COLOR_WHITE, (CENTER_LINE_X, BORDER_THICKNESS), (CENTER_LINE_X, self.screen_height - BORDER_THICKNESS), 2)
        pygame.draw.circle(canvas, COLOR_WHITE, (CENTER_LINE_X, self.screen_height // 2), 60, 2) # Center circle
        pygame.draw.circle(canvas, COLOR_WHITE, (CENTER_LINE_X, self.screen_height // 2), 5, 0) # Center dot


        # Draw paddles
        pygame.draw.circle(canvas, COLOR_NEON_BLUE, self.opponent_paddle_pos, self.paddle_radius) # Opponent (Human/Rule-based)
        pygame.draw.circle(canvas, COLOR_NEON_YELLOW, self.ai_paddle_pos, self.paddle_radius)    # AI Agent
        
        # Draw puck
        pygame.draw.circle(canvas, COLOR_NEON_PINK, self.puck_pos, self.puck_radius)

        # Modificar la parte del renderizado de puntajes para evitar parpadeo
        if scores is not None:
            current_scores = scores
            if self.last_scores != current_scores or self.score_surface is None:
                # Solo recrear las superficies de texto si los puntajes cambiaron
                
                opp_score_text = FONT_SCORE.render(str(scores[0]), True, COLOR_NEON_BLUE)
                ai_score_text = FONT_SCORE.render(str(scores[1]), True, COLOR_NEON_YELLOW)
                
                # Crear una nueva superficie combinada para los puntajes
                self.score_surface = pygame.Surface((self.screen_width, 50), pygame.SRCALPHA)
                self.score_surface.blit(opp_score_text, 
                    (self.screen_width * 0.25 - opp_score_text.get_width() // 2, 0))
                self.score_surface.blit(ai_score_text, 
                    (self.screen_width * 0.75 - ai_score_text.get_width() // 2, 0))
                self.last_scores = current_scores

            # Usar la superficie pre-renderizada
            if self.score_surface is not None:
                canvas.blit(self.score_surface, (0, 20))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump() 
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            return None
        elif to_array: 
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            # pygame.quit() # Delay full quit to main, to avoid issues if multiple envs are made
            self.screen = None
            self.clock = None

    def _reset_puck_to_center(self):
        """Reset puck position to center with random velocity"""
        self.puck_pos = [self.screen_width / 2, self.screen_height / 2]
        
        # Velocidad aleatoria hacia el último perdedor
        if self.score_ai > self.score_opponent:
            # Hacia el oponente
            self.puck_vel = [random.uniform(-3, -1), random.uniform(-2, 2)]
        else:
            # Hacia la IA
            self.puck_vel = [random.uniform(1, 3), random.uniform(-2, 2)]

# --- RL Agent (DQN) ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256) # Increased neurons
        self.layer2 = nn.Linear(256, 256)            # Increased neurons
        self.layer3 = nn.Linear(256, 256)            # Increased neurons
        self.layer4 = nn.Linear(256, 256)            # Increased neurons
        self.layer5 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=LEARNING_RATE, gamma=GAMMA, 
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY,
                 memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Añadir el optimizador aquí
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        
        self.steps_done = 0 # For epsilon decay based on steps, or can be tied to episodes
        self.prediction_queue = Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.last_action = 8  # Acción por defecto (stay still)

    def predict_action_async(self, state):
        """Predicts action asynchronously"""
        def predict():
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].view(1, 1).item()
        
        return self.thread_pool.submit(predict)

    def select_action(self, state, evaluation_mode=False, async_mode=False):
        """Selects an action using epsilon-greedy policy or greedy for evaluation."""
        self.steps_done += 1
        
        if not evaluation_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if evaluation_mode or random.random() > self.epsilon:
            if async_mode:
                # Retornar la última acción mientras se procesa la nueva
                future = self.predict_action_async(state)
                try:
                    # Intentar obtener el resultado con un timeout más generoso
                    action = future.result(timeout=0.05)  # Aumentado de 0.001 a 0.05 segundos (50ms)
                    self.last_action = action
                    return action
                except (TimeoutError, concurrent.futures._base.TimeoutError):
                    # Capturar explícitamente ambos tipos de TimeoutError
                    # y asegurarse de retornar la última acción
                    return self.last_action
            else:
                # Modo síncrono normal para entrenamiento
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    return q_values.max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_dim)
        
    def store_transition(self, state, action, next_state, reward, done):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE)
        action_tensor = torch.tensor([[action]], dtype=torch.long, device=DEVICE)
        
        # Handle next_state if it's None (e.g. at the end of an episode)
        if next_state is None:
            next_state_tensor = None
        else:
            next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float32, device=DEVICE)
            
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
        done_tensor = torch.tensor([done], dtype=torch.bool, device=DEVICE) # Store as boolean
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)

    def optimize_model(self):
        if len(self.memory) < self.batch_size or len(self.memory) < MIN_REPLAY_SIZE_TO_TRAIN:
            return None # Not enough samples yet

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
        
        # Filter out None next_states before stacking
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) == 0: # All next states in batch are None (all episodes ended)
             # If all next_states are None, then Q values for next states are all 0.
            next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        else:
            non_final_next_states = torch.stack(non_final_next_states_list)
            next_state_values = torch.zeros(self.batch_size, device=DEVICE)
            # Compute Q values for non-final next states using target_net
            # .detach() is used because we don't want to backpropagate through the target network
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()


        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute the expected Q values: R + gamma * max_a' Q_target(s', a')
        # For final states, the expected Q value is just the reward.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss (smooth L1 loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Cipping (helps stabilize training for some complex environments)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item() # Return the loss value for logging

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename="dqn_air_hockey_model.pth"):
        path = os.path.join(MODEL_DIR, filename)
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, filename="dqn_air_hockey_model.pth"):
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
            self.policy_net.eval() # Set to evaluation mode after loading
            self.target_net.eval()
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"No model found at {path}, starting from scratch.")
            return False


# --- Training Loop ---
def train_agent(num_episodes=1000, training_opponent_level=2, model_save_name_prefix="air_hockey_dqn"):
    env = AirHockeyEnv(render_mode=None, training_opponent_level=training_opponent_level)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    # Log file setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")
    
    episode_rewards = []
    episode_losses = []
    epsilon_values = []
    ai_scores_log = []
    opponent_scores_log = []

    start_time = time.time()
    
    print(f"Starting training for {num_episodes} episodes. Opponent level: {training_opponent_level}")
    print(f"Logging to {log_filename}")

    # Crear barra de progreso
    pbar = tqdm(range(1, num_episodes + 1), desc="Training Progress")
    
    for i_episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        current_episode_losses = []
        
        # Reset scores
        env.score_ai = 0
        env.score_opponent = 0

        for t in range(env.max_episode_steps): # Max steps per episode
            action = agent.select_action(state)
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
            
            episode_reward += reward

            # Store experience in replay memory
            # If terminated (goal scored or timeout), next_state might be from a new puck drop
            # For DQN, if 'done' is true, the value of next_state doesn't matter as much since its Q-value contribution will be zeroed out.
            # However, it's good practice to pass the actual next state if available, or None if truly terminal.
            # Here, 'terminated' means a goal is scored or max steps, so the game resets for the next logical 'step' (new round)
            # We can consider the "next_state_raw" as the s' for this transition.
            # If a goal is scored, the "done" for RL learning step is true.
            
            agent.store_transition(state, action, next_state_raw, reward, done) # Use 'done' which is True if terminated
            
            state = next_state_raw

            loss = agent.optimize_model()
            if loss is not None:
                current_episode_losses.append(loss)

            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(current_episode_losses) if current_episode_losses else 0
        episode_losses.append(avg_loss)
        epsilon_values.append(agent.epsilon)
        ai_scores_log.append(info['score_ai']) # env.score_ai
        opponent_scores_log.append(info['score_opponent']) # env.score_opponent

        if i_episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            #print(f"Target network updated at episode {i_episode}")

        if i_episode % 20 == 0 or i_episode == num_episodes:
            avg_reward_last_20 = np.mean(episode_rewards[-20:])
            avg_loss_last_20 = np.mean(episode_losses[-20:])
            avg_ai_score_last_20 = np.mean(ai_scores_log[-20:])
            avg_opp_score_last_20 = np.mean(opponent_scores_log[-20:])
            
            # Actualizar descripción de la barra de progreso
            pbar.set_description(
                f"Reward: {avg_reward_last_20:.2f} | "
                f"Loss: {avg_loss_last_20:.4f} | "
                f"AI Score: {avg_ai_score_last_20:.1f} | "
                f"Opp Score: {avg_opp_score_last_20:.1f}"
            )

            # Guardar en el log
            log_message = (
                f"Episode {i_episode}/{num_episodes} | "
                f"Avg Reward (last 20): {avg_reward_last_20:.2f} | "
                f"Avg Loss (last 20): {avg_loss_last_20:.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Avg AI Score: {avg_ai_score_last_20:.2f} | "
                f"Avg Opp Score: {avg_opp_score_last_20:.2f} | "
                f"Total Steps Done: {agent.steps_done}"
            )
            with open(log_filename, "a") as f:
                f.write(log_message + "\n")

    # Save the final model with a timestamp
    final_model_name = f"{model_save_name_prefix}_{timestamp}.pth"
    agent.save_model(final_model_name)
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")
    print(f"Final model saved as: {final_model_name}")
    with open(log_filename, "a") as f:
        f.write(f"\nTraining completed in {training_duration:.2f} seconds.\n")
        f.write(f"Final model saved as: {final_model_name}\n")
    
    env.close()
    #Plotting results (optional, requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.subplot(2,2,2)
    plt.plot(episode_losses)
    plt.title('Average Episode Loss')
    plt.subplot(2,2,3)
    plt.plot(epsilon_values)
    plt.title('Epsilon Decay')
    plt.subplot(2,2,4)
    plt.plot(ai_scores_log, label='AI Score')
    plt.plot(opponent_scores_log, label='Opponent Score')
    plt.title('Scores per Episode')
    plt.legend()
    plt.tight_layout()
    plot_filename = os.path.join(LOG_DIR, f"training_plot_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Training plots saved to {plot_filename}")
    plt.show()

    return final_model_name


# --- Game Mode (Play against AI) ---
# Control keys
KEY_PAUSE = pygame.K_p
KEY_RESET = pygame.K_r
KEY_RESUME = pygame.K_SPACE

def play_game_with_ai(model_filename=None, winning_score=5):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Air Hockey - You (Blue) vs AI (Yellow)")
    clock = pygame.time.Clock()

    # Create environment for playing, human opponent is True
    env = AirHockeyEnv(render_mode="human", human_opponent=True) # human_opponent=True makes opponent paddle mouse controlled
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    if model_filename:
        if not agent.load_model(model_filename):
            print("Could not load model. AI will play randomly.")
            # Fallback: AI plays randomly if model not found, or you can exit.
    else:
        print("No model specified for AI. AI will play randomly.")

    running = True
    game_over_message = None
    paused = False
    
    env.score_ai = 0
    env.score_opponent = 0 # Human player's score (playing as opponent to the AI)
    
    state, _ = env.reset() # Initial reset of puck and paddle positions

    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == KEY_PAUSE and not game_over_message:
                    paused = not paused
                elif event.key == KEY_RESET and not game_over_message:
                    # Reset game state
                    env.score_ai = 0
                    env.score_opponent = 0
                    state, _ = env.reset()
                    print("Game Reset!")
                elif event.key == KEY_RESUME and paused:
                    paused = False
                elif game_over_message and event.key == pygame.K_r:
                    env.score_ai = 0
                    env.score_opponent = 0
                    state, _ = env.reset()
                    game_over_message = None

        if not game_over_message and not paused:
            # AI's turn to select action - ahora en modo asíncrono
            ai_action = agent.select_action(state, evaluation_mode=True, async_mode=True)

            # El resto del código sigue igual
            next_state, reward, terminated, truncated, info = env.step(ai_action, human_mouse_pos=mouse_pos)
            
            state = next_state

        # Rendering con el orden correcto para las puntuaciones
        env._render_frame(scores=(env.score_opponent, env.score_ai))

        # Draw pause/controls message
        if paused:
            pause_surface = FONT_MENU.render("PAUSED", True, COLOR_RED)
            pause_rect = pause_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(pause_surface, pause_rect)
            
            resume_surface = FONT_INFO.render("Press SPACE to Resume", True, COLOR_WHITE)
            resume_rect = resume_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            screen.blit(resume_surface, resume_rect)
        else:
            # Show controls hint
            controls_surface = FONT_SMALL_INFO.render("P: Pause | R: Reset | ESC: Menu", True, COLOR_GRAY)
            controls_rect = controls_surface.get_rect(bottomright=(SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10))
            screen.blit(controls_surface, controls_rect)

        if game_over_message:
            msg_surface = FONT_MENU.render(game_over_message, True, COLOR_RED)
            msg_rect = msg_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(msg_surface, msg_rect)

            restart_surface = FONT_INFO.render("Press 'R' to Restart Game, ESC to Main Menu", True, COLOR_WHITE)
            restart_rect = restart_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            screen.blit(restart_surface, restart_rect)
            pygame.display.flip() # Ensure game over message is shown

        clock.tick(60) # Cap FPS for gameplay

    env.close()
    # pygame.quit() # Moved to main after loop finishes

# --- Main Menu ---
def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Air Hockey - Main Menu")
    clock = pygame.time.Clock()

    selected_option = 0
    menu_options = ["Train AI", "Play against AI", "Quit"]
    
    # Find the latest trained model
    latest_model_file = None
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
        if model_files:
            latest_model_file = max([os.path.join(MODEL_DIR, f) for f in model_files], key=os.path.getctime)
            latest_model_file = os.path.basename(latest_model_file) # Just the filename for display/use


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(menu_options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(menu_options)
                elif event.key == pygame.K_RETURN:
                    if menu_options[selected_option] == "Train AI":
                        num_episodes_str = get_user_input(screen, "Enter number of training episodes (e.g., 1000):")
                        try:
                            num_episodes = int(num_episodes_str)
                            if num_episodes <= 0: raise ValueError
                            # pygame.display.quit() # Temporarily quit display for console output
                            train_agent(num_episodes=num_episodes)
                            # Re-initialize for menu if needed, or menu handles it
                            latest_model_file = get_latest_model() # Update latest model after training
                        except ValueError:
                            print("Invalid number of episodes. Please enter a positive integer.")
                            # Show error on screen briefly
                            error_surf = FONT_INFO.render("Invalid input. Enter a positive number.", True, COLOR_RED)
                            screen.blit(error_surf, (SCREEN_WIDTH //2 - error_surf.get_width()//2, SCREEN_HEIGHT - 70))
                            pygame.display.flip()
                            pygame.time.wait(2000)


                    elif menu_options[selected_option] == "Play against AI":
                        if latest_model_file:
                            print(f"Starting game with AI model: {latest_model_file}")
                            play_game_with_ai(model_filename=latest_model_file)
                            # After game, pygame might be quit by play_game, re-init if necessary for menu
                            pygame.init() # Re-initialize pygame for the menu
                            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # Re-set mode
                            pygame.display.set_caption("Air Hockey - Main Menu")

                        else:
                            print("No trained model found. Train an AI first or AI will play randomly.")
                            # Display message on screen
                            no_model_surf = FONT_INFO.render("No trained AI model found. Train first or AI plays randomly.", True, COLOR_RED)
                            no_model_rect = no_model_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 70))
                            screen.blit(no_model_surf, no_model_rect)
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            # Option to proceed with random AI
                            play_game_with_ai(model_filename=None)
                            pygame.init() 
                            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                            pygame.display.set_caption("Air Hockey - Main Menu")


                    elif menu_options[selected_option] == "Quit":
                        running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Find latest model again in loop in case training happened
        latest_model_file = get_latest_model()


        screen.fill(COLOR_BLACK)
        title_text = FONT_MENU.render("Neon Air Hockey RL", True, COLOR_NEON_GREEN)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        screen.blit(title_text, title_rect)

        for i, option in enumerate(menu_options):
            color = COLOR_NEON_YELLOW if i == selected_option else COLOR_WHITE
            text_surface = FONT_MENU.render(option, True, color)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 60))
            screen.blit(text_surface, text_rect)
            
            if option == "Play against AI" and latest_model_file:
                 model_info_text = FONT_SMALL_INFO.render(f"(Using: {latest_model_file[:20]}...)", True, COLOR_GRAY)
                 model_info_rect = model_info_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 60 + 30))
                 screen.blit(model_info_text, model_info_rect)
            elif option == "Play against AI" and not latest_model_file:
                 model_info_text = FONT_SMALL_INFO.render("(No model found, AI random)", True, COLOR_RED)
                 model_info_rect = model_info_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 60 + 30))
                 screen.blit(model_info_text, model_info_rect)


        pygame.display.flip()
        clock.tick(15) # Menu doesn't need high FPS

    pygame.quit()


def get_user_input(screen, prompt):
    """Helper to get text input from user in Pygame."""
    input_text = ""
    input_active = True
    prompt_surface = FONT_INFO.render(prompt, True, COLOR_WHITE)
    prompt_rect = prompt_surface.get_rect(midtop=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
    
    input_box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2, 300, 40)

    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isdigit(): # Only allow digits
                    input_text += event.unicode
        
        screen.fill(COLOR_BLACK) # Redraw background
        screen.blit(prompt_surface, prompt_rect)
        
        pygame.draw.rect(screen, COLOR_GRAY, input_box_rect, 2) # Input box border
        text_surface = FONT_INFO.render(input_text, True, COLOR_WHITE)
        # Position text inside the box
        screen.blit(text_surface, (input_box_rect.x + 5, input_box_rect.y + 5))
        
        pygame.display.flip()
        pygame.time.Clock().tick(30)
        
    return input_text

def get_latest_model():
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
        if model_files:
            latest_model_path = max([os.path.join(MODEL_DIR, f) for f in model_files], key=os.path.getctime)
            return os.path.basename(latest_model_path)
    return None

if __name__ == '__main__':
    main_menu()