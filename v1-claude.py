import os
import sys
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import time
import datetime
import math

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (144, 238, 144)
LIGHT_RED = (255, 182, 193)

# Game parameters
FRICTION = 0.999  # 99% of original velocity (reduced friction as specified)
PADDLE_RADIUS = 40
PUCK_RADIUS = 25
MAX_VELOCITY = 15
AI_MOVE_SPEED = 8
PLAYER_MOVE_SPEED = 8

# AI training parameters
BATCH_SIZE = 128  # Increased from 64 for more efficient learning
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000  # Reduced for faster exploration decay
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001

# Progress tracking frequency
PRINT_FREQUENCY = 50  # Print progress every 50 episodes
SAVE_FREQUENCY = 500  # Save model every 500 episodes

# Reward structure for reinforcement learning
GOAL_REWARD = 10.0  # Reward for scoring a goal
CONCEDE_PENALTY = -10.0  # Penalty for conceding a goal
MOVING_TOWARD_PUCK_REWARD = 0.05  # Small reward for moving toward the puck
HITTING_PUCK_REWARD = 1.0  # Reward for hitting the puck
DEFENSIVE_POSITION_REWARD = 0.1  # Reward for being in a good defensive position
BOUNCE_OFF_WALL_PENALTY = -0.2  # Small penalty for puck bouncing off walls

# Paths for saving models
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Create a named tuple for storing experiences
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save an experience"""
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Vector2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Paddle:
    def __init__(self, x, y, color, x_min, x_max, y_min, y_max):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0, 0)
        self.color = color
        self.radius = PADDLE_RADIUS
        # Movement constraints
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def move(self, dx, dy, speed):
        target_x = self.position.x + dx * speed
        target_y = self.position.y + dy * speed
        # Clamp to bounds
        self.position.x = max(self.x_min + self.radius, min(target_x, self.x_max - self.radius))
        self.position.y = max(self.y_min + self.radius, min(target_y, self.y_max - self.radius))
    
    def draw(self, screen):
        # Draw outer circle
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.radius)
        # Draw inner circle (lighter shade)
        inner_color = LIGHT_RED if self.color == RED else (LIGHT_GREEN if self.color == GREEN else LIGHT_BLUE)
        pygame.draw.circle(screen, inner_color, (int(self.position.x), int(self.position.y)), self.radius//2)

class Puck:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0, 0)
        self.radius = PUCK_RADIUS
    
    def update(self):
        # Apply friction
        self.velocity = self.velocity * FRICTION
        
        # Stop if velocity is very small
        if self.velocity.length() < 0.1:
            self.velocity = Vector2(0, 0)
        
        # Update position
        self.position = self.position + self.velocity
    
    def check_collision_with_walls(self, screen_width, screen_height, goal_width):
        collision = False
        
        # Left wall (not goal)
        if self.position.x - self.radius < 0:
            if self.position.y < (SCREEN_HEIGHT / 2 - goal_width / 2) or \
               self.position.y > (SCREEN_HEIGHT / 2 + goal_width / 2):
                self.position.x = self.radius
                self.velocity.x = -self.velocity.x
                collision = True
        
        # Right wall (not goal)
        if self.position.x + self.radius > screen_width:
            if self.position.y < (SCREEN_HEIGHT / 2 - goal_width / 2) or \
               self.position.y > (SCREEN_HEIGHT / 2 + goal_width / 2):
                self.position.x = screen_width - self.radius
                self.velocity.x = -self.velocity.x
                collision = True
        
        # Top and bottom walls
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.velocity.y = -self.velocity.y
            collision = True
        elif self.position.y + self.radius > screen_height:
            self.position.y = screen_height - self.radius
            self.velocity.y = -self.velocity.y
            collision = True
        
        return collision
    
    def check_goal(self, screen_width, screen_height, goal_width):
        # Left goal (Player 1's goal)
        if self.position.x - self.radius < 0:
            if (self.position.y > (SCREEN_HEIGHT / 2 - goal_width / 2) and 
                self.position.y < (SCREEN_HEIGHT / 2 + goal_width / 2)):
                return 2  # Player 2 scores
        
        # Right goal (Player 2's goal)
        if self.position.x + self.radius > screen_width:
            if (self.position.y > (SCREEN_HEIGHT / 2 - goal_width / 2) and 
                self.position.y < (SCREEN_HEIGHT / 2 + goal_width / 2)):
                return 1  # Player 1 scores
        
        return 0  # No goal
    
    def check_collision_with_paddle(self, paddle):
        distance = self.position.distance_to(paddle.position)
        
        if distance < self.radius + paddle.radius:
            # Calculate direction vector from paddle to puck
            direction = Vector2(
                self.position.x - paddle.position.x,
                self.position.y - paddle.position.y
            ).normalize()
            
            # Calculate reflection vector
            reflection_speed = min(MAX_VELOCITY, max(3, self.velocity.length() + 2))
            self.velocity = direction * reflection_speed
            
            # Move puck outside of paddle to prevent sticking
            overlap = self.radius + paddle.radius - distance
            self.position = self.position + direction * overlap
            
            return True
        
        return False
    
    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.position.x), int(self.position.y)), self.radius)

class AirHockeyEnv:
    def __init__(self, render_mode=None):
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.goal_width = 100
        self.center_circle_radius = 80
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Field boundaries
        self.field_width = self.screen_width
        self.field_height = self.screen_height
        self.center_x = self.field_width // 2
        self.half_width = self.field_width // 2
        
        # Initialize paddles and puck
        self.reset()
        
        # State space: [player_x, player_y, opponent_x, opponent_y, puck_x, puck_y, puck_vx, puck_vy]
        self.state_size = 8
        # Action space: [up, down, left, right, stay]
        self.action_size = 5
        
        # Max steps to prevent infinite episodes
        self.max_steps_per_episode = 2000
        self.current_steps = 0
        
        if self.render_mode == 'human':
            self._init_pygame()
    
    def _init_pygame(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Air Hockey - Advanced Physics")
        self.clock = pygame.time.Clock()
    
    def reset(self):
        # Create paddles
        self.player1 = Paddle(
            self.screen_width * 0.25, self.screen_height // 2, RED,
            0, self.center_x, 0, self.screen_height
        )
        self.player2 = Paddle(
            self.screen_width * 0.75, self.screen_height // 2, GREEN,
            self.center_x, self.screen_width, 0, self.screen_height
        )
        
        # Create puck at center
        self.puck = Puck(self.screen_width // 2, self.screen_height // 2)
        
        # Reset scores
        self.score_p1 = 0
        self.score_p2 = 0
        
        # Last distance from AI to puck (for reward calculation)
        self.last_distance = self.player2.position.distance_to(self.puck.position)
        
        # Last distance from puck to opponent's goal (for reward calculation)
        self.last_puck_to_goal_distance = abs(self.puck.position.x)
        
        # Reset done flag and step counter
        self.done = False
        self.current_steps = 0
        
        # Return initial state
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            self.player1.position.x / self.screen_width,
            self.player1.position.y / self.screen_height,
            self.player2.position.x / self.screen_width,
            self.player2.position.y / self.screen_height,
            self.puck.position.x / self.screen_width,
            self.puck.position.y / self.screen_height,
            self.puck.velocity.x / MAX_VELOCITY,
            self.puck.velocity.y / MAX_VELOCITY
        ], dtype=np.float32)
    
    def step(self, action, player_move=None):
        # Increment step counter
        self.current_steps += 1
        
        # Process AI action
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -1
        elif action == 1:  # down
            dy = 1
        elif action == 2:  # left
            dx = -1
        elif action == 3:  # right
            dx = 1
        # action 4 is stay (dx=0, dy=0)
        
        # Move AI paddle (player2)
        self.player2.move(dx, dy, AI_MOVE_SPEED)
        
        # Move player paddle if in play mode
        if player_move is not None:
            self.player1.move(player_move[0], player_move[1], PLAYER_MOVE_SPEED)
        else:
            # Rule-based opponent during training
            self._rule_based_opponent_move()
        
        # Update puck physics
        self.puck.update()
        
        # Check for collisions with paddles
        p1_hit = self.puck.check_collision_with_paddle(self.player1)
        p2_hit = self.puck.check_collision_with_paddle(self.player2)
        
        # Check for wall collisions
        wall_collision = self.puck.check_collision_with_walls(self.screen_width, self.screen_height, self.goal_width)
        
        # Check for goals
        goal = self.puck.check_goal(self.screen_width, self.screen_height, self.goal_width)
        if goal == 1:
            self.score_p1 += 1
            self._reset_positions()
        elif goal == 2:
            self.score_p2 += 1
            self._reset_positions()
        
        # Calculate reward for AI
        reward = self._calculate_reward(p2_hit, goal, wall_collision)
        
        # Update last distances for next reward calculation
        self.last_distance = self.player2.position.distance_to(self.puck.position)
        self.last_puck_to_goal_distance = abs(self.puck.position.x)
        
        # Get new state
        new_state = self._get_state()
        
        # Check if done (for training episodes)
        if self.score_p1 >= 5 or self.score_p2 >= 5 or self.current_steps >= self.max_steps_per_episode:
            self.done = True
        
        return new_state, reward, self.done, {"score": (self.score_p1, self.score_p2)}
    
    def _reset_positions(self):
        """Reset positions after a goal"""
        self.puck.position = Vector2(self.screen_width // 2, self.screen_height // 2)
        self.puck.velocity = Vector2(0, 0)
        # Random initial velocity for more varied gameplay
        if random.random() > 0.5:
            self.puck.velocity.x = random.uniform(-2, 2)
            self.puck.velocity.y = random.choice([-2, 2]) * random.random()
    
    def _rule_based_opponent_move(self):
        """Simple rule-based movement for the training opponent"""
        # If puck is moving towards opponent, try to intercept
        if self.puck.velocity.x < 0:
            target_y = self.puck.position.y + (self.puck.velocity.y * 
                      (self.player1.position.x - self.puck.position.x) / max(0.1, -self.puck.velocity.x))
            target_y = max(self.player1.radius, min(self.screen_height - self.player1.radius, target_y))
        else:
            # Otherwise, move back towards center with some defensive positioning
            target_y = self.screen_height // 2
            if self.puck.position.y < self.screen_height // 2:
                target_y -= 50
            else:
                target_y += 50
        
        # Move towards target
        if abs(self.player1.position.y - target_y) > 5:
            dy = 1 if target_y > self.player1.position.y else -1
            self.player1.move(0, dy, PLAYER_MOVE_SPEED * 0.7)  # Move slightly slower than AI for fairness
        
        # Move towards puck horizontally if it's on our side
        if self.puck.position.x < self.center_x and self.puck.position.x > self.player1.position.x:
            self.player1.move(1, 0, PLAYER_MOVE_SPEED * 0.7)
        elif self.puck.position.x < self.player1.position.x:
            self.player1.move(-1, 0, PLAYER_MOVE_SPEED * 0.7)
    
    def _calculate_reward(self, hit_puck, goal, wall_collision):
        """Calculate reward for the AI agent"""
        reward = 0
        
        # Major rewards/penalties for goals
        if goal == 2:  # AI scored
            reward += GOAL_REWARD
        elif goal == 1:  # AI conceded
            reward += CONCEDE_PENALTY
        
        # Reward for hitting the puck
        if hit_puck:
            reward += HITTING_PUCK_REWARD
            
            # Extra reward if hitting the puck towards opponent's goal
            if self.puck.velocity.x < 0:
                reward += 0.5
        
        # Small reward for moving toward the puck
        current_distance = self.player2.position.distance_to(self.puck.position)
        if current_distance < self.last_distance:
            reward += MOVING_TOWARD_PUCK_REWARD
        
        # Defensive positioning reward
        if self.puck.position.x > self.center_x:  # Puck on AI side
            # Good defensive position is between puck and goal
            goal_pos = Vector2(self.screen_width, self.screen_height / 2)
            puck_to_goal = goal_pos.distance_to(self.puck.position)
            paddle_to_goal = goal_pos.distance_to(self.player2.position)
            
            if paddle_to_goal < puck_to_goal:
                reward += DEFENSIVE_POSITION_REWARD
        
        # Small penalty for wall collisions
        if wall_collision:
            reward += BOUNCE_OFF_WALL_PENALTY
            
        return reward
    
    def render(self):
        if self.render_mode != 'human':
            return
        
        # Asegurarse de que pygame esté inicializado
        if not pygame.get_init():
            pygame.init()
            self._init_pygame()
            
        self.screen.fill(BLACK)
        
        # Draw center line
        pygame.draw.line(self.screen, WHITE, (self.screen_width // 2, 0), 
                         (self.screen_width // 2, self.screen_height), 2)
        
        # Draw center circle
        pygame.draw.circle(self.screen, WHITE, (self.screen_width // 2, self.screen_height // 2), 
                           self.center_circle_radius, 2)
        
        # Draw goals
        goal_height = self.goal_width
        goal_y_start = self.screen_height // 2 - goal_height // 2
        
        # Left goal (red)
        pygame.draw.rect(self.screen, RED, (0, goal_y_start, 5, goal_height))
        
        # Right goal (green)
        pygame.draw.rect(self.screen, GREEN, (self.screen_width - 5, goal_y_start, 5, goal_height))
        
        # Draw paddles
        self.player1.draw(self.screen)
        self.player2.draw(self.screen)
        
        # Draw puck
        self.puck.draw(self.screen)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{self.score_p1} : {self.score_p2}", True, WHITE)
        self.screen.blit(score_text, (self.screen_width // 2 - 30, 10))
        
        # Display FPS and Reset instruction
        info_font = pygame.font.Font(None, 24)
        fps_text = info_font.render(f"F: FPS | R: Reset", True, WHITE)
        self.screen.blit(fps_text, (10, self.screen_height - 30))
        
        # Display mode
        mode_text = info_font.render(f"Mode: Simple AI", True, WHITE)
        self.screen.blit(mode_text, (self.screen_width // 2 - 60, self.screen_height - 30))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        if self.screen is not None:
            # Solo cerramos la ventana, sin cerrar pygame completamente
            pygame.display.quit()
            self.screen = None

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Parameters
        self.steps_done = 0
        self.eps_threshold = EPS_START
    
    def select_action(self, state, training=True):
        if training:
            # Decrease epsilon over time
            self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1
            
            # Epsilon-greedy action selection
            if random.random() > self.eps_threshold:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    return self.policy_net(state_tensor).max(1)[1].item()
            else:
                return random.randrange(self.action_size)
        else:
            # During gameplay, always choose the best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].item()
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample experiences
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*transitions))
        
        # Create batch tensors - more efficiently
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        
        # More efficient tensor creation by first converting to numpy array
        next_states = [s for s in batch.next_state if s is not None]
        if next_states:
            non_final_next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        else:
            non_final_next_states = torch.FloatTensor([]).to(self.device)
            
        # Use numpy arrays first for better performance
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        
        # Compute Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        if len(non_final_next_states) > 0:  # Check if there are any non-final states
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Gradient clipping
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, iteration):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_DIR, f"air_hockey_model_{timestamp}_{iteration}.pth")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.eps_threshold
        }, model_path)
        
        # Also save as latest model
        latest_path = os.path.join(MODELS_DIR, "latest_model.pth")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.eps_threshold
        }, latest_path)
        
        return model_path
    
    def load_model(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "latest_model.pth")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.eps_threshold = checkpoint['epsilon']
            return True
        return False

def train_model(num_episodes):
    # Create environment and agent
    env = AirHockeyEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Try to load the latest model if available
    model_loaded = agent.load_model()
    if model_loaded:
        print("Loaded existing model. Continuing training...")
    else:
        print("No existing model found. Starting fresh training...")
    
    # Training metrics
    episode_rewards = []
    losses = []
    win_rate = []
    
    # Time tracking
    start_time = time.time()
    last_time = start_time
    
    # Log file setup
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Start training
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Training on device: {agent.device}")
    print(f"Memory size: {MEMORY_SIZE}, Batch size: {BATCH_SIZE}")
    print(f"Initial epsilon: {agent.eps_threshold:.4f}")
    print(f"Max steps per episode: {env.max_steps_per_episode}")
    print("-" * 50)
    
    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.datetime.now()}\n")
        log.write(f"Episodes: {num_episodes}, Memory size: {MEMORY_SIZE}, Batch size: {BATCH_SIZE}\n")
        log.write(f"Device: {agent.device}\n")
        log.write(f"Max steps per episode: {env.max_steps_per_episode}\n")
        log.write("-" * 50 + "\n")
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_loss = 0
            hits = 0  # Track puck hits
            
            # Debug status for first episode
            if episode == 0:
                print(f"Starting episode 1, state shape: {np.shape(state)}")
                
            # Track time for stuck detection
            episode_start_time = time.time()
            last_update_time = episode_start_time
            
            while not env.done:
                # Periodically print status to indicate progress
                current_time = time.time()
                if current_time - last_update_time > 30:  # Print every 30 seconds if stuck
                    print(f"Episode {episode+1} still running... Step: {steps}, Score: {env.score_p1}-{env.score_p2}")
                    last_update_time = current_time
                
                # Safety check - if episode takes too long, print debug info and break
                if current_time - episode_start_time > 300:  # 5 minutes max per episode
                    print(f"WARNING: Episode {episode+1} appears to be stuck!")
                    print(f"Current steps: {steps}, Max steps: {env.max_steps_per_episode}")
                    print(f"Current score: {env.score_p1}-{env.score_p2}")
                    print(f"Puck position: ({env.puck.position.x:.1f}, {env.puck.position.y:.1f})")
                    print(f"Puck velocity: ({env.puck.velocity.x:.1f}, {env.puck.velocity.y:.1f})")
                    break
                    
                # Select and perform an action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store the transition in memory
                agent.memory.push(state, action, next_state if not done else None, reward, done)
                
                # Move to the next state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Track hits with the puck
                if reward > 0.5:  # Assuming hitting the puck gives reward > 0.5
                    hits += 1
                
                # Perform one step of the optimization
                loss = agent.learn()
                if loss is not None:
                    episode_loss += loss
                
                # Update the target network periodically
                if steps % TARGET_UPDATE == 0:
                    agent.update_target_network()
                    
                # Progress indicator for first few episodes
                if episode < 3 and steps % 500 == 0:
                    print(f"Episode {episode+1} - Step {steps}, Score: {env.score_p1}-{env.score_p2}")
            
            # End of episode stats
            avg_loss = episode_loss / steps if steps > 0 else 0
            episode_rewards.append(total_reward)
            losses.append(avg_loss)
            
            # Simple progress indicator for every episode
            episode_time = time.time() - episode_start_time
            print(f"Episode {episode+1} completed in {episode_time:.1f}s, Steps: {steps}, Score: {env.score_p2}:{env.score_p1}")
            
            # Calculate win rate (AI scoring more than opponent)
            ai_win = 1 if env.score_p2 > env.score_p1 else 0
            win_rate.append(ai_win)
            recent_win_rate = sum(win_rate[-100:]) / min(len(win_rate), 100)
            
            # Detailed logging every few episodes
            if (episode + 1) % PRINT_FREQUENCY == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                episodes_per_sec = PRINT_FREQUENCY / elapsed if elapsed > 0 else 0
                total_elapsed = current_time - start_time
                avg_reward = sum(episode_rewards[-PRINT_FREQUENCY:]) / PRINT_FREQUENCY
                avg_loss = sum(losses[-PRINT_FREQUENCY:]) / PRINT_FREQUENCY
                
                # Memory stats
                memory_usage = len(agent.memory) / MEMORY_SIZE * 100
                
                # Print detailed status
                status_msg = (
                    f"\n--- Episode {episode + 1}/{num_episodes} ---\n"
                    f"Time: {total_elapsed:.1f}s | Speed: {episodes_per_sec:.2f} eps/s\n"
                    f"Score: {env.score_p2}:{env.score_p1} | Puck hits: {hits}\n"
                    f"Steps: {steps} | Reward: {total_reward:.2f} | Avg reward: {avg_reward:.2f}\n"
                    f"Loss: {avg_loss:.4f} | Epsilon: {agent.eps_threshold:.4f}\n"
                    f"Win rate (last 100): {recent_win_rate:.2%}\n"
                    f"Memory: {len(agent.memory)}/{MEMORY_SIZE} ({memory_usage:.1f}%)\n"
                )
                
                print(status_msg)
                log.write(status_msg + "\n")
                log.flush()  # Ensure logs are written even if program crashes
                
                last_time = current_time
            
            # Save model periodically
            if (episode + 1) % SAVE_FREQUENCY == 0 or episode == num_episodes - 1:
                model_path = agent.save_model(episode + 1)
                save_msg = f"Model saved to {model_path} (Episode {episode + 1})"
                print(save_msg)
                log.write(save_msg + "\n")
                log.flush()
        
        # Final stats
        total_time = time.time() - start_time
        final_msg = (
            f"\n==== Training Complete ====\n"
            f"Total time: {total_time:.2f} seconds\n"
            f"Average episodes per second: {num_episodes / total_time:.2f}\n"
            f"Final epsilon: {agent.eps_threshold:.4f}\n"
            f"Win rate (last 100): {recent_win_rate:.2%}\n"
            f"Average reward (last 100): {sum(episode_rewards[-100:]) / min(len(episode_rewards), 100):.2f}\n"
        )
        print(final_msg)
        log.write(final_msg)
    
    print(f"Training log saved to {log_file}")
    return agent

def play_game():
    # Load the trained model
    env = AirHockeyEnv(render_mode='human')
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Check if model exists
    if not agent.load_model():
        print("No trained model found. Please train the model first.")
        return
    
    print("Loaded trained model. Starting game...")
    
    # Game loop
    running = True
    state = env.reset()
    clock = pygame.time.Clock()
    
    # Display info about controls
    print("Game Controls:")
    print("- Move paddle with mouse")
    print("- Press R to reset the game")
    print("- Close window to exit")
    print("\nFirst to 5 points wins!")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset
                    state = env.reset()
                    print("Game reset!")
                elif event.key == pygame.K_f:  # Toggle FPS display
                    pass  # Placeholder for FPS toggle
                elif event.key == pygame.K_ESCAPE:  # Exit
                    running = False
        
        # Get mouse position for player movement
        mouse_pos = pygame.mouse.get_pos()
        
        # Calculate movement direction for player paddle
        # Use direct position control for smoother player movement
        target_x = max(env.player1.x_min + env.player1.radius, 
                      min(mouse_pos[0], env.player1.x_max - env.player1.radius))
        target_y = max(env.player1.y_min + env.player1.radius, 
                      min(mouse_pos[1], env.player1.y_max - env.player1.radius))
        
        # Move toward mouse position
        dx = 1 if target_x > env.player1.position.x else (-1 if target_x < env.player1.position.x else 0)
        dy = 1 if target_y > env.player1.position.y else (-1 if target_y < env.player1.position.y else 0)
        
        # Get AI action
        action = agent.select_action(state, training=False)
        
        # Step environment
        state, _, done, info = env.step(action, player_move=(dx, dy))
        
        # Display score changes
        if env.score_p1 > 0 or env.score_p2 > 0:
            score_text = f"Score: {env.score_p1} - {env.score_p2}"
            if not hasattr(play_game, 'last_score') or play_game.last_score != score_text:
                print(score_text)
                play_game.last_score = score_text
        
        # Render
        env.render()
        
        # Cap framerate
        clock.tick(FPS)
        
        if done:
            print(f"Game over! Final score: {env.score_p1} - {env.score_p2}")
            # Wait briefly before resetting
            pygame.time.wait(2000)
            state = env.reset()
            play_game.last_score = "0 - 0"
    
    # En lugar de cerrar pygame completamente, solo cerramos la ventana y liberamos recursos
    env.close()

def display_menu():
    # Initialize pygame if needed
    if not pygame.get_init():
        pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Air Hockey - Reinforcement Learning")
    clock = pygame.time.Clock()
    
    # Font setup
    title_font = pygame.font.Font(None, 64)
    menu_font = pygame.font.Font(None, 36)
    
    # Menu options
    options = [
        "Train Model",
        "Play Game",
        "Exit"
    ]
    selected = 0
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if options[selected] == "Train Model":
                        # Get number of episodes
                        num_episodes = get_episode_input(screen, clock)
                        if num_episodes > 0:
                            pygame.quit()  # Close menu
                            train_model(num_episodes)
                            return display_menu()  # Reopen menu after training
                    elif options[selected] == "Play Game":
                        pygame.quit()  # Close menu
                        play_game()
                        return display_menu()  # Reopen menu after game
                    elif options[selected] == "Exit":
                        running = False
        
        # Draw menu
        screen.fill(BLACK)
        
        # Title
        title_text = title_font.render("Air Hockey RL", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        # Options
        for i, option in enumerate(options):
            color = YELLOW if i == selected else WHITE
            text = menu_font.render(option, True, color)
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 250 + i * 60))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

def get_episode_input(screen, clock):
    # Font setup
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    input_text = ""
    
    # Input box
    input_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, 200, 40)
    active = True
    
    # Preset options
    presets = [
        {"text": "Short Training (1,000)", "value": 1000},
        {"text": "Medium Training (5,000)", "value": 5000},
        {"text": "Long Training (20,000)", "value": 20000}
    ]
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        episodes = int(input_text)
                        if episodes > 0:
                            return episodes
                    except ValueError:
                        pass
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    return 0
                elif event.unicode.isdigit():
                    input_text += event.unicode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if any preset option was clicked
                mouse_pos = pygame.mouse.get_pos()
                for i, preset in enumerate(presets):
                    btn_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 80 + i*40, 300, 30)
                    if btn_rect.collidepoint(mouse_pos):
                        return preset["value"]
        
        # Draw input screen
        screen.fill(BLACK)
        
        # Title
        title_text = font.render("Enter number of training episodes:", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 2 - 80))
        
        # Input box
        pygame.draw.rect(screen, WHITE, input_rect, 2)
        text_surface = font.render(input_text, True, WHITE)
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
        
        # Instructions
        instruction_text = small_font.render("Press ENTER to confirm, ESC to cancel", True, WHITE)
        screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 
                                       SCREEN_HEIGHT // 2 + 50))
        
        # Preset options
        preset_title = small_font.render("Or choose a preset training duration:", True, WHITE)
        screen.blit(preset_title, (SCREEN_WIDTH // 2 - preset_title.get_width() // 2, SCREEN_HEIGHT // 2 + 80 - 30))
        for i, preset in enumerate(presets):
            btn_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 80 + i*40, 300, 30)
            # Highlight the button if mouse is over it
            mouse_pos = pygame.mouse.get_pos()
            button_color = (100, 100, 100) if btn_rect.collidepoint(mouse_pos) else (50, 50, 50)
            pygame.draw.rect(screen, button_color, btn_rect)
            pygame.draw.rect(screen, WHITE, btn_rect, 1)
            option_text = small_font.render(preset["text"], True, WHITE)
            screen.blit(option_text, (btn_rect.centerx - option_text.get_width() // 2, 
                                      btn_rect.centery - option_text.get_height() // 2))
        
        pygame.display.flip()
        clock.tick(30)
    
    return 0

if __name__ == "__main__":
    display_menu()
