import pygame
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import csv
import datetime

# ================== Game Configuration ==================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_WIDTH = 20
PLAYER_HEIGHT = 100
PUCK_SIZE = 15
ZONE_WIDTH = SCREEN_WIDTH // 5  # Color-coded zones for rewards/penalties

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Game Zones
REWARD_ZONES = {
    'goal': (SCREEN_WIDTH - ZONE_WIDTH, 0, ZONE_WIDTH, SCREEN_HEIGHT * 0.4),
    'defensive': (0, 0, ZONE_WIDTH, SCREEN_HEIGHT),
    'neutral': (ZONE_WIDTH, 0, SCREEN_WIDTH - 2*ZONE_WIDTH, SCREEN_HEIGHT)
}

# ================== Reinforcement Learning Environment ==================
class AirHockeyEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=no movement
        self.action_space = spaces.Discrete(5)
        
        # State space: AI pos, puck pos & velocity, player pos
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -5, -5, 0, 0]),
            high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT]*3 + [5, 5]),
            dtype=np.float32
        )
        
        # Initialize game elements
        self.reset()
        self.screen = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize positions
        self.ai_pos = np.array([SCREEN_WIDTH - ZONE_WIDTH*1.5, SCREEN_HEIGHT/2])
        self.player_pos = np.array([ZONE_WIDTH*1.5, SCREEN_HEIGHT/2])
        self.puck_pos = np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])
        self.puck_vel = np.array([0, 0], dtype=np.float32)
        
        # Game state
        self.ai_score = 0
        self.player_score = 0
        self.steps = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.concatenate([
            self.ai_pos, self.puck_pos, self.puck_vel, self.player_pos
        ])
    
    def _check_zone(self, pos):
        x, y = pos
        if x > SCREEN_WIDTH - ZONE_WIDTH and y < SCREEN_HEIGHT * 0.4:
            return 'goal'
        elif x < ZONE_WIDTH:
            return 'defensive'
        else:
            return 'neutral'
    
    def step(self, action):
        # AI movement
        move_speed = 5
        if action == 0:  # Up
            self.ai_pos[1] = max(0, self.ai_pos[1] - move_speed)
        elif action == 1:  # Down
            self.ai_pos[1] = min(SCREEN_HEIGHT, self.ai_pos[1] + move_speed)
        elif action == 2:  # Left
            self.ai_pos[0] = max(SCREEN_WIDTH - ZONE_WIDTH*2, 
                                self.ai_pos[0] - move_speed)
        elif action == 3:  # Right
            self.ai_pos[0] = min(SCREEN_WIDTH - ZONE_WIDTH,
                                self.ai_pos[0] + move_speed)
        
        # Simulate puck physics (with 0.99% friction)
        self.puck_vel *= 0.999
        self.puck_pos += self.puck_vel
        
        # Collision detection
        if self._check_collision(self.ai_pos, self.puck_pos):
            self.puck_vel = (self.ai_pos - self.puck_pos) * 0.1
            
        # Player AI logic (simple tracking)
        if self.puck_pos[1] > self.player_pos[1]:
            self.player_pos[1] = min(SCREEN_HEIGHT, self.player_pos[1] + 3)
        else:
            self.player_pos[1] = max(0, self.player_pos[1] - 3)
            
        # Scoring
        done = False
        reward = 0
        
        # Goal detection
        if (self.puck_pos[0] > SCREEN_WIDTH - ZONE_WIDTH and 
            self.puck_pos[1] < SCREEN_HEIGHT * 0.4):
            reward = 100
            self.ai_score += 1
            done = True
            
        elif (self.puck_pos[0] < ZONE_WIDTH and 
              self.puck_pos[1] < SCREEN_HEIGHT * 0.4):
            reward = -100
            self.player_score += 1
            done = True
            
        # Zone penalties/rewards
        zone = self._check_zone(self.ai_pos)
        if zone == 'goal':
            reward -= 5  # Penalty for staying in goal area
        elif zone == 'defensive':
            reward += 0.1  # Reward for staying in defensive area
            
        # Proactive behavior reward
        if np.linalg.norm(self.puck_pos - self.ai_pos) < 50:
            reward += 0.5
            
        # Movement penalty
        if np.linalg.norm(self.puck_vel) < 0.1:
            reward -= 0.1
            
        self.steps += 1
        truncated = self.steps > 1000
        
        return self._get_obs(), reward, done, truncated, {}
    
    def _check_collision(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) < (PLAYER_WIDTH + PUCK_SIZE)/2
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Air Field Hockey")
            
        self.screen.fill(BLACK)
        
        # Draw zones
        pygame.draw.rect(self.screen, GREEN, REWARD_ZONES['goal'])
        pygame.draw.rect(self.screen, BLUE, REWARD_ZONES['defensive'])
        
        # Draw players
        pygame.draw.rect(self.screen, YELLOW, 
                        (*self.ai_pos, PLAYER_WIDTH, PLAYER_HEIGHT))
        pygame.draw.rect(self.screen, WHITE, 
                        (*self.player_pos, PLAYER_WIDTH, PLAYER_HEIGHT))
        
        # Draw puck
        pygame.draw.circle(self.screen, RED, 
                         (int(self.puck_pos[0]), int(self.puck_pos[1])), PUCK_SIZE)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"AI: {self.ai_score} | Player: {self.player_score}", 
                               True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        
    def close(self):
        if self.screen:
            pygame.quit()

# ================== Deep Q-Network Implementation ==================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        return self.net(x)

# ================== Training Framework ==================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename='ai_model.pth'):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename='ai_model.pth'):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()

# ================== Game Logic with AI Integration ==================
class AirHockeyGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Air Field Hockey")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.menu_active = True
        self.training_active = False
        self.game_active = False
        self.ai_agent = DQNAgent(8, 5)  # 8-state observation space
        self.load_ai_model()
        
    def load_ai_model(self):
        if os.path.exists('ai_model.pth'):
            self.ai_agent.load()
            print("AI model loaded successfully")
    
    def menu(self):
        while self.menu_active:
            self.screen.fill(BLACK)
            title = self.font.render("Air Field Hockey", True, WHITE)
            train = self.font.render("1. Train AI", True, WHITE)
            play = self.font.render("2. Play vs AI", True, WHITE)
            
            self.screen.blit(title, (SCREEN_WIDTH//2 - 100, 100))
            self.screen.blit(train, (SCREEN_WIDTH//2 - 50, 250))
            self.screen.blit(play, (SCREEN_WIDTH//2 - 50, 300))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.menu_active = False
                    return False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.menu_active = False
                        self.training_active = True
                        return True
                    elif event.key == pygame.K_2:
                        self.menu_active = False
                        self.game_active = True
                        return True
            
            self.clock.tick(30)
        return False
    
    def train_ai(self, episodes=1000):
        env = AirHockeyEnv()
        best_score = -np.inf
        log_file = f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Epsilon'])
            
            for episode in range(episodes):
                state, _ = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = self.ai_agent.act(state)
                    next_state, reward, done, _, _ = env.step(action)
                    self.ai_agent.remember(state, action, reward, next_state, done)
                    self.ai_agent.replay()
                    state = next_state
                    total_reward += reward
                    
                if episode % 10 == 0:
                    self.ai_agent.update_target_model()
                    
                if episode % 50 == 0:
                    self.ai_agent.save()
                    
                writer.writerow([episode, total_reward, self.ai_agent.epsilon])
                print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {self.ai_agent.epsilon:.2f}")
                
        env.close()
        self.training_active = False
        self.menu_active = True
    
    def play_game(self):
        env = AirHockeyEnv()
        player_pos = np.array([ZONE_WIDTH*1.5, SCREEN_HEIGHT/2])
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # Player control with mouse
            mouse_y = pygame.mouse.get_pos()[1]
            player_pos[1] = mouse_y
            
            # AI control
            state = env._get_obs()
            action = self.ai_agent.act(state)
            
            # Execute action
            next_state, _, done, _, _ = env.step(action)
            env.player_pos = player_pos  # Override player position
            
            if done:
                env.reset()
                
            env.render()
            self.clock.tick(60)
            
        env.close()
        self.game_active = False
        self.menu_active = True

# ================== Main Execution ==================
if __name__ == "__main__":
    game = AirHockeyGame()
    
    while True:
        if not game.menu():
            break
            
        if game.training_active:
            print("Starting training...")
            game.train_ai(episodes=10)
            
        elif game.game_active:
            print("Starting game...")
            game.play_game()