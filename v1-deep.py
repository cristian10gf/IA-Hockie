import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from sklearn.preprocessing import MinMaxScaler
import os
import time

# Environment constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PADDLE_SIZE = 40
PUCK_SIZE = 20
GOAL_HEIGHT = SCREEN_HEIGHT * 0.4
MAX_SPEED = 8
FRICTION = 0.99

# Reinforcement Learning constants
TRAIN_STEPS = 100000
SAVE_DIR = "saved_models"
LOG_DIR = "logs"

# Reward/Penalty rubric
REWARDS = {
    'goal_scored': 10,
    'goal_conceded': -5,
    'puck_interception': 1,
    'illegal_movement': -0.5,
    'defensive_positioning': 0.2,
    'puck_approach': 0.5
}

class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 directions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -MAX_SPEED, -MAX_SPEED]),
            high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, MAX_SPEED, MAX_SPEED]),
            dtype=np.float32
        )
        
        # Game state
        self.ai_paddle = None
        self.opponent_paddle = None
        self.puck = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def reset(self, seed=None):
        # Initialize positions
        self.ai_paddle = pygame.Rect(SCREEN_WIDTH/4 - PADDLE_SIZE/2, SCREEN_HEIGHT/2 - PADDLE_SIZE/2, PADDLE_SIZE, PADDLE_SIZE)
        self.opponent_paddle = pygame.Rect(3*SCREEN_WIDTH/4 - PADDLE_SIZE/2, SCREEN_HEIGHT/2 - PADDLE_SIZE/2, PADDLE_SIZE, PADDLE_SIZE)
        self.puck = {
            'rect': pygame.Rect(SCREEN_WIDTH/2 - PUCK_SIZE/2, SCREEN_HEIGHT/2 - PUCK_SIZE/2, PUCK_SIZE, PUCK_SIZE),
            'velocity': [0.0, 0.0]
        }
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.ai_paddle.x, self.ai_paddle.y,
            self.puck['rect'].x, self.puck['rect'].y,
            self.puck['velocity'][0], self.puck['velocity'][1]
        ], dtype=np.float32)
    
    def step(self, action):
        # Move AI paddle
        if action == 0: self.ai_paddle.y -= MAX_SPEED  # Up
        elif action == 1: self.ai_paddle.y += MAX_SPEED  # Down
        elif action == 2: self.ai_paddle.x -= MAX_SPEED  # Left
        elif action == 3: self.ai_paddle.x += MAX_SPEED  # Right
        
        # Keep paddle in bounds
        self.ai_paddle.x = np.clip(self.ai_paddle.x, 0, SCREEN_WIDTH/2 - PADDLE_SIZE)
        self.ai_paddle.y = np.clip(self.ai_paddle.y, 0, SCREEN_HEIGHT - PADDLE_SIZE)
        
        # Update puck position
        self.puck['rect'].x += self.puck['velocity'][0]
        self.puck['rect'].y += self.puck['velocity'][1]
        self.puck['velocity'] = [v * FRICTION for v in self.puck['velocity']]
        
        # Check collisions
        reward = self._handle_collisions()
        done = self._check_goal()
        
        return self._get_obs(), reward, done, False, {}
    
    def _handle_collisions(self):
        reward = 0
        # Paddle collision
        if self.ai_paddle.colliderect(self.puck['rect']):
            reward += REWARDS['puck_interception']
            puck_center = self.puck['rect'].center
            paddle_center = self.ai_paddle.center
            self.puck['velocity'] = [
                (puck_center[0] - paddle_center[0]) * 0.2,
                (puck_center[1] - paddle_center[1]) * 0.2
            ]
        
        # Wall collision
        if self.puck['rect'].top <= 0 or self.puck['rect'].bottom >= SCREEN_HEIGHT:
            self.puck['velocity'][1] *= -1
        return reward
    
    def _check_goal(self):
        # Check if puck entered either goal
        if self.puck['rect'].x <= 0:
            self.reward += REWARDS['goal_conceded']
            return True
        if self.puck['rect'].x + PUCK_SIZE >= SCREEN_WIDTH:
            self.reward += REWARDS['goal_scored']
            return True
        return False

class TrainCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"air_hockey_{self.n_calls}")
            self.model.save(model_path)
        return True

class AirHockeyGame:
    def __init__(self, ai_model=None, human_player=True):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.ai_model = ai_model
        self.human_player = human_player
        self.env = AirHockeyEnv()
        
    def run(self):
        obs, _ = self.env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Human player input
            if self.human_player:
                mouse_pos = pygame.mouse.get_pos()
                self.env.opponent_paddle.center = mouse_pos
                self.env.opponent_paddle.x = np.clip(self.env.opponent_paddle.x, SCREEN_WIDTH/2, SCREEN_WIDTH - PADDLE_SIZE)
                self.env.opponent_paddle.y = np.clip(self.env.opponent_paddle.y, 0, SCREEN_HEIGHT - PADDLE_SIZE)
            
            # AI prediction
            if self.ai_model:
                action, _ = self.ai_model.predict(obs)
                obs, _, done, _, _ = self.env.step(action)
                if done:
                    obs, _ = self.env.reset()
            
            # Render
            self.screen.fill((0, 0, 0))
            # Draw center line
            pygame.draw.line(self.screen, (255, 255, 255), (SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, SCREEN_HEIGHT), 2)
            # Draw goals
            pygame.draw.rect(self.screen, (0, 255, 0), (0, (SCREEN_HEIGHT - GOAL_HEIGHT)/2, 10, GOAL_HEIGHT))
            pygame.draw.rect(self.screen, (0, 255, 0), (SCREEN_WIDTH - 10, (SCREEN_HEIGHT - GOAL_HEIGHT)/2, 10, GOAL_HEIGHT))
            # Draw paddles and puck
            pygame.draw.rect(self.screen, (255, 255, 0), self.env.ai_paddle)
            pygame.draw.rect(self.screen, (0, 0, 255) if self.human_player else (255, 0, 0), self.env.opponent_paddle)
            pygame.draw.ellipse(self.screen, (255, 255, 255), self.env.puck['rect'])
            
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

def train_model(timesteps=TRAIN_STEPS):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    env = AirHockeyEnv()
    check_env(env)
    
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR)
    callback = TrainCallback(check_freq=10000, save_path=SAVE_DIR)
    
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(SAVE_DIR, "air_hockey_final"))
    print("Training completed!")

def play_game():
    latest_model = os.path.join(SAVE_DIR, "air_hockey_final.zip")
    if os.path.exists(latest_model):
        model = PPO.load(latest_model)
        game = AirHockeyGame(ai_model=model, human_player=True)
        game.run()
    else:
        print("No trained model found! Train first.")

def main_menu():
    while True:
        print("\nAir Hockey RL")
        print("1. Train Model")
        print("2. Play Against AI")
        print("3. Exit")
        choice = input("Select option: ")
        
        if choice == '1':
            timesteps = int(input("Enter training timesteps: "))
            train_model(timesteps)
        elif choice == '2':
            play_game()
        elif choice == '3':
            break
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main_menu()