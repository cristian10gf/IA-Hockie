import os
import sys
import math
import random
import pickle
import argparse
from collections import deque
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------
# 1. Entorno personalizado AirHockey
# -----------------------------------
class AirHockeyEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        # Dimensiones
        self.W, self.H = 600, 300
        self.goal_h = int(self.H * 0.4)
        # Definir espacio de observación: [x_agent, y_agent, x_op, y_op, x_disk, y_disk, vx, vy]
        high = np.array([self.W, self.H, self.W, self.H, self.W, self.H, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 4 acciones discretas
        self.action_space = spaces.Discrete(4)
        # Parámetros físicos
        self.friction = 0.01
        # Estado
        self.reset()

    def reset(self):
        # Posiciones iniciales
        self.agent_pos = np.array([self.W*0.5, self.H*0.75], dtype=np.float32)
        self.op_pos    = np.array([self.W*0.5, self.H*0.25], dtype=np.float32)

        self.disk_pos  = np.array([self.W*0.5, self.H*0.5])
        self.disk_vel  = np.zeros(2)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.op_pos, self.disk_pos, self.disk_vel]).astype(np.float32)

    def step(self, action):
        # Mover agente
        delta = {0:(0,-5),1:(0,5),2:(-5,0),3:(5,0)}[action]
        new_pos = self.agent_pos + np.array(delta)
        # Penalizar movimientos fuera de zona
        if not (0 <= new_pos[0] <= self.W and self.H*0.5 <= new_pos[1] <= self.H):
            reward = -5
        else:
            self.agent_pos = new_pos
            reward = 0
        # Actualizar física del disco
        self.disk_pos += self.disk_vel
        self.disk_vel *= (1 - self.friction)
        # Colisión disco–pala agente
        if np.linalg.norm(self.agent_pos - self.disk_pos) < 20:
            dir = (self.disk_pos - self.agent_pos)
            self.disk_vel += dir/np.linalg.norm(dir)*5
            reward += 1  # golpe válido
        # Oponente de reglas simple
        self._opponent_move()
        # Comprobar goles
        done = False
        # Agente anota
        if self.disk_pos[1] < 0:
            reward += 100; done=True
        # Oponente anota
        if self.disk_pos[1] > self.H:
            reward -= 100; done=True
        # Shaping: reducción de distancia al gol
        dist_prev = abs((self.H*0) - self.disk_pos[1] - self.disk_vel[1])
        dist_now  = abs(0 - self.disk_pos[1])
        reward += 0.99*( -dist_now ) - ( -dist_prev )
        # Step penalty
        reward -= 0.1
        return self._get_obs(), reward, done, False, {}

    def _opponent_move(self):
        # Persigue el disco
        direction = self.disk_pos - self.op_pos
        norm = np.linalg.norm(direction) or 1
        self.op_pos = self.op_pos + (direction/norm)*3

    def render(self, mode='human'):
        # Lo implementamos en pygame fuera del entorno
        pass

# -----------------------
# 2. Red neuronal DQN
# -----------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,128), nn.ReLU(),
            nn.Linear(128,128),    nn.ReLU(),
            nn.Linear(128,n_actions)
        )
    def forward(self,x): return self.net(x)

# --------------------------
# 3. Agente DQN con replay
# --------------------------
class Agent:
    def __init__(self, obs_dim, n_actions, lr=1e-3):
        self.net = DQN(obs_dim, n_actions)
        self.tgt = DQN(obs_dim, n_actions)
        self.tgt.load_state_dict(self.net.state_dict())
        self.buf = deque(maxlen=10000)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = 0.99
        self.eps   = 1.0
        self.steps = 0

    def act(self, s):
        if random.random() < self.eps:
            return random.randrange(env.action_space.n)
        q = self.net(torch.FloatTensor(s)).detach().numpy()
        return np.argmax(q)

    def push(self, s,a,r,ns,d):
        self.buf.append((s,a,r,ns,d))
        self.steps += 1
        if self.steps % 4 == 0: self.learn()

    def learn(self):
        if len(self.buf) < 1000: return
        batch = random.sample(self.buf, 64)
        s,a,r,ns,d = zip(*batch)
        s,ns = torch.FloatTensor(s), torch.FloatTensor(ns)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        d = torch.FloatTensor(d)
        q = self.net(s).gather(1,a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            tgt_q = self.tgt(ns).max(1)[0]
            y = r + self.gamma*(1-d)*tgt_q
        loss = nn.MSELoss()(q, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        # Actualizar target
        if self.steps % 1000 == 0:
            self.tgt.load_state_dict(self.net.state_dict())

# --------------------------
# 4. Entrenamiento / Juego
# --------------------------
def train(num_episodes):
    writer = SummaryWriter('logs')
    ag = Agent(env.observation_space.shape[0], env.action_space.n)
    for ep in range(num_episodes):
        s,_ = env.reset()
        total_r = 0
        for t in range(1000):
            a = ag.act(s)
            ns,r,done,_,_ = env.step(a)
            ag.push(s,a,r,ns,done)
            s = ns; total_r += r
            if done: break
        # Registra recompensa
        writer.add_scalar('Reward/episode', total_r, ep)
        ag.eps = max(0.1, ag.eps*0.995)
        print(f"Ep {ep+1}/{num_episodes}: R={total_r:.1f}, eps={ag.eps:.3f}")
    # Guardar modelo
    torch.save(ag.net.state_dict(), 'models/dqn_airhockey.pth')
    writer.close()

def play():
    # Inicialización
    net = DQN(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load('models/dqn_airhockey.pth'))
    pygame.init()
    screen = pygame.display.set_mode((env.W, env.H))
    clock = pygame.time.Clock()

    # Estado inicial
    s, _ = env.reset()
    running = True
    done = False

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        if not done:
            # Actualizar oponente según mouse
            mx, my = pygame.mouse.get_pos()
            env.op_pos = np.array([mx, my], dtype=np.float32)

            # Acción del agente
            a = net(torch.FloatTensor(s)).argmax().item()
            s, r, done, _, _ = env.step(a)

        # --- RENDER ---
        screen.fill((0, 0, 0))  # fondo negro

        # Metas
        pygame.draw.rect(screen, (255, 0, 0), (0, 0, env.W, 5))             # portería IA (arriba)
        pygame.draw.rect(screen, (0, 0, 255), (0, env.H-5, env.W, 5))      # portería jugador (abajo)

        # Jugadores y disco
        pygame.draw.circle(screen, (255, 255, 0), env.agent_pos.astype(int), 15)  # IA amarilla
        pygame.draw.circle(screen, (0, 255, 255), env.op_pos.astype(int), 15)     # jugador cian
        pygame.draw.circle(screen, (255, 255, 255), env.disk_pos.astype(int), 10) # disco blanco

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# --------------------------
# 5. Menú principal
# --------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','play'], required=True)
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Número de episodios para entrenar')
    args = parser.parse_args()
    env = AirHockeyEnv()
    if args.mode=='train':
        train(args.episodes)
    else:
        if not os.path.exists('models/dqn_airhockey.pth'):
            print("No hay modelo entrenado. Ejecuta primero con --mode train")
        else:
            play()
