"""Minimal tabular Q-learning agent for tank.py.

Train: python3 tankai.py --episodes 500
Watch: python3 tankai.py --watch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

if '--watch' not in sys.argv:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame

from tank import Game, hits_wall, unit

DT = 1 / 30
ACTIONS = [(forward, turn, shoot)
           for forward in (-1, 0, 1)
           for turn in (-1, 0, 1)
           for shoot in (False, True)]
DEFAULT_MODEL = Path(__file__).with_name('tankai_qtable.json')


def angle_difference(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class TankEnv:
    """One episode of the actual game physics against its existing rule-based AI."""

    def __init__(self, max_steps=600):
        self.game = Game()
        self.game.start(True)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        self.game.new_round()
        self.steps = 0
        return self.observe()

    def observe(self):
        game = self.game
        player, enemy = game.tanks
        offset = enemy.pos - player.pos
        bearing = angle_difference(math.atan2(offset.y, offset.x), player.angle)
        bearing_bin = int(((bearing + math.pi) / (2 * math.pi)) * 8) % 8
        distance_bin = 0 if offset.length() < 180 else (1 if offset.length() < 380 else 2)

        # Compact local map: collision probes ahead, ahead-left, ahead-right.
        wall_bits = 0
        for bit, delta in enumerate((0, -.6, .6)):
            ray = unit(player.angle + delta)
            if any(hits_wall(player.pos + ray * length, 8, game.maze.walls)
                   for length in (27, 43, 60)):
                wall_bits |= 1 << bit

        # Closest approaching enemy bullet; its direction is already updated on ricochet.
        danger = 0
        closest = 150
        for bullet in game.bullets:
            if bullet.owner == 0:
                continue
            displacement = player.pos - bullet.pos
            distance = displacement.length()
            if distance < closest and displacement.dot(bullet.velocity) > 0:
                closest = distance
                side = unit(player.angle).cross(bullet.pos - player.pos)
                danger = 1 if side < 0 else 2
        ready = int(player.cooldown <= 0 and sum(b.owner == 0 for b in game.bullets) < 5)
        return (bearing_bin, distance_bin, wall_bits, danger, ready)

    def step(self, action):
        game = self.game
        player = game.tanks[0]
        forward, turn, shoot = ACTIONS[action]
        player.update(DT, forward, turn, game.maze.walls)
        if shoot:
            game.shoot(0)
        game.ai_update(DT)

        for bullet in game.bullets[:]:
            if not bullet.update(DT, game.maze.walls):
                game.bullets.remove(bullet)
                continue
            for index, tank in enumerate(game.tanks):
                if tank.alive and (index != bullet.owner or bullet.age > .42) \
                        and tank.pos.distance_to(bullet.pos) < 20:
                    tank.alive = False
                    self.steps += 1
                    return self.observe(), (1.0 if index == 1 else -1.0), True
        self.steps += 1
        return self.observe(), 0.0, self.steps >= self.max_steps


class QAgent:
    def __init__(self, q=None):
        self.q = defaultdict(lambda: [0.0] * len(ACTIONS))
        if q:
            for key, values in q.items():
                self.q[tuple(map(int, key.split(',')))] = values

    def act(self, state, epsilon=0):
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        values = self.q[state]
        best = max(values)
        return random.choice([i for i, value in enumerate(values) if value == best])

    def learn(self, state, action, reward, next_state, done, alpha=.15, gamma=.98):
        target = reward if done else reward + gamma * max(self.q[next_state])
        self.q[state][action] += alpha * (target - self.q[state][action])

    def save(self, path):
        data = {','.join(map(str, state)): values for state, values in self.q.items()}
        path.write_text(json.dumps({'format': 1, 'q': data}, separators=(',', ':')), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('format') != 1:
            raise ValueError('Unsupported Q-table format')
        return cls(data['q'])


def train(episodes, model_path, seed):
    random.seed(seed)
    env = TankEnv()
    agent = QAgent.load(model_path) if model_path.exists() else QAgent()
    wins = losses = draws = 0
    for episode in range(1, episodes + 1):
        state = env.reset()
        epsilon = max(.05, .8 * (.995 ** (episode - 1)))
        while True:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                wins += reward > 0
                losses += reward < 0
                draws += reward == 0
                break
        if episode % 25 == 0 or episode == episodes:
            print(f'{episode}/{episodes}  win={wins} loss={losses} draw={draws} '
                  f'states={len(agent.q)} epsilon={epsilon:.2f}', flush=True)
            wins = losses = draws = 0
    agent.save(model_path)
    print(f'Saved model: {model_path}')
    pygame.quit()


def watch(model_path):
    agent = QAgent.load(model_path)
    env = TankEnv()
    state = env.reset()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        action = agent.act(state)
        state, reward, done = env.step(action)
        env.game.draw()
        if done:
            print('AI win' if reward > 0 else 'AI loss' if reward < 0 else 'Draw')
            pygame.time.wait(500)
            state = env.reset()
        clock.tick(30)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Train a simple Tank Trouble Q-learning agent')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--watch', action='store_true', help='Watch a trained agent against the rule AI')
    args = parser.parse_args()
    if args.watch:
        watch(args.model)
    else:
        if args.episodes < 1:
            parser.error('--episodes must be at least 1')
        train(args.episodes, args.model, args.seed)


if __name__ == '__main__':
    main()
