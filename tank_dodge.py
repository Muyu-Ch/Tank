"""Train a pure neural dodging agent: python3 tank_dodge.py --episodes 2000."""
from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from tank import BG, HEIGHT, MUTED, TEXT, WALL_COLOR, WIDTH, YELLOW, Bullet, Game, Tank, hits_wall, unit

FPS = 30
DT = 1 / FPS
MAX_STEPS = 600
FIRE_INTERVAL = .65
BULLET_SPEED = 320
WALL_RADIUS = 130
BULLET_RADIUS = 210
BULLET_SLOTS = 6
WALL_RAYS = 8
FRAME_SIZE = WALL_RAYS + BULLET_SLOTS * 4
INPUT_SIZE = FRAME_SIZE * 2  # Two observations let the agent infer bullet motion.
ACTIONS = [(forward, turn) for forward in (-1, 0, 1) for turn in (-1, 0, 1)]
MODEL = Path(__file__).with_name('tank_dodge_best.npz')
ARENA = pygame.Rect(32, 112, 1056, 616)
WALL = 10


def make_walls():
    x, y, w, h = ARENA
    return [
        pygame.Rect(x - WALL, y - WALL, w + 2 * WALL, WALL),
        pygame.Rect(x - WALL, y + h, w + 2 * WALL, WALL),
        pygame.Rect(x - WALL, y, WALL, h),
        pygame.Rect(x + w, y, WALL, h),
        pygame.Rect(x + 240, y + 130, WALL, 120),
        pygame.Rect(x + 760, y + 360, WALL, 120),
        pygame.Rect(x + 460, y + 280, 130, WALL),
    ]


def dodge_frame(tank, walls, bullets):
    """One local observation in the tank's reference frame."""
    features = []
    for i in range(WALL_RAYS):
        ray = unit(tank.angle + i * 2 * math.pi / WALL_RAYS)
        distance = WALL_RADIUS
        for step in range(10, WALL_RADIUS + 1, 10):
            if hits_wall(tank.pos + ray * step, 9, walls):
                distance = step
                break
        features.append(distance / WALL_RADIUS)
    nearest = sorted(bullets, key=lambda bullet: tank.pos.distance_squared_to(bullet.pos))
    for bullet in nearest[:BULLET_SLOTS]:
        delta = bullet.pos - tank.pos
        angle = math.atan2(delta.y, delta.x) - tank.angle
        features.extend((min(delta.length() / BULLET_RADIUS, 1.0),
                         math.sin(angle), math.cos(angle), 1.0))
    features.extend([0.0] * (BULLET_SLOTS - min(len(nearest), BULLET_SLOTS)) * 4)
    return np.asarray(features, dtype=np.float32)


class DodgeEnv:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.walls = make_walls()
        self.tank = Tank(pygame.Vector2(), 0, YELLOW, '闪避 AI')
        self.bullets = []
        self.steps = 0
        self.fire_clock = 0.0
        self.direction_index = 0
        self.phase = 0.0
        self.previous = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.shots = 0
        self.reset()

    def reset(self):
        while True:
            pos = pygame.Vector2(self.rng.uniform(ARENA.left + 80, ARENA.right - 80),
                                 self.rng.uniform(ARENA.top + 80, ARENA.bottom - 80))
            if not hits_wall(pos, 28, self.walls):
                break
        self.tank.pos = pos
        self.tank.angle = self.rng.uniform(-math.pi, math.pi)
        self.tank.alive = True
        self.bullets = []
        self.steps = 0
        self.fire_clock = 0.0
        self.direction_index = 0
        self.phase = self.rng.uniform(-math.pi, math.pi)
        self.shots = 0
        current = self.frame()
        self.previous = current.copy()
        return np.concatenate((current, self.previous))

    def frame(self):
        return dodge_frame(self.tank, self.walls, self.bullets)

    def fire(self):
        # Cycle through all eight compass directions, with a small angle variation.
        for offset in range(8):
            index = (self.direction_index + offset) % 8
            angle = self.phase + index * math.pi / 4 + self.rng.uniform(-.12, .12)
            ray = unit(angle)
            for distance in (170, 150, 130, 110, 90):
                origin = self.tank.pos + ray * distance
                if hits_wall(origin, 7, self.walls):
                    continue
                if any(hits_wall(self.tank.pos + ray * part, 6, self.walls)
                       for part in range(20, distance, 10)):
                    continue
                target = self.tank.pos + pygame.Vector2(self.rng.uniform(-15, 15),
                                                         self.rng.uniform(-15, 15))
                velocity = (target - origin).normalize() * BULLET_SPEED
                self.bullets.append(Bullet(origin, velocity, -1, life=4.0))
                self.shots += 1
                self.direction_index = (index + 1) % 8
                return
        self.direction_index = (self.direction_index + 1) % 8

    def step(self, action):
        forward, turn = ACTIONS[action]
        self.tank.update(DT, forward, turn, self.walls)
        self.fire_clock += DT
        if self.fire_clock >= FIRE_INTERVAL:
            self.fire_clock -= FIRE_INTERVAL
            self.fire()
        for bullet in self.bullets[:]:
            if not bullet.update(DT, self.walls):
                self.bullets.remove(bullet)
                continue
            if self.tank.pos.distance_to(bullet.pos) < 22:
                self.tank.alive = False
        self.steps += 1
        current = self.frame()
        state = np.concatenate((current, self.previous))
        self.previous = current
        terminated = not self.tank.alive
        truncated = self.steps >= MAX_STEPS and not terminated
        return state, DT if not terminated else 0.0, terminated, truncated


class DodgeNetwork:
    def __init__(self, rng):
        sizes = (INPUT_SIZE, 64, 64, len(ACTIONS))
        self.weights = [rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2 / a)
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(value) for value in self.weights + self.biases]
        self.v = [np.zeros_like(value) for value in self.weights + self.biases]
        self.updates = 0

    def predict(self, x, cache=False):
        activations = [x]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1] @ weight + bias, 0))
        output = activations[-1] @ self.weights[-1] + self.biases[-1]
        return (output, activations) if cache else output

    def train_batch(self, states, actions, targets):
        q, activations = self.predict(states, cache=True)
        error = q[np.arange(len(actions)), actions] - targets
        delta = np.zeros_like(q)
        delta[np.arange(len(actions)), actions] = np.clip(error, -1, 1) / len(actions)
        dw = [None] * 3
        db = [None] * 3
        for layer in (2, 1, 0):
            dw[layer] = activations[layer].T @ delta
            db[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer].T) * (activations[layer] > 0)
        self.updates += 1
        for index, (param, gradient) in enumerate(zip(self.weights + self.biases, dw + db)):
            gradient = np.clip(gradient, -1, 1)
            self.m[index] = .9 * self.m[index] + .1 * gradient
            self.v[index] = .999 * self.v[index] + .001 * gradient * gradient
            mhat = self.m[index] / (1 - .9 ** self.updates)
            vhat = self.v[index] / (1 - .999 ** self.updates)
            param -= .0005 * mhat / (np.sqrt(vhat) + 1e-8)
        return float(np.abs(error).mean())

    def copy_from(self, other):
        for target, source in zip(self.weights + self.biases, other.weights + other.biases):
            target[:] = source

    def save(self, path):
        np.savez_compressed(path, input_size=INPUT_SIZE,
                            **{f'w{i}': value for i, value in enumerate(self.weights)},
                            **{f'b{i}': value for i, value in enumerate(self.biases)})

    @classmethod
    def load(cls, path):
        net = cls(np.random.default_rng(0))
        with np.load(path, allow_pickle=False) as data:
            if int(data['input_size']) != INPUT_SIZE:
                raise ValueError('Dodging model input size mismatch')
            for i in range(3):
                net.weights[i][:] = data[f'w{i}']
                net.biases[i][:] = data[f'b{i}']
        return net


def choose(net, state, epsilon=0):
    if epsilon and random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(net.predict(state[None, :])[0]))


def evaluate(net, games=100, seed=12345):
    env = DodgeEnv(seed)
    times = []
    for _ in range(games):
        state = env.reset()
        while True:
            state, _, terminated, truncated = env.step(choose(net, state))
            if terminated or truncated:
                times.append(env.steps * DT)
                break
    return sum(times) / len(times), sum(time >= MAX_STEPS * DT for time in times)


def train(episodes, model_path, seed, eval_interval=100):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = DodgeEnv(seed)
    online = DodgeNetwork(rng)
    target = DodgeNetwork(rng)
    target.copy_from(online)
    replay = deque(maxlen=20000)
    total_steps = 0
    best_time = -1.0
    durations = []
    latest = model_path.with_name(model_path.stem + '_latest.npz')
    for episode in range(1, episodes + 1):
        state = env.reset()
        epsilon = max(.05, 1 - .95 * (episode - 1) / max(1, .8 * episodes))
        while True:
            action = choose(online, state, epsilon)
            next_state, reward, terminated, truncated = env.step(action)
            replay.append((state, action, reward, next_state, terminated))
            state = next_state
            total_steps += 1
            if len(replay) >= 512 and total_steps % 4 == 0:
                batch = random.sample(replay, 64)
                states = np.stack([item[0] for item in batch])
                actions = np.asarray([item[1] for item in batch], dtype=np.int64)
                rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
                next_states = np.stack([item[3] for item in batch])
                dead = np.asarray([item[4] for item in batch], dtype=np.float32)
                targets = rewards + .99 * (1 - dead) * target.predict(next_states).max(axis=1)
                online.train_batch(states, actions, targets)
            if total_steps % 500 == 0:
                target.copy_from(online)
            if terminated or truncated:
                durations.append(env.steps * DT)
                break
        if episode % 25 == 0 or episode == episodes:
            print(f'{episode}/{episodes} mean_survival={sum(durations)/len(durations):.2f}s '
                  f'epsilon={epsilon:.2f} steps={total_steps}', flush=True)
            durations.clear()
        if episode % eval_interval == 0 or episode == episodes:
            mean_time, completed = evaluate(online, games=30, seed=10000)
            online.save(latest)
            if mean_time > best_time:
                best_time = mean_time
                online.save(model_path)
            print(f'validation {episode}: {mean_time:.2f}s, full_survival={completed}/30; '
                  f'best={best_time:.2f}s', flush=True)
    print(f'Saved best: {model_path}; latest: {latest}')


def watch(model_path, seed):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('坦克动荡 · 纯神经网络闪避训练')
    font = Game.load_font(25)
    small = Game.load_font(18)
    clock = pygame.time.Clock()
    env = DodgeEnv(seed)
    net = DodgeNetwork.load(model_path)
    state = env.reset()
    last_result = ''
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                state = env.reset()
                last_result = ''
        if not running:
            break
        if not paused:
            state, _, terminated, truncated = env.step(choose(net, state))
            if terminated or truncated:
                last_result = f'存活 {env.steps * DT:.1f} 秒'
                state = env.reset()
        screen.fill(BG)
        for wall in env.walls:
            pygame.draw.rect(screen, WALL_COLOR, wall)
        for bullet in env.bullets:
            bullet.draw(screen)
        env.tank.draw(screen)
        title = font.render(f'纯神经网络闪避   当前 {env.steps * DT:.1f} 秒', True, TEXT)
        screen.blit(title, (32, 38))
        hint = small.render('Esc 退出  P 暂停  N 新一局', True, MUTED)
        screen.blit(hint, (WIDTH - hint.get_width() - 32, 48))
        if last_result:
            result = small.render(last_result, True, TEXT)
            screen.blit(result, (32, HEIGHT - 26))
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Train or watch a pure neural dodging agent')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate', type=int, default=0, metavar='GAMES')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        watch(args.model, args.seed)
    elif args.evaluate:
        net = DodgeNetwork.load(args.model)
        mean_time, completed = evaluate(net, args.evaluate, args.seed)
        print(f'mean survival: {mean_time:.2f}s; full 20s: {completed}/{args.evaluate}')
    else:
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        train(args.episodes, args.model, args.seed)


if __name__ == '__main__':
    main()
