"""NumPy DQN for Tank Trouble. Train with: python3 tank_nn.py --episodes 2000"""
from __future__ import annotations

import argparse
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from tank import COLS, ROWS, OX, OY, FIELD_W, FIELD_H
from tankai import ACTIONS, TankEnv

MODEL = Path(__file__).with_name('tank_nn_v2_best.npz')
MAP_SEED = 42
MAX_BULLETS = 10
MAP_FEATURES = COLS * ROWS * 2
INPUT_SIZE = MAP_FEATURES + 10 + 5 + MAX_BULLETS * 6
HIDDEN = 128
HIDDEN_LAYERS = 3
GAMMA = .98
N_STEP = 3


def observation(game):
    """Full map plus tank-relative enemy and distance-sorted bullet features."""
    values = []
    for y in range(ROWS):
        for x in range(COLS):
            opened = game.maze.openings[(x, y)]
            values.extend((float('E' in opened), float('S' in opened)))
    for tank in game.tanks:
        values.extend(((tank.pos.x - OX) / FIELD_W,
                       (tank.pos.y - OY) / FIELD_H,
                       np.cos(tank.angle), np.sin(tank.angle),
                       min(tank.cooldown / .34, 1.0)))
    player, enemy = game.tanks
    forward = np.asarray((np.cos(player.angle), np.sin(player.angle)))
    side = np.asarray((-forward[1], forward[0]))
    offset = np.asarray(enemy.pos - player.pos)
    relative_angle = enemy.angle - player.angle
    values.extend((float(offset @ forward / FIELD_W),
                   float(offset @ side / FIELD_H),
                   float(np.linalg.norm(offset) / FIELD_W),
                   float(np.cos(relative_angle)), float(np.sin(relative_angle))))
    bullets = sorted(game.bullets, key=lambda bullet: player.pos.distance_squared_to(bullet.pos))
    for bullet in bullets[:MAX_BULLETS]:
        relative_pos = np.asarray(bullet.pos - player.pos)
        velocity = np.asarray(bullet.velocity)
        values.extend((float(relative_pos @ forward / FIELD_W),
                       float(relative_pos @ side / FIELD_H),
                       float(velocity @ forward / 420),
                       float(velocity @ side / 420),
                       1.0 if bullet.owner == 0 else -1.0, 1.0))
    values.extend([0.0] * (MAX_BULLETS - min(len(bullets), MAX_BULLETS)) * 6)
    return np.asarray(values, dtype=np.float32)


class Network:
    """Three hidden ReLU layers; hand-written NumPy forward/backward pass."""

    def __init__(self, rng):
        sizes = (INPUT_SIZE,) + (HIDDEN,) * HIDDEN_LAYERS + (len(ACTIONS),)
        self.weights = [rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2 / a)
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(x) for x in self.weights + self.biases]
        self.v = [np.zeros_like(x) for x in self.weights + self.biases]
        self.updates = 0

    def predict(self, x, cache=False):
        activations = [x]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1] @ weight + bias, 0))
        q = activations[-1] @ self.weights[-1] + self.biases[-1]
        return (q, activations) if cache else q

    def train_batch(self, x, actions, targets, learning_rate=0.0005):
        q, activations = self.predict(x, cache=True)
        chosen = q[np.arange(len(actions)), actions]
        error = chosen - targets
        # Huber derivative keeps unusually large TD errors from destabilizing training.
        gradient = np.where(np.abs(error) < 1, error, np.sign(error)) / len(actions)
        delta = np.zeros_like(q)
        delta[np.arange(len(actions)), actions] = gradient
        layer_count = len(self.weights)
        dw = [None] * layer_count
        db = [None] * layer_count
        for layer in range(layer_count - 1, -1, -1):
            dw[layer] = activations[layer].T @ delta
            db[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer].T) * (activations[layer] > 0)
        self.updates += 1
        gradients = dw + db
        params = self.weights + self.biases
        for i, (param, grad) in enumerate(zip(params, gradients)):
            grad = np.clip(grad, -1, 1)
            self.m[i] = .9 * self.m[i] + .1 * grad
            self.v[i] = .999 * self.v[i] + .001 * grad * grad
            mhat = self.m[i] / (1 - .9 ** self.updates)
            vhat = self.v[i] / (1 - .999 ** self.updates)
            param -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
        return float(np.mean(np.where(np.abs(error) < 1, .5 * error ** 2, np.abs(error) - .5)))

    def copy_from(self, other):
        for target, source in zip(self.weights + self.biases, other.weights + other.biases):
            target[:] = source

    def save(self, path):
        np.savez_compressed(path, input_size=INPUT_SIZE, action_size=len(ACTIONS),
                            hidden_size=HIDDEN, hidden_layers=HIDDEN_LAYERS,
                            **{f'w{i}': value for i, value in enumerate(self.weights)},
                            **{f'b{i}': value for i, value in enumerate(self.biases)})

    @classmethod
    def load(cls, path):
        net = cls(np.random.default_rng(0))
        with np.load(path, allow_pickle=False) as data:
            if ('hidden_size' not in data or 'hidden_layers' not in data or
                int(data['input_size']) != INPUT_SIZE or
                int(data['action_size']) != len(ACTIONS) or
                int(data['hidden_size']) != HIDDEN or
                int(data['hidden_layers']) != HIDDEN_LAYERS):
                raise ValueError('Model architecture does not match this version')
            for i in range(len(net.weights)):
                net.weights[i][:] = data[f'w{i}']
                net.biases[i][:] = data[f'b{i}']
        return net


def choose(net, state, epsilon):
    if epsilon > 0 and random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(net.predict(state[None, :])[0]))


def evaluate(net, games=100, seed=12345, fixed_map=True):
    random.seed(seed)
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED,
                  spawn_mode='anywhere', spawn_seed=seed)
    outcomes = [0, 0, 0]
    self_hits = 0
    steps = 0
    for _ in range(games):
        env.reset()
        while True:
            action = choose(net, observation(env.game), 0)
            _, reward, terminated, truncated = env.step(action)
            if terminated or truncated:
                outcomes[0 if reward > 0 else 1 if reward < 0 else 2] += 1
                self_hits += bool(env.last_hit and env.last_hit[0] == env.last_hit[1] == 0)
                steps += env.steps
                break
    return {'win': outcomes[0], 'loss': outcomes[1], 'draw': outcomes[2],
            'self_hits': self_hits, 'mean_steps': round(steps / games, 1)}


def add_n_step(pending, replay, flush=False):
    while len(pending) >= N_STEP or (flush and pending):
        window = list(pending)[:N_STEP]
        accumulated = sum((GAMMA ** i) * item[2] for i, item in enumerate(window))
        first, last = window[0], window[-1]
        replay.append((first[0], first[1], accumulated, last[3], last[4], len(window)))
        pending.popleft()


def train(episodes, model_path, seed, fixed_map=True, eval_interval=200,
          eval_games=40, resume=False):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED,
                  spawn_mode='anywhere', spawn_seed=seed)
    online = Network.load(model_path) if resume else Network(rng)
    target = Network(rng)
    target.copy_from(online)
    replay = deque(maxlen=10000)
    total_steps = 0
    outcomes = [0, 0, 0]
    losses = []
    best_wins = -1
    latest_path = model_path.with_name(model_path.stem + '_latest.npz')
    for episode in range(1, episodes + 1):
        env.reset()
        state = observation(env.game)
        pending = deque()
        epsilon = max(.05, 1.0 - .95 * (episode - 1) / max(1, .8 * episodes))
        while True:
            action = choose(online, state, epsilon)
            _, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            next_state = observation(env.game)
            pending.append((state, action, reward, next_state, terminated))
            add_n_step(pending, replay, flush=done)
            state = next_state
            total_steps += 1
            if len(replay) >= 256 and total_steps % 4 == 0:
                batch = random.sample(replay, 64)
                states = np.stack([item[0] for item in batch])
                actions = np.asarray([item[1] for item in batch], dtype=np.int64)
                rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
                next_states = np.stack([item[3] for item in batch])
                terminated_batch = np.asarray([item[4] for item in batch], dtype=np.float32)
                horizons = np.asarray([item[5] for item in batch], dtype=np.int32)
                next_q = target.predict(next_states).max(axis=1)
                targets = rewards + (GAMMA ** horizons) * (1 - terminated_batch) * next_q
                losses.append(online.train_batch(states, actions, targets))
            if total_steps % 500 == 0:
                target.copy_from(online)
            if done:
                outcomes[0 if reward > 0 else 1 if reward < 0 else 2] += 1
                break
        if episode % 25 == 0 or episode == episodes:
            mean_loss = sum(losses) / len(losses) if losses else 0
            print(f'{episode}/{episodes} steps={total_steps} win/loss/draw={outcomes} '
                  f'epsilon={epsilon:.2f} loss={mean_loss:.4f}', flush=True)
            outcomes = [0, 0, 0]
            losses = []
        if episode % eval_interval == 0 or episode == episodes:
            saved_random_state = random.getstate()
            score = evaluate(online, eval_games, seed=10000, fixed_map=fixed_map)
            random.setstate(saved_random_state)
            online.save(latest_path)
            if score['win'] > best_wins:
                best_wins = score['win']
                online.save(model_path)
            print(f'validation at {episode}: {score}; best wins={best_wins}/{eval_games}',
                  flush=True)
    pygame.quit()
    print(f'Saved best: {model_path}; latest: {latest_path}')


def watch(model_path, fixed_map=True):
    net = Network.load(model_path)
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED,
                  spawn_mode='anywhere')
    env.reset()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        action = choose(net, observation(env.game), 0)
        _, reward, terminated, truncated = env.step(action)
        done = terminated or truncated
        env.game.draw()
        if done:
            print('Neural AI win' if reward > 0 else 'Neural AI loss' if reward < 0 else 'Draw')
            pygame.time.wait(500)
            env.reset()
        clock.tick(30)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Train a small NumPy DQN tank agent')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--evaluate', type=int, metavar='GAMES')
    parser.add_argument('--random-map', action='store_true',
                        help='Use a new random maze each round instead of the fixed training maze')
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--eval-games', type=int, default=40)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.watch:
        watch(args.model, fixed_map=not args.random_map)
    elif args.evaluate is not None:
        if args.evaluate < 1:
            parser.error('--evaluate must be positive')
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        print(evaluate(Network.load(args.model), args.evaluate,
                       args.seed, fixed_map=not args.random_map))
    else:
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        if args.eval_interval < 1 or args.eval_games < 1:
            parser.error('evaluation interval and game count must be positive')
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        train(args.episodes, args.model, args.seed, fixed_map=not args.random_map,
              eval_interval=args.eval_interval, eval_games=args.eval_games,
              resume=args.resume)


if __name__ == '__main__':
    main()
