"""Train a wall-aware dodge DQN with dense-fire and corner scenarios."""
from __future__ import annotations

import argparse
import math
import random
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from tank import BG, HEIGHT, MUTED, TEXT, WALL_COLOR, WIDTH, YELLOW, Bullet, Game, Tank, hits_wall, unit

FPS = 30
DT = 1 / FPS
MAX_STEPS = 600
WALL_RADIUS = 220
BULLET_RADIUS = 210
BULLET_SLOTS = 6
WALL_SLOTS = 24
WALL_FEATURES = 5
FRAME_SIZE = 2 + WALL_SLOTS * WALL_FEATURES + BULLET_SLOTS * 6
INPUT_SIZE = FRAME_SIZE * 2  # Two observations let the agent infer bullet motion.
ACTIONS = [(forward, turn) for forward in (-1, 0, 1) for turn in (-1, 0, 1)]
AI15_MODEL = Path(__file__).with_name('tank_dodge_walls_best.npz')
CANDIDATE_MODEL = Path(tempfile.gettempdir()) / 'tank_dodge_candidate.npz'
ARENA = pygame.Rect(32, 112, 1056, 616)
WALL = 10


def make_walls(rng, count):
    x, y, w, h = ARENA
    walls = [
        pygame.Rect(x - WALL, y - WALL, w + 2 * WALL, WALL),
        pygame.Rect(x - WALL, y + h, w + 2 * WALL, WALL),
        pygame.Rect(x - WALL, y, WALL, h),
        pygame.Rect(x + w, y, WALL, h),
    ]
    for _ in range(count):
        length = rng.randrange(70, 161, 10)
        px = rng.randrange(x + 100, x + w - 100, 20)
        py = rng.randrange(y + 90, y + h - 90, 20)
        horizontal = rng.random() < .5
        walls.append(pygame.Rect(px, py, length if horizontal else WALL,
                                 WALL if horizontal else length))
    return walls


def dodge_frame(tank, walls, bullets):
    """Encode every visible wall as local left/top/right/bottom coordinates."""
    x, y = tank.pos
    local_box = pygame.Rect(x - WALL_RADIUS, y - WALL_RADIUS,
                            2 * WALL_RADIUS, 2 * WALL_RADIUS)
    visible = [wall.clip(local_box) for wall in walls if wall.colliderect(local_box)]
    visible.sort(key=lambda wall: (max(wall.left - x, 0, x - wall.right) ** 2 +
                                   max(wall.top - y, 0, y - wall.bottom) ** 2,
                                   wall.left, wall.top, wall.right, wall.bottom))
    features = [math.sin(tank.angle), math.cos(tank.angle)]
    for wall in visible[:WALL_SLOTS]:
        features.extend(((wall.left - x) / WALL_RADIUS,
                         (wall.top - y) / WALL_RADIUS,
                         (wall.right - x) / WALL_RADIUS,
                         (wall.bottom - y) / WALL_RADIUS, 1.0))
    features.extend([0.0] * (WALL_SLOTS - min(len(visible), WALL_SLOTS)) * WALL_FEATURES)
    nearest = sorted(bullets, key=lambda bullet: tank.pos.distance_squared_to(bullet.pos))
    forward = unit(tank.angle)
    side = pygame.Vector2(-forward.y, forward.x)
    for bullet in nearest[:BULLET_SLOTS]:
        delta = bullet.pos - tank.pos
        angle = math.atan2(delta.y, delta.x) - tank.angle
        features.extend((min(delta.length() / BULLET_RADIUS, 1.0),
                         math.sin(angle), math.cos(angle),
                         bullet.velocity.dot(forward) / 440,
                         bullet.velocity.dot(side) / 440, 1.0))
    features.extend([0.0] * (BULLET_SLOTS - min(len(nearest), BULLET_SLOTS)) * 6)
    return np.asarray(features, dtype=np.float32)


class DodgeEnv:
    def __init__(self, seed=42, scenario='mixed'):
        self.rng = random.Random(seed)
        self.scenario_setting = scenario
        self.difficulty = 0.0
        self.walls = make_walls(self.rng, 3)
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
        self.scenario = (self.rng.choices(('standard', 'crossfire', 'corner'),
                                          weights=(4, 4, 2))[0]
                         if self.scenario_setting == 'mixed' else self.scenario_setting)
        low = 2 + round(4 * self.difficulty)
        high = 4 + round(6 * self.difficulty)
        if self.scenario == 'corner':
            low += 2
            high += 3
        self.walls = make_walls(self.rng, self.rng.randint(low, high))
        while True:
            pos = pygame.Vector2(self.rng.uniform(ARENA.left + 80, ARENA.right - 80),
                                 self.rng.uniform(ARENA.top + 80, ARENA.bottom - 80))
            if hits_wall(pos, 28, self.walls):
                continue
            if self.scenario == 'corner' and self.wall_clearance(pos) > 75:
                continue
            if self.scenario != 'corner' or self.wall_clearance(pos) >= 30:
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
        self.rebound_shots = 0
        self.last_death_bounces = None
        self.threat_steps = 0
        self.near_wall_steps = 0
        self.stuck_retreat_steps = 0
        if self.scenario == 'standard':
            self.fire_interval = self.rng.uniform(.55 - .2 * self.difficulty,
                                                  .8 - .2 * self.difficulty)
        else:
            self.fire_interval = self.rng.uniform(.48 - .1 * self.difficulty,
                                                  .68 - .1 * self.difficulty)
        current = self.frame()
        self.previous = current.copy()
        return np.concatenate((current, self.previous))

    def frame(self):
        return dodge_frame(self.tank, self.walls, self.bullets)

    def wall_clearance(self, position):
        return min(pygame.Vector2(max(wall.left - position.x, 0,
                                      position.x - wall.right),
                                  max(wall.top - position.y, 0,
                                      position.y - wall.bottom)).length()
                   for wall in self.walls)

    def fire(self):
        speed = self.rng.uniform(320, 440)
        if self.rng.random() < .2 + .5 * self.difficulty and self.fire_rebound(speed):
            return
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
                velocity = (target - origin).normalize() * speed
                self.bullets.append(Bullet(origin, velocity, -1, life=4.0))
                self.shots += 1
                self.direction_index = (index + 1) % 8
                return
        self.direction_index = (self.direction_index + 1) % 8

    def fire_rebound(self, speed):
        candidates = list(self.walls[4:])
        self.rng.shuffle(candidates)
        for wall in candidates:
            if wall.height > wall.width:
                normal = pygame.Vector2(-1 if self.tank.pos.x < wall.centerx else 1, 0)
                point = pygame.Vector2(wall.left - 6 if normal.x < 0 else wall.right + 6,
                                       max(wall.top + 8, min(self.tank.pos.y + self.rng.uniform(-30, 30), wall.bottom - 8)))
            else:
                normal = pygame.Vector2(0, -1 if self.tank.pos.y < wall.centery else 1)
                point = pygame.Vector2(max(wall.left + 8, min(self.tank.pos.x + self.rng.uniform(-30, 30), wall.right - 8)),
                                       wall.top - 6 if normal.y < 0 else wall.bottom + 6)
            outgoing = self.tank.pos - point
            if outgoing.length() < 35 or outgoing.length() > 230:
                continue
            incoming = outgoing.normalize().reflect(normal)
            origin = point - incoming * self.rng.uniform(90, 170)
            if not ARENA.collidepoint(origin) or hits_wall(origin, 7, self.walls):
                continue
            probe = Bullet(origin, incoming * speed, -1, life=4.0)
            for _ in range(32):
                if not probe.update(DT, self.walls):
                    break
                if probe.bounces >= 1 and probe.pos.distance_to(self.tank.pos) < 30:
                    self.bullets.append(Bullet(origin, incoming * speed, -1, life=4.0))
                    self.shots += 1
                    self.rebound_shots += 1
                    return True
        return False

    def step(self, action):
        forward, turn = ACTIONS[action]
        before = self.wall_clearance(self.tank.pos)
        old_pos = self.tank.pos.copy()
        self.tank.update(DT, forward, turn, self.walls)
        after = self.wall_clearance(self.tank.pos)
        self.fire_clock += DT
        if self.fire_clock >= self.fire_interval:
            self.fire_clock -= self.fire_interval
            burst = (1 if self.scenario == 'standard' else
                     2 if self.scenario == 'corner' else self.rng.choice((2, 3)))
            for _ in range(burst):
                self.fire()
        for bullet in self.bullets[:]:
            if not bullet.update(DT, self.walls):
                self.bullets.remove(bullet)
                continue
            if self.tank.pos.distance_to(bullet.pos) < 22:
                self.tank.alive = False
                self.last_death_bounces = bullet.bounces
        self.steps += 1
        if len(self.bullets) >= 2:
            self.threat_steps += 1
            self.near_wall_steps += after < 38
            self.stuck_retreat_steps += (forward == -1 and
                                         old_pos.distance_to(self.tank.pos) < .25)
        current = self.frame()
        state = np.concatenate((current, self.previous))
        self.previous = current
        terminated = not self.tank.alive
        truncated = self.steps >= MAX_STEPS and not terminated
        reward = DT if not terminated else 0.0
        if self.scenario != 'standard' and len(self.bullets) >= 2:
            reward += .04 * (min(after, 70) - min(before, 70)) / 70
            if after < 38:
                reward -= .008
            if forward and old_pos.distance_to(self.tank.pos) < .25:
                reward -= .012
        return state, reward, terminated, truncated


class DodgeNetwork:
    def __init__(self, rng, hidden=(96, 96)):
        sizes = (INPUT_SIZE, *hidden, len(ACTIONS))
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

    @classmethod
    def expand_from(cls, old, rng):
        """Embed a smaller policy exactly while leaving new units trainable."""
        net = cls(rng)
        for index, (source, target) in enumerate(zip(old.weights, net.weights)):
            if index == 0:
                target[:, :source.shape[1]] = source
            elif index == len(net.weights) - 1:
                target[:source.shape[0], :] = source
                target[source.shape[0]:, :] = 0
            else:
                target[:source.shape[0], :source.shape[1]] = source
                target[source.shape[0]:, :source.shape[1]] = 0
        for source, target in zip(old.biases, net.biases):
            target[:len(source)] = source
        return net

    def restore_base(self, old):
        """Keep the proven 64-unit policy while training the added units."""
        width = old.weights[0].shape[1]
        self.weights[0][:, :width] = old.weights[0]
        self.weights[1][:width, :width] = old.weights[1]
        self.weights[1][width:, :width] = 0
        self.weights[2][:width, :] = old.weights[2]
        for target, source in zip(self.biases, old.biases):
            target[:len(source)] = source

    def save(self, path):
        np.savez_compressed(path, input_size=INPUT_SIZE,
                            **{f'w{i}': value for i, value in enumerate(self.weights)},
                            **{f'b{i}': value for i, value in enumerate(self.biases)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['input_size']) != INPUT_SIZE:
                raise ValueError('Dodging model input size mismatch')
            net = cls(np.random.default_rng(0),
                      hidden=(data['w0'].shape[1], data['w1'].shape[1]))
            for i in range(3):
                net.weights[i][:] = data[f'w{i}']
                net.biases[i][:] = data[f'b{i}']
        return net


def choose(net, state, epsilon=0):
    if epsilon and random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(net.predict(state[None, :])[0]))


def evaluate(net, games=100, seed=12345, scenario='standard'):
    times = []
    rebound_deaths = 0
    direct_deaths = 0
    rebound_shots = 0
    threat_steps = 0
    near_wall_steps = 0
    stuck_retreat_steps = 0
    for episode in range(games):
        env = DodgeEnv(seed + episode * 9973, scenario)
        env.difficulty = 1.0
        state = env.reset()
        while True:
            state, _, terminated, truncated = env.step(choose(net, state))
            if terminated or truncated:
                times.append(env.steps * DT)
                rebound_deaths += bool(env.last_death_bounces is not None and env.last_death_bounces > 0)
                direct_deaths += bool(env.last_death_bounces == 0)
                rebound_shots += env.rebound_shots
                threat_steps += env.threat_steps
                near_wall_steps += env.near_wall_steps
                stuck_retreat_steps += env.stuck_retreat_steps
                break
    return {'mean_seconds': round(sum(times) / len(times), 2),
            'full_survival': sum(time >= MAX_STEPS * DT for time in times),
            'rebound_deaths': rebound_deaths, 'direct_deaths': direct_deaths,
            'rebound_shots': rebound_shots,
            'near_wall_rate': round(near_wall_steps / max(1, threat_steps), 3),
            'stuck_retreat_rate': round(stuck_retreat_steps / max(1, threat_steps), 3)}


def validation_score(net, games=30, seed=10000):
    results = {scene: evaluate(net, games, seed, scene)
               for scene in ('standard', 'crossfire', 'corner')}
    survival = sum(weight * results[scene]['mean_seconds']
                for scene, weight in (('standard', .25), ('crossfire', .4),
                                      ('corner', .35)))
    score = (survival - .8 * results['corner']['near_wall_rate'] -
             .2 * results['crossfire']['near_wall_rate'])
    return score, results


def train(episodes, model_path, seed, eval_interval=200, resume=False,
          validation_games=30, bootstrap_old=True):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = DodgeEnv(seed)
    old = DodgeNetwork.load(AI15_MODEL) if bootstrap_old and AI15_MODEL.is_file() else None
    online = (DodgeNetwork.load(model_path) if resume else
              DodgeNetwork.expand_from(old, rng)
              if bootstrap_old and AI15_MODEL.is_file() else DodgeNetwork(rng))
    target = DodgeNetwork(rng)
    target.copy_from(online)
    replay = deque(maxlen=20000)
    total_steps = 0
    best_score, baseline = validation_score(online, validation_games)
    online.save(model_path)
    print(f'initial validation: {baseline}; score={best_score:.2f}', flush=True)
    durations = []
    latest = model_path.with_name(model_path.stem + '_latest.npz')
    for episode in range(1, episodes + 1):
        env.difficulty = (1.0 if resume or bootstrap_old else
                          min(1.0, episode / max(1, .7 * episodes)))
        state = env.reset()
        epsilon_start = .3 if resume or bootstrap_old else 1.0
        epsilon = max(.05, epsilon_start - (epsilon_start - .05) *
                      (episode - 1) / max(1, .8 * episodes))
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
                if old is not None:
                    online.restore_base(old)
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
            score, result = validation_score(online, validation_games)
            online.save(latest)
            if score > best_score + .01:
                best_score = score
                online.save(model_path)
            print(f'validation {episode}: {result}; '
                  f'best={best_score:.2f}s', flush=True)
    print(f'Saved best: {model_path}; latest: {latest}')


def watch(model_path, seed):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('坦克动荡 · 纯神经网络闪避训练')
    font = Game.load_font(25)
    small = Game.load_font(18)
    clock = pygame.time.Clock()
    env = DodgeEnv(seed)
    env.difficulty = 1.0
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
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate', type=int, default=0, metavar='GAMES')
    parser.add_argument('--scenario', choices=('standard', 'crossfire', 'corner',
                                                'mixed'), default='standard')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Continue training the best model at full difficulty')
    parser.add_argument('--scratch', action='store_true',
                        help='Start without embedding the previous dodge model')
    parser.add_argument('--validation-games', type=int, default=30)
    args = parser.parse_args()
    if args.watch:
        watch(args.model or AI15_MODEL, args.seed)
    elif args.evaluate:
        net = DodgeNetwork.load(args.model or AI15_MODEL)
        print(evaluate(net, args.evaluate, args.seed, args.scenario))
    else:
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        train(args.episodes, args.model or CANDIDATE_MODEL, args.seed,
              resume=args.resume, validation_games=args.validation_games,
              bootstrap_old=not args.scratch)


if __name__ == '__main__':
    main()
