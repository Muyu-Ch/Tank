"""Manstein: tactical shooting, neural dodging, and fire-pressure navigation."""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from tank import CELL, COLS, ROWS, Bullet, Tank, cell_center, grid_cell, hits_wall, unit

DT = 1 / 30
MODEL_PATH = Path(__file__).with_name('manstein.npz')
WALL_RADIUS = 220
BULLET_RADIUS = 210
WALL_SLOTS = 24
BULLET_SLOTS = 6
FRAME_SIZE = 2 + WALL_SLOTS * 5 + BULLET_SLOTS * 6
INPUT_SIZE = FRAME_SIZE * 2
DODGE_ACTIONS = [(forward, turn)
                 for forward in (-1, 0, 1) for turn in (-1, 0, 1)]
ACTIONS = [(forward, turn, shoot)
           for forward in (-1, 0, 1)
           for turn in (-1, 0, 1)
           for shoot in (False, True)]


def dodge_frame(tank, walls, bullets):
    """Encode the visible wall rectangles and nearest bullets for the model."""
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
    features.extend([0.0] * (WALL_SLOTS - min(len(visible), WALL_SLOTS)) * 5)
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


class DodgeNetwork:
    """Inference-only loader for Manstein's trained dodge model."""

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    @classmethod
    def load(cls, path=MODEL_PATH):
        with np.load(path, allow_pickle=False) as data:
            if int(data['input_size']) != INPUT_SIZE:
                raise ValueError('闪避模型输入维度不匹配')
            weights = [data[f'w{i}'].astype(np.float32) for i in range(3)]
            biases = [data[f'b{i}'].astype(np.float32) for i in range(3)]
        sizes = [INPUT_SIZE, weights[0].shape[1], weights[1].shape[1],
                 len(DODGE_ACTIONS)]
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.shape != (sizes[i], sizes[i + 1]) or bias.shape != (sizes[i + 1],):
                raise ValueError('闪避模型参数维度不匹配')
        return cls(weights, biases)

    def predict(self, states):
        hidden = states
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.maximum(hidden @ weight + bias, 0)
        return hidden @ self.weights[-1] + self.biases[-1]


class WallIndex:
    """Nearby wall lookup with enough padding for one physics step."""

    def __init__(self, walls, cell_size=88):
        self.cell_size = cell_size
        self.cells = {}
        for wall in walls:
            for x in range((wall.left - 18) // cell_size,
                           (wall.right + 18) // cell_size + 1):
                for y in range((wall.top - 18) // cell_size,
                               (wall.bottom + 18) // cell_size + 1):
                    self.cells.setdefault((x, y), []).append(wall)

    def near(self, position):
        return self.cells.get((int(position.x // self.cell_size),
                               int(position.y // self.cell_size)), ())


class AttackPlanner:
    """Find verified direct or single-bounce shots and useful firing cells."""

    def __init__(self, player_index=0):
        self.player_index = player_index
        self.shot_timer = 0.0
        self.plan_clock = 0.0
        self.shot_plan = None
        self.firing_cell = None
        self.planned_from = None
        self.planned_enemy = None
        self.cell_cache = {}
        self.route_cache = {}
        self.wall_index = None
        self.indexed_maze = None
        self.search_key = None
        self.search_candidates = []
        self.search_index = 0
        self.search_best = None

    def reset(self, game=None):
        self.shot_timer = 0.0
        self.plan_clock = 0.0
        self.shot_plan = None
        self.firing_cell = None
        self.planned_from = None
        self.planned_enemy = None
        self.cell_cache = {}
        self.route_cache = {}
        self.wall_index = None
        self.indexed_maze = None
        self.search_key = None
        self.search_candidates = []
        self.search_index = 0
        self.search_best = None

    @staticmethod
    def candidate_angles(source, target, walls):
        direct = target - source
        angles = [math.atan2(direct.y, direct.x)]
        for wall in walls:
            for x in (wall.left - 5, wall.right + 5):
                if (source.x - x) * (target.x - x) <= 0:
                    continue
                mirrored_x = 2 * x - target.x
                denominator = mirrored_x - source.x
                if abs(denominator) < 1e-6:
                    continue
                part = (x - source.x) / denominator
                impact_y = source.y + part * (target.y - source.y)
                if 0 < part < 1 and wall.top + 5 <= impact_y <= wall.bottom - 5:
                    angles.append(math.atan2(target.y - source.y, denominator))
            for y in (wall.top - 5, wall.bottom + 5):
                if (source.y - y) * (target.y - y) <= 0:
                    continue
                mirrored_y = 2 * y - target.y
                denominator = mirrored_y - source.y
                if abs(denominator) < 1e-6:
                    continue
                part = (y - source.y) / denominator
                impact_x = source.x + part * (target.x - source.x)
                if 0 < part < 1 and wall.left + 5 <= impact_x <= wall.right - 5:
                    angles.append(math.atan2(mirrored_y - source.y,
                                             target.x - source.x))
        unique = []
        for angle in angles:
            if all(abs((angle - other + math.pi) % (2 * math.pi) - math.pi) > .012
                   for other in unique):
                unique.append(angle)
        return unique

    @staticmethod
    def simulate_shot(source, target, angle, walls, max_bounces=1,
                      wall_index=None):
        direction = unit(angle)
        muzzle = source + direction * 28
        if hits_wall(muzzle, 5, wall_index.near(muzzle) if wall_index else walls):
            return None
        bullet = Bullet(muzzle, direction * 420, -1)
        for frame in range(1, 151):
            nearby = wall_index.near(bullet.pos) if wall_index else walls
            if not bullet.update(1 / 60, nearby) or bullet.bounces > max_bounces:
                break
            if bullet.age > .42 and bullet.pos.distance_to(source) < 24:
                return None
            if bullet.pos.distance_to(target) < 20:
                return bullet.bounces, frame / 60
        return None

    def find_shot(self, source, target, walls):
        best = None
        for angle in self.candidate_angles(source, target, walls):
            result = self.simulate_shot(source, target, angle, walls,
                                        wall_index=self.wall_index)
            if result is None:
                continue
            bounces, time = result
            score = 1.0 - .23 * bounces - .08 * time
            if best is None or score > best[0]:
                best = (score, angle, bounces)
                if bounces == 0:
                    return best
        return best

    def firing_candidates(self, game, player, enemy):
        here = grid_cell(player.pos)
        if here not in self.route_cache:
            self.route_cache[here] = {
                (x, y): game.maze.route(here, (x, y))
                for y in range(ROWS) for x in range(COLS)
            }
        candidates = []
        for y in range(ROWS):
            for x in range(COLS):
                cell = (x, y)
                route = self.route_cache[here][cell]
                steps = len(route) - 1
                if steps > 5:
                    continue
                position = cell_center(cell)
                separation = position.distance_to(enemy.pos)
                if separation < 145:
                    continue
                travel = steps + player.pos.distance_to(position) / 88
                candidates.append((travel + abs(separation - 300) / 300, cell, route))
        candidates.sort(key=lambda item: item[0])
        return candidates[:18]

    def search_firing_cell(self, game, player, enemy):
        key = (grid_cell(player.pos), int(enemy.pos.x / 28),
               int(enemy.pos.y / 28))
        if key != self.search_key:
            self.search_key = key
            self.search_candidates = self.firing_candidates(game, player, enemy)
            self.search_index = 0
            self.search_best = None
            self.firing_cell = None
        if self.search_index >= len(self.search_candidates):
            return
        _, cell, route = self.search_candidates[self.search_index]
        self.search_index += 1
        cache_key = (cell, key[1], key[2])
        if cache_key not in self.cell_cache:
            self.cell_cache[cache_key] = self.find_shot(
                cell_center(cell), enemy.pos, game.maze.walls)
        shot = self.cell_cache[cache_key]
        if shot is not None:
            score = shot[0] - .08 * (len(route) - 1)
            if self.search_best is None or score > self.search_best:
                self.search_best = score
                self.firing_cell = cell

    def act(self, game, dt=DT):
        self.shot_timer -= dt
        self.plan_clock -= dt
        player = game.tanks[self.player_index]
        enemy = game.tanks[1 - self.player_index]
        if self.indexed_maze is not game.maze:
            self.wall_index = WallIndex(game.maze.walls)
            self.indexed_maze = game.maze
            self.search_key = None
            self.route_cache = {}
        if (self.plan_clock <= 0 or self.planned_from is None or
                player.pos.distance_to(self.planned_from) > 24 or
                enemy.pos.distance_to(self.planned_enemy) > 24):
            self.shot_plan = self.find_shot(player.pos, enemy.pos, game.maze.walls)
            self.planned_from = player.pos.copy()
            self.planned_enemy = enemy.pos.copy()
            self.plan_clock = .2
        if self.shot_plan is not None:
            _, angle, bounces = self.shot_plan
            difference = (angle - player.angle + math.pi) % (2 * math.pi) - math.pi
            turn = 0 if abs(difference) < .035 else (1 if difference > 0 else -1)
            distance = player.pos.distance_to(enemy.pos)
            forward = -1 if distance < 125 and abs(difference) < .3 else 0
            shoot = (abs(difference) < (.055 if bounces else .09) and
                     self.shot_timer <= 0 and player.cooldown <= 0 and
                     self.simulate_shot(player.pos, enemy.pos, player.angle,
                                        game.maze.walls,
                                        wall_index=self.wall_index) is not None)
            if shoot:
                self.shot_timer = .35
                forward = 0
                turn = 0
            return ACTIONS.index((forward, turn, shoot))
        self.search_firing_cell(game, player, enemy)
        here = grid_cell(player.pos)
        destination_cell = self.firing_cell or grid_cell(enemy.pos)
        route = game.maze.route(here, destination_cell)
        destination = cell_center(route[1] if len(route) > 1 else destination_cell)
        direction = destination - player.pos
        difference = (math.atan2(direction.y, direction.x) - player.angle + math.pi) % (2 * math.pi) - math.pi
        turn = 0 if abs(difference) < .05 else (1 if difference > 0 else -1)
        forward = 1 if abs(difference) < .85 and direction.length() > 14 else 0
        if self.firing_cell is None and player.pos.distance_to(enemy.pos) < 145:
            forward = 0
        return ACTIONS.index((forward, turn, False))


class FirePressureNavigator:
    """Choose an open route before an approaching volley reaches the tank."""

    def __init__(self, player_index):
        self.player_index = player_index
        self.reset()

    def reset(self):
        self.target = None
        self.pressure_center = None
        self.scan_cooldown = 0
        self.active_steps = 0
        self.target_hold = 0
        self.entries = 0

    def projected_fire(self, game, wall_index):
        player = game.tanks[self.player_index]
        candidates = []
        paths = []
        for bullet in game.bullets:
            initial = bullet.pos.distance_to(player.pos)
            if initial > 620 or (bullet.owner == self.player_index and
                                 bullet.age + 1.2 <= .42):
                continue
            projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                               bullet.life, bullet.bounces, bullet.age)
            path = []
            for _ in range(72):
                if not projected.update(1 / 60, wall_index.near(projected.pos)):
                    break
                path.append(None if bullet.owner == self.player_index and
                            projected.age <= .42 else projected.pos.copy())
            paths.append(path)
            distances = [(step, point.distance_to(player.pos))
                         for step, point in enumerate(path) if point is not None]
            if not distances:
                continue
            closest = min(distance for _, distance in distances)
            if closest > 245 or initial - closest < 40:
                continue
            entry = next((step for step, distance in distances if distance < 260), None)
            if entry is not None:
                candidates.append((entry, path[entry]))
        best = None
        for entry, point in candidates:
            group = [(other_entry, other_point)
                     for other_entry, other_point in candidates
                     if abs(entry - other_entry) <= 24 and
                     point.distance_to(other_point) < 230]
            if len(group) < 2:
                continue
            center = sum((position for _, position in group), pygame.Vector2()) / len(group)
            rank = len(group) / (12 + sum(step for step, _ in group) / len(group))
            if best is None or rank > best[0]:
                best = (rank, center)
        return (best[1] if best else None), paths

    @staticmethod
    def routes(maze, start):
        directions = {'E': (1, 0), 'W': (-1, 0),
                      'S': (0, 1), 'N': (0, -1)}
        routes = {start: (start,)}
        queue = deque([start])
        while queue:
            here = queue.popleft()
            if len(routes[here]) > 3:
                continue
            for direction in maze.openings[here]:
                dx, dy = directions[direction]
                next_cell = (here[0] + dx, here[1] + dy)
                if next_cell not in routes:
                    routes[next_cell] = routes[here] + (next_cell,)
                    queue.append(next_cell)
        return routes

    @staticmethod
    def route_risk(route, paths):
        risk = 0.0
        for hop, cell in enumerate(route[1:], 1):
            center = cell_center(cell)
            arrival = min(71, round((.18 + hop * CELL / 185) * 60))
            for path in paths:
                for step in (arrival - 6, arrival, arrival + 6):
                    if 0 <= step < len(path) and path[step] is not None:
                        distance = path[step].distance_to(center)
                        if distance < 85:
                            risk += (85 - distance) / 85 * 1.8
        return risk

    def choose_target(self, game, source, paths):
        player = game.tanks[self.player_index]
        enemy = game.tanks[1 - self.player_index]
        away = player.pos - source
        if away.length_squared() < 1:
            return None
        away = away.normalize()
        routes = self.routes(game.maze, grid_cell(player.pos))
        scored = []
        for cell, route in routes.items():
            if len(route) == 1:
                continue
            position = cell_center(cell)
            degree = len(game.maze.openings[cell])
            displacement = (position - player.pos).dot(away) / CELL
            score = (1.35 * min(displacement, 2.5) - .28 * (len(route) - 1) +
                     .45 * (degree - 2) - self.route_risk(route, paths))
            if degree <= 1:
                score -= 1.6
            enemy_distance = position.distance_to(enemy.pos)
            if enemy_distance < 130:
                score -= .6
            elif enemy_distance > 440:
                score -= .15 * (enemy_distance - 440) / CELL
            scored.append((score, cell))
        if not scored:
            return None
        score, cell = max(scored)
        if self.target in routes:
            old_score = next((value for value, candidate in scored
                              if candidate == self.target), -math.inf)
            if self.target_hold > 0 and old_score >= score - .8:
                return self.target
        return cell if score > -.5 else None

    def update(self, game, wall_index):
        self.scan_cooldown -= 1
        self.target_hold = max(0, self.target_hold - 1)
        self.active_steps = max(0, self.active_steps - 1)
        if self.scan_cooldown <= 0:
            self.scan_cooldown = 6
            source, paths = self.projected_fire(game, wall_index)
            if source is not None:
                target = self.choose_target(game, source, paths)
                if target is not None:
                    if self.active_steps == 0:
                        self.entries += 1
                    if target != self.target:
                        self.target_hold = 18
                    self.target = target
                    self.pressure_center = source
                    self.active_steps = 24
        if self.active_steps == 0:
            self.target = None
            self.pressure_center = None
        return self.waypoint(game)

    def waypoint(self, game):
        if self.target is None:
            return None
        player = game.tanks[self.player_index]
        route = game.maze.route(grid_cell(player.pos), self.target)
        if len(route) == 1:
            return cell_center(self.target)
        return cell_center(route[1])

    @staticmethod
    def move_toward(player, waypoint):
        direction = waypoint - player.pos
        difference = (math.atan2(direction.y, direction.x) - player.angle +
                      math.pi) % (2 * math.pi) - math.pi
        turn = 0 if abs(difference) < .05 else (1 if difference > 0 else -1)
        forward = 1 if abs(difference) < .85 and direction.length() > 14 else 0
        return forward, turn

    def guide_dodge(self, game, values, waypoint, wall_index):
        if waypoint is None:
            return values
        player = game.tanks[self.player_index]
        initial_distance = player.pos.distance_to(waypoint)
        adjusted = values.copy()
        for index, (forward, turn) in enumerate(DODGE_ACTIONS):
            projected = Tank(player.pos.copy(), player.angle, player.color, player.name)
            for _ in range(12):
                projected.update(1 / 60, forward, turn, wall_index.near(projected.pos))
            progress = (initial_distance - projected.pos.distance_to(waypoint)) / 25
            direction = waypoint - projected.pos
            alignment = (unit(projected.angle).dot(direction.normalize())
                         if direction.length_squared() > 1 else 1.0)
            adjusted[index] += .45 * max(-1, min(1, progress)) + .12 * alignment
        return adjusted


class MansteinAI:
    """Tactical shooting with self-bullet aware predictive dodging."""

    def __init__(self, network, player_index, frame_fn=dodge_frame,
                 pressure_navigation=True):
        self.network = network
        self.player_index = player_index
        self.frame_fn = frame_fn
        self.planner = AttackPlanner(player_index)
        self.pressure = (FirePressureNavigator(player_index)
                         if pressure_navigation else None)
        self.min_dodge_steps = 12
        self.clear_steps = 4
        self.q_margin = .08
        self.reset_state()

    def reset(self, game):
        self.planner.reset()
        if self.pressure is not None:
            self.pressure.reset()
        self.reset_state()
        self.previous = self.frame_fn(game.tanks[self.player_index],
                                      game.maze.walls, [])

    def reset_state(self):
        self.previous = None
        self.dodging = False
        self.minimum_remaining = 0
        self.clear_count = 0
        self.last_dodge_action = None
        self.safety_overrides = 0
        self.dodge_decisions = 0
        self.self_threats = 0
        self.crowd_entries = 0

    @staticmethod
    def local_walls(walls, position, reach):
        radius = reach + 20
        area = pygame.Rect(position.x - radius, position.y - radius,
                           2 * radius, 2 * radius)
        return [wall for wall in walls if wall.colliderect(area)]

    def safe_values(self, game, values):
        player = game.tanks[self.player_index]
        wall_index = self.planner.wall_index
        nearby = [bullet for bullet in game.bullets
                  if bullet.pos.distance_squared_to(player.pos) < 350 ** 2 and
                  (bullet.owner != self.player_index or bullet.age + 24 / 60 > .42)]
        if not nearby:
            return values
        trajectories = []
        for bullet in nearby:
            projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                               bullet.life, bullet.bounces, bullet.age)
            path = []
            for _ in range(24):
                if not projected.update(1 / 60,
                                        wall_index.near(projected.pos)):
                    break
                path.append((projected.pos.copy(),
                             projected.owner != self.player_index or projected.age > .42))
            trajectories.append(path)
        adjusted = values.copy()
        for index, (forward, turn) in enumerate(DODGE_ACTIONS):
            projected_tank = Tank(player.pos.copy(), player.angle, player.color, player.name)
            closest = 350.0
            collision_step = None
            for step in range(24):
                projected_tank.update(1 / 60, forward, turn,
                                      wall_index.near(projected_tank.pos))
                for path in trajectories:
                    if step < len(path) and path[step][1]:
                        distance = projected_tank.pos.distance_to(path[step][0])
                        closest = min(closest, distance)
                        if distance < 22 and collision_step is None:
                            collision_step = step
                if collision_step is not None:
                    break
            if collision_step is not None:
                adjusted[index] -= 5.0 - 2.0 * collision_step / 24
            elif closest < 50:
                adjusted[index] -= (50 - closest) / 28 * .4
        return adjusted

    def predicted_threats(self, game, forward, turn):
        player = game.tanks[self.player_index]
        tank = Tank(player.pos.copy(), player.angle, player.color, player.name)
        wall_index = self.planner.wall_index
        positions = []
        for _ in range(36):
            tank.update(1 / 60, forward, turn, wall_index.near(tank.pos))
            positions.append(tank.pos.copy())
        own_threats = []
        danger = False
        own_danger = False
        paths = []
        for bullet in game.bullets:
            if bullet.pos.distance_to(player.pos) > 400:
                continue
            projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                               bullet.life, bullet.bounces, bullet.age)
            closest = 999.0
            path = []
            for position in positions:
                if not projected.update(1 / 60,
                                        wall_index.near(projected.pos)):
                    break
                if bullet.owner == self.player_index and projected.age <= .42:
                    path.append(None)
                    continue
                path.append(projected.pos.copy())
                closest = min(closest, projected.pos.distance_to(position))
            paths.append((bullet, path))
            if closest < 48:
                danger = True
                if bullet.owner == self.player_index:
                    own_danger = True
            if bullet.owner == self.player_index and closest < 70:
                own_threats.append(bullet)
        crowds = []
        for step in (5, 11, 17, 23, 29, 35):
            approaching = []
            for bullet, path in paths:
                if len(path) <= step or path[step] is None:
                    continue
                future = path[step]
                distance = future.distance_to(positions[step])
                if (distance < 160 and
                        distance < bullet.pos.distance_to(player.pos) - 12):
                    approaching.append(future)
            for i, first in enumerate(approaching):
                group = [first] + [other for other in approaching[i + 1:]
                                   if first.distance_to(other) < 160]
                if len(group) >= 2:
                    center = sum(group, pygame.Vector2()) / len(group)
                    crowds.append((step, center, len(group)))
                    break
        return danger or bool(crowds), own_threats, own_danger, crowds

    def crowd_values(self, game, values, crowds):
        if not crowds:
            return values
        player = game.tanks[self.player_index]
        wall_index = self.planner.wall_index
        event_steps = {step for step, _, _ in crowds}
        adjusted = values.copy()
        for index, (forward, turn) in enumerate(DODGE_ACTIONS):
            projected = Tank(player.pos.copy(), player.angle, player.color,
                             player.name)
            positions = {}
            for step in range(max(event_steps) + 1):
                projected.update(1 / 60, forward, turn,
                                 wall_index.near(projected.pos))
                if step in event_steps:
                    positions[step] = projected.pos.copy()
            score = 0.0
            for step, center, count in crowds:
                distance = positions[step].distance_to(center)
                starting_distance = player.pos.distance_to(center)
                weight = 1 - .4 * step / 36
                score += weight * (min(distance, 220) - min(starting_distance, 220)) / 100
                if distance < 90:
                    score -= weight * (90 - distance) / 90
            adjusted[index] += .45 * score / len(crowds)
        return adjusted

    def wall_trap_values(self, game, values):
        player = game.tanks[self.player_index]
        walls = self.local_walls(game.maze.walls, player.pos, 95)
        if not walls:
            return values

        def clearance(position):
            return min(math.hypot(max(wall.left - position.x, 0,
                                      position.x - wall.right),
                                  max(wall.top - position.y, 0,
                                      position.y - wall.bottom))
                       for wall in walls)

        start = clearance(player.pos)
        if start > 80:
            return values
        adjusted = values.copy()
        for index, (forward, turn) in enumerate(DODGE_ACTIONS):
            if not forward:
                continue
            projected = Tank(player.pos.copy(), player.angle, player.color,
                             player.name)
            for _ in range(10):
                projected.update(1 / 60, forward, turn, walls)
            end = clearance(projected.pos)
            travel = projected.pos.distance_to(player.pos)
            if travel < 3:
                adjusted[index] -= .7
            if end < start - 5 and end < 55:
                adjusted[index] -= .15 + .3 * (start - end) / 55
        return adjusted

    def act(self, game, dt=DT):
        forward, turn, shoot = ACTIONS[self.planner.act(game, dt)]
        planned_shot = shoot
        waypoint = (self.pressure.update(game, self.planner.wall_index)
                    if self.pressure is not None else None)
        if waypoint is not None:
            forward, turn = self.pressure.move_toward(
                game.tanks[self.player_index], waypoint)
            shoot = shoot and turn == 0
            if planned_shot and not shoot:
                self.planner.shot_timer = 0.0
        danger, own_threats, own_danger, crowds = self.predicted_threats(
            game, forward, turn)
        if own_danger and not self.dodging:
            self.self_threats += 1
        if crowds and not self.dodging:
            self.crowd_entries += 1
        observed = [bullet for bullet in game.bullets
                    if bullet.owner != self.player_index or bullet in own_threats]
        current = self.frame_fn(game.tanks[self.player_index], game.maze.walls, observed)
        if danger:
            self.clear_count = 0
        elif self.dodging:
            self.clear_count += 1
        if not self.dodging and danger:
            self.dodging = True
            self.minimum_remaining = self.min_dodge_steps
        elif self.dodging and self.minimum_remaining == 0 and self.clear_count >= self.clear_steps:
            self.dodging = False
            self.last_dodge_action = None
        if self.dodging:
            state = np.concatenate((current, self.previous))
            values = self.network.predict(state[None, :])[0]
            raw_index = int(np.argmax(values))
            values = self.safe_values(game, values)
            values = self.crowd_values(game, values, crowds)
            values = self.wall_trap_values(game, values)
            if waypoint is not None:
                values = self.pressure.guide_dodge(
                    game, values, waypoint, self.planner.wall_index)
            index = int(np.argmax(values))
            if (self.last_dodge_action is not None and
                    values[self.last_dodge_action] >= values[index] - self.q_margin):
                index = self.last_dodge_action
            forward, turn = DODGE_ACTIONS[index]
            self.safety_overrides += index != raw_index
            self.last_dodge_action = index
            self.minimum_remaining = max(0, self.minimum_remaining - 1)
            self.dodge_decisions += 1
        self.previous = current
        return ACTIONS.index((forward, turn, shoot))
