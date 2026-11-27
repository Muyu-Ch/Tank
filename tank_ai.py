"""Rule-based attack and trained neural dodge controllers for the current game."""
from __future__ import annotations

import math

import numpy as np
import pygame

from tank import COLS, ROWS, Bullet, Tank, cell_center, grid_cell, hits_wall, unit
from tank_dodge_walls import ACTIONS as DODGE_ACTIONS, dodge_frame

DT = 1 / 30
ACTIONS = [(forward, turn, shoot)
           for forward in (-1, 0, 1)
           for turn in (-1, 0, 1)
           for shoot in (False, True)]


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


class PlannerAgent:
    """Navigate toward the enemy and fire only through a clear corridor."""

    def __init__(self, player_index=0):
        self.player_index = player_index
        self.shot_timer = 0.0

    def reset(self, game=None):
        self.shot_timer = 0.0

    def act(self, game, dt=DT):
        self.shot_timer -= dt
        player = game.tanks[self.player_index]
        enemy = game.tanks[1 - self.player_index]
        ray = enemy.pos - player.pos
        clear = all(not hits_wall(player.pos + ray * (i / 24), 5, game.maze.walls)
                    for i in range(2, 24))
        route = game.maze.route(grid_cell(player.pos), grid_cell(enemy.pos))
        destination = enemy.pos if len(route) <= 2 or clear else cell_center(route[1])
        direction = destination - player.pos
        difference = (math.atan2(direction.y, direction.x) - player.angle + math.pi) % (2 * math.pi) - math.pi
        turn = 0 if abs(difference) < .05 else (1 if difference > 0 else -1)
        forward = 1 if abs(difference) < .85 else 0

        aim = (math.atan2(ray.y, ray.x) - player.angle + math.pi) % (2 * math.pi) - math.pi
        aim_limit = min(.22, math.atan2(18, max(1, ray.length())))
        shoot = abs(aim) < aim_limit and clear and self.shot_timer <= 0
        if shoot:
            self.shot_timer = .35
        return ACTIONS.index((forward, turn, shoot))


def imminent_bullet(game, player_index, direct_horizon=.3, direct_distance=30,
                    bounce_steps=9, bounce_distance=35):
    player = game.tanks[player_index]
    for bullet in game.bullets:
        if bullet.owner == player_index:
            continue
        displacement = player.pos - bullet.pos
        velocity = bullet.velocity
        time_to_nearest = max(0.0, min(direct_horizon,
                                       displacement.dot(velocity) / velocity.length_squared()))
        if (time_to_nearest > 0 and
                (displacement - velocity * time_to_nearest).length() < direct_distance):
            return True
        if displacement.length_squared() >= 300 ** 2:
            continue
        projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                           bullet.life, bullet.bounces, bullet.age)
        for _ in range(bounce_steps):
            if not projected.update(DT, game.maze.walls):
                break
            if (projected.bounces > bullet.bounces and
                    projected.pos.distance_to(player.pos) < bounce_distance):
                return True
    return False


class DodgeAttackAgent:
    """Immediate danger switching, retained as a comparison baseline."""

    def __init__(self, network, player_index, frame_fn=dodge_frame):
        self.network = network
        self.player_index = player_index
        self.frame_fn = frame_fn
        self.planner = PlannerAgent(player_index)
        self.previous = None
        self.dodge_decisions = 0
        self.mode_switches = 0
        self.movement_reversals = 0
        self._was_dodging = False
        self._last_motion = None

    def frame(self, game):
        enemy_bullets = [bullet for bullet in game.bullets if bullet.owner != self.player_index]
        return self.frame_fn(game.tanks[self.player_index], game.maze.walls, enemy_bullets)

    def reset(self, game):
        self.planner.reset()
        self.previous = self.frame(game)
        self.dodge_decisions = 0
        self.mode_switches = 0
        self.movement_reversals = 0
        self._was_dodging = False
        self._last_motion = None

    def finish_action(self, forward, turn, shoot, dodging):
        if dodging != self._was_dodging:
            self.mode_switches += 1
        if self._last_motion is not None:
            old_forward, old_turn = self._last_motion
            if forward * old_forward < 0 or turn * old_turn < 0:
                self.movement_reversals += 1
        self._was_dodging = dodging
        self._last_motion = (forward, turn)
        return ACTIONS.index((forward, turn, shoot))

    def act(self, game, dt=DT):
        forward, turn, shoot = ACTIONS[self.planner.act(game, dt)]
        current = self.frame(game)
        dodging = imminent_bullet(game, self.player_index)
        if dodging:
            state = np.concatenate((current, self.previous))
            index = int(np.argmax(self.network.predict(state[None, :])[0]))
            forward, turn = DODGE_ACTIONS[index]
            self.dodge_decisions += 1
        self.previous = current
        return self.finish_action(forward, turn, shoot, dodging)


class StableDodgeAttackAgent(DodgeAttackAgent):
    """Predict ricochets, hold dodge mode briefly, and reject unsafe actions."""

    def __init__(self, network, player_index, frame_fn=dodge_frame,
                 min_dodge_steps=12, clear_steps=4, bounce_steps=18,
                 bounce_distance=45, q_margin=.08, safety_filter=True):
        super().__init__(network, player_index, frame_fn)
        self.min_dodge_steps = min_dodge_steps
        self.clear_steps = clear_steps
        self.prediction_steps = bounce_steps
        self.prediction_distance = bounce_distance
        self.q_margin = q_margin
        self.safety_filter = safety_filter
        self.dodging = False
        self.minimum_remaining = 0
        self.clear_count = 0
        self.last_dodge_action = None
        self.safety_overrides = 0

    def reset(self, game):
        super().reset(game)
        self.dodging = False
        self.minimum_remaining = 0
        self.clear_count = 0
        self.last_dodge_action = None
        self.safety_overrides = 0

    @staticmethod
    def local_walls(walls, position, reach):
        radius = reach + 20
        area = pygame.Rect(position.x - radius, position.y - radius,
                           2 * radius, 2 * radius)
        return [wall for wall in walls if wall.colliderect(area)]

    def safe_values(self, game, values):
        player = game.tanks[self.player_index]
        walls = game.maze.walls
        nearby = [bullet for bullet in game.bullets
                  if bullet.pos.distance_squared_to(player.pos) < 250 ** 2 and
                  (bullet.owner != self.player_index or bullet.age + 8 * DT > .42)]
        if not nearby:
            return values
        trajectories = []
        for bullet in nearby:
            projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                               bullet.life, bullet.bounces, bullet.age)
            bullet_walls = self.local_walls(walls, bullet.pos,
                                           bullet.velocity.length() * 8 * DT)
            path = []
            for _ in range(8):
                if not projected.update(DT, bullet_walls):
                    break
                path.append((projected.pos.copy(),
                             projected.owner != self.player_index or projected.age > .42))
            trajectories.append(path)
        adjusted = values.copy()
        tank_walls = self.local_walls(walls, player.pos, 185 * 8 * DT)
        for index, (forward, turn) in enumerate(DODGE_ACTIONS):
            projected_tank = Tank(player.pos.copy(), player.angle, player.color, player.name)
            closest = 250.0
            for step in range(8):
                projected_tank.update(DT, forward, turn, tank_walls)
                for path in trajectories:
                    if step < len(path) and path[step][1]:
                        closest = min(closest, projected_tank.pos.distance_to(path[step][0]))
            if closest < 24:
                adjusted[index] -= 5.0
            elif closest < 50:
                adjusted[index] -= (50 - closest) / 26 * .4
        return adjusted

    def bounce_threat(self, game, forward, turn):
        player = game.tanks[self.player_index]
        walls = game.maze.walls
        nearby_bullets = [bullet for bullet in game.bullets
                          if bullet.owner != self.player_index and
                          bullet.pos.distance_squared_to(player.pos) < 350 ** 2]
        if not nearby_bullets:
            return False
        predicted_tank = Tank(player.pos.copy(), player.angle, player.color, player.name)
        tank_walls = self.local_walls(walls, player.pos,
                                      185 * self.prediction_steps * DT)
        planned_positions = []
        for _ in range(self.prediction_steps):
            predicted_tank.update(DT, forward, turn, tank_walls)
            planned_positions.append(predicted_tank.pos.copy())
        for bullet in nearby_bullets:
            projected = Bullet(bullet.pos.copy(), bullet.velocity.copy(), bullet.owner,
                               bullet.life, bullet.bounces, bullet.age)
            bullet_walls = self.local_walls(walls, bullet.pos,
                                           bullet.velocity.length() * self.prediction_steps * DT)
            for position in planned_positions:
                if not projected.update(DT, bullet_walls):
                    break
                if (projected.bounces > bullet.bounces and
                        projected.pos.distance_to(position) < self.prediction_distance):
                    return True
        return False

    def act(self, game, dt=DT):
        forward, turn, shoot = ACTIONS[self.planner.act(game, dt)]
        current = self.frame(game)
        danger = (imminent_bullet(game, self.player_index,
                                  direct_horizon=.3, direct_distance=30,
                                  bounce_steps=0) or
                  self.bounce_threat(game, forward, turn))
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
            if self.safety_filter:
                values = self.safe_values(game, values)
            index = int(np.argmax(values))
            if (self.last_dodge_action is not None and
                    values[self.last_dodge_action] >= values[index] - self.q_margin):
                index = self.last_dodge_action
            forward, turn = DODGE_ACTIONS[index]
            self.safety_overrides += self.safety_filter and index != raw_index
            self.last_dodge_action = index
            self.minimum_remaining = max(0, self.minimum_remaining - 1)
            self.dodge_decisions += 1
        self.previous = current
        return self.finish_action(forward, turn, shoot, self.dodging)


class TacticalPlannerAgent(PlannerAgent):
    """Find verified direct or single-bounce shots and useful firing cells."""

    def __init__(self, player_index=0):
        super().__init__(player_index)
        self.plan_clock = 0.0
        self.shot_plan = None
        self.firing_cell = None
        self.planned_from = None
        self.planned_enemy = None
        self.cell_cache = {}
        self.wall_index = None
        self.indexed_maze = None
        self.search_key = None
        self.search_candidates = []
        self.search_index = 0
        self.search_best = None

    def reset(self, game=None):
        super().reset(game)
        self.plan_clock = 0.0
        self.shot_plan = None
        self.firing_cell = None
        self.planned_from = None
        self.planned_enemy = None
        self.cell_cache = {}
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
        candidates = []
        for y in range(ROWS):
            for x in range(COLS):
                cell = (x, y)
                route = game.maze.route(here, cell)
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


class TankAI15(StableDodgeAttackAgent):
    """Tactical shooting with self-bullet aware predictive dodging."""

    def __init__(self, network, player_index, frame_fn=dodge_frame):
        super().__init__(network, player_index, frame_fn)
        self.planner = TacticalPlannerAgent(player_index)
        self.self_threats = 0
        self.crowd_entries = 0

    def reset(self, game):
        super().reset(game)
        self.self_threats = 0
        self.crowd_entries = 0

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
        return self.finish_action(forward, turn, shoot, self.dodging)
