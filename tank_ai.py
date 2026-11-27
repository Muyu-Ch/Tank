"""Rule-based attack and trained neural dodge controllers for the current game."""
from __future__ import annotations

import math

import numpy as np
import pygame

from tank import Bullet, Tank, cell_center, grid_cell, hits_wall
from tank_dodge_walls import ACTIONS as DODGE_ACTIONS, dodge_frame

DT = 1 / 30
ACTIONS = [(forward, turn, shoot)
           for forward in (-1, 0, 1)
           for turn in (-1, 0, 1)
           for shoot in (False, True)]


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
