"""Hybrid tank agent: trained DQN for imminent evasive movement, planner for navigation and safe fire."""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pygame

from tank import cell_center, grid_cell, hits_wall
from tankai import ACTIONS, DT, TankEnv
from tank_nn import MAP_SEED, MODEL, Network, observation


class HybridAgent:
    def __init__(self, network, use_network=True, player_index=0):
        self.network = network
        self.use_network = use_network
        self.player_index = player_index
        self.shot_timer = 0.0
        self.network_decisions = 0

    def reset(self):
        self.shot_timer = 0.0
        self.network_decisions = 0

    @staticmethod
    def imminent_bullet(game, player_index=0):
        player = game.tanks[player_index]
        for bullet in game.bullets:
            if bullet.owner == player_index:
                continue
            displacement = player.pos - bullet.pos
            velocity = bullet.velocity
            time_to_nearest = max(0.0, min(.4, displacement.dot(velocity) /
                                           velocity.length_squared()))
            if time_to_nearest > 0 and (displacement - velocity * time_to_nearest).length() < 35:
                return True
        return False

    def act(self, game, dt=DT):
        self.shot_timer -= dt
        player = game.tanks[self.player_index]
        enemy = game.tanks[1 - self.player_index]
        ray = enemy.pos - player.pos
        clear = all(not hits_wall(player.pos + ray * (i / 24), 5, game.maze.walls)
                    for i in range(2, 24))
        route = game.maze.route(grid_cell(player.pos), grid_cell(enemy.pos))
        destination = (enemy.pos if len(route) <= 2 or clear else cell_center(route[1]))
        direction = destination - player.pos
        difference = (math.atan2(direction.y, direction.x) - player.angle + math.pi) % (2 * math.pi) - math.pi
        turn = 0 if abs(difference) < .05 else (1 if difference > 0 else -1)
        forward = 1 if abs(difference) < .85 else 0

        aim = (math.atan2(ray.y, ray.x) - player.angle + math.pi) % (2 * math.pi) - math.pi
        aim_limit = min(.22, math.atan2(18, max(1, ray.length())))
        shoot = abs(aim) < aim_limit and clear and self.shot_timer <= 0
        if shoot:
            self.shot_timer = .35

        if self.use_network and self.imminent_bullet(game, self.player_index):
            values = self.network.predict(observation(game, self.player_index)[None, :])[0]
            forward, turn, _ = ACTIONS[int(np.argmax(values))]
            self.network_decisions += 1
        return ACTIONS.index((forward, turn, shoot))


def evaluate(network, games=200, seed=2026, fixed_map=True, use_network=True):
    random.seed(seed)
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED,
                  spawn_mode='anywhere', spawn_seed=seed)
    agent = HybridAgent(network, use_network=use_network)
    outcomes = [0, 0, 0]
    self_hits = 0
    network_decisions = 0
    for _ in range(games):
        env.reset()
        agent.reset()
        while True:
            _, reward, terminated, truncated = env.step(agent.act(env.game))
            if terminated or truncated:
                outcomes[0 if reward > 0 else 1 if reward < 0 else 2] += 1
                self_hits += bool(env.last_hit and env.last_hit[0] == env.last_hit[1] == 0)
                network_decisions += agent.network_decisions
                break
    return {'win': outcomes[0], 'loss': outcomes[1], 'draw': outcomes[2],
            'self_hits': self_hits, 'network_decisions': network_decisions}


def watch(network, fixed_map=True):
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED, spawn_mode='anywhere')
    agent = HybridAgent(network)
    env.reset()
    clock = pygame.time.Clock()
    while True:
        if any(event.type == pygame.QUIT or
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
               for event in pygame.event.get()):
            break
        _, reward, terminated, truncated = env.step(agent.act(env.game))
        env.game.draw()
        if terminated or truncated:
            print('win' if reward > 0 else 'loss' if reward < 0 else 'draw')
            pygame.time.wait(500)
            env.reset()
            agent.reset()
        clock.tick(30)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Evaluate or watch the hybrid tank agent')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--evaluate', type=int, default=0, metavar='GAMES')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--random-map', action='store_true')
    parser.add_argument('--planner-only', action='store_true', help='Disable neural evasion for comparison')
    args = parser.parse_args()
    if args.evaluate < 0:
        parser.error('--evaluate must be nonnegative')
    network = Network.load(args.model)
    if args.evaluate:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        print(evaluate(network, args.evaluate, args.seed,
                       fixed_map=not args.random_map, use_network=not args.planner_only))
        pygame.quit()
    else:
        watch(network, fixed_map=not args.random_map)


if __name__ == '__main__':
    main()
