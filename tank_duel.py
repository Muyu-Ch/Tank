"""Head-to-head: previous hybrid AI versus dodge-network AI with identical attack rules."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pygame

from tank import COLS, ROWS, Game, Maze
from tank_dodge import DodgeNetwork, MODEL as DODGE_MODEL
from tank_hybrid import DodgeAttackAgent, HybridAgent
from tank_nn import MAP_SEED, MODEL as HYBRID_MODEL, Network
from tankai import ACTIONS, DT

MAX_STEPS = 600


class DuelEnv:
    def __init__(self, hybrid_network, dodge_network, seed=2026,
                 strongest_index=0, fixed_map=True):
        self.game = Game()
        self.game.start(False)
        self.spawn_rng = random.Random(seed)
        self.fixed_maze = Maze(random.Random(MAP_SEED)) if fixed_map else None
        self.strongest_index = strongest_index
        self.strongest = HybridAgent(hybrid_network, player_index=strongest_index)
        self.dodger = DodgeAttackAgent(dodge_network, player_index=1 - strongest_index)
        self.controllers = [None, None]
        self.controllers[strongest_index] = self.strongest
        self.controllers[1 - strongest_index] = self.dodger
        self.steps = 0
        self.last_hits = {}
        self.reset()

    def reset(self):
        cells = [(x, y) for y in range(ROWS) for x in range(COLS)]
        while True:
            left, right = self.spawn_rng.sample(cells, 2)
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) >= 6:
                break
        self.game.new_round(maze=self.fixed_maze, left=left, right=right)
        self.game.tanks[self.strongest_index].name = '旧最强AI'
        self.game.tanks[1 - self.strongest_index].name = '闪避AI'
        self.strongest.reset()
        self.dodger.reset(self.game)
        self.steps = 0
        self.last_hits = {}

    def step(self):
        game = self.game
        decisions = [controller.act(game) for controller in self.controllers]
        for index, action in enumerate(decisions):
            forward, turn, _ = ACTIONS[action]
            game.tanks[index].update(DT, forward, turn, game.maze.walls)
        for index, action in enumerate(decisions):
            if ACTIONS[action][2]:
                game.shoot(index)
        hits = {}
        for bullet in game.bullets[:]:
            if not bullet.update(DT, game.maze.walls):
                game.bullets.remove(bullet)
                continue
            for index, tank in enumerate(game.tanks):
                if index != bullet.owner or bullet.age > .42:
                    if tank.pos.distance_to(bullet.pos) < 20:
                        hits.setdefault(index, bullet.owner)
        self.steps += 1
        self.last_hits = hits
        if hits or self.steps >= MAX_STEPS:
            for index in hits:
                game.tanks[index].alive = False
            if len(hits) == 2 or not hits:
                game.message = '平局'
                return 0, True
            dead_index = next(iter(hits))
            winner = 1 - dead_index
            game.message = f'{game.tanks[winner].name} 得分！'
            return (-1 if dead_index == self.strongest_index else 1), True
        return 0, False


def evaluate(hybrid_network, dodge_network, games=200, seed=2026, fixed_map=True):
    results = []
    for strongest_index in (0, 1):
        env = DuelEnv(hybrid_network, dodge_network, seed, strongest_index, fixed_map)
        scores = {'strongest': 0, 'dodger': 0, 'draw': 0,
                  'strongest_self_hits': 0, 'dodger_self_hits': 0}
        for _ in range(games):
            env.reset()
            while True:
                outcome, done = env.step()
                if done:
                    scores['strongest' if outcome > 0 else 'dodger' if outcome < 0 else 'draw'] += 1
                    for dead, owner in env.last_hits.items():
                        if dead == owner:
                            key = 'strongest_self_hits' if dead == strongest_index else 'dodger_self_hits'
                            scores[key] += 1
                    break
        scores['strongest_seat'] = strongest_index
        results.append(scores)
    return results


def watch(hybrid_network, dodge_network, seed=2026, strongest_index=0):
    env = DuelEnv(hybrid_network, dodge_network, seed, strongest_index)
    clock = pygame.time.Clock()
    paused = False
    result_until = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                env.reset()
                result_until = 0
        if not paused:
            if result_until and pygame.time.get_ticks() < result_until:
                pass
            elif result_until:
                env.reset()
                result_until = 0
            else:
                outcome, done = env.step()
                if done:
                    if outcome:
                        winner = env.strongest_index if outcome > 0 else 1 - env.strongest_index
                        env.game.tanks[winner].score += 1
                    result_until = pygame.time.get_ticks() + 900
        env.game.draw()
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description='Simulate previous hybrid AI versus new dodging AI')
    parser.add_argument('--games', type=int, default=200, help='Games per seat assignment')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--random-map', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--swap', action='store_true', help='Put the previous strongest AI in seat 2')
    parser.add_argument('--hybrid-model', type=Path, default=HYBRID_MODEL)
    parser.add_argument('--dodge-model', type=Path, default=DODGE_MODEL)
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    if not args.watch:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    hybrid_network = Network.load(args.hybrid_model)
    dodge_network = DodgeNetwork.load(args.dodge_model)
    if args.watch:
        watch(hybrid_network, dodge_network, args.seed, int(args.swap))
    else:
        for result in evaluate(hybrid_network, dodge_network, args.games,
                               args.seed, fixed_map=not args.random_map):
            print(result)
        pygame.quit()


if __name__ == '__main__':
    main()
