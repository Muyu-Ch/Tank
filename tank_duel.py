"""Headless and visual evaluations of the current dodge AI."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pygame

from tank import COLS, ROWS, Game, Maze
from tank_ai import ACTIONS, DodgeAttackAgent, PlannerAgent, StableDodgeAttackAgent
from tank_dodge_walls import DodgeNetwork, MODEL

PHYSICS_DT = 1 / 60
MAX_DECISIONS = 600


def spawn_pair(rng):
    cells = [(x, y) for y in range(ROWS) for x in range(COLS)]
    while True:
        left, right = rng.sample(cells, 2)
        if abs(left[0] - right[0]) + abs(left[1] - right[1]) >= 6:
            return left, right


class DuelEnv:
    def __init__(self, network, seed=2026, agent_index=0, legacy=False,
                 planner_only=False):
        self.game = Game()
        self.game.start(False)
        self.rng = random.Random(seed)
        self.agent_index = agent_index
        self.dodger = (PlannerAgent(agent_index) if planner_only else
                       DodgeAttackAgent(network, agent_index) if legacy else
                       StableDodgeAttackAgent(network, agent_index))
        self.opponent = PlannerAgent(1 - agent_index)
        self.controllers = [None, None]
        self.controllers[agent_index] = self.dodger
        self.controllers[1 - agent_index] = self.opponent
        self.decisions = 0
        self.last_hit = None
        self.reset()

    def reset(self):
        left, right = spawn_pair(self.rng)
        self.game.new_round(maze=Maze(self.rng), left=left, right=right)
        self.game.tanks[self.agent_index].name = '闪避 AI'
        self.game.tanks[1 - self.agent_index].name = '规则 AI'
        self.dodger.reset(self.game)
        self.opponent.reset()
        self.decisions = 0
        self.last_hit = None

    def step(self):
        actions = [ACTIONS[controller.act(self.game)] for controller in self.controllers]
        for _ in range(2):
            self.last_hit = self.game.step_actions(actions, PHYSICS_DT)
            if self.last_hit:
                break
        self.decisions += 1
        if self.last_hit:
            return (1 if self.last_hit[0] != self.agent_index else -1), True
        return 0, self.decisions >= MAX_DECISIONS


def evaluate(network, games=100, seed=2026, legacy=False, planner_only=False):
    totals = {'win': 0, 'loss': 0, 'draw': 0, 'rebound_deaths': 0,
              'self_hits': 0, 'dodge_decisions': 0,
              'mode_switches': 0, 'movement_reversals': 0}
    for agent_index in (0, 1):
        env = DuelEnv(network, seed, agent_index, legacy, planner_only)
        for _ in range(games):
            env.reset()
            while True:
                outcome, done = env.step()
                if done:
                    totals['win' if outcome > 0 else 'loss' if outcome < 0 else 'draw'] += 1
                    if env.last_hit and env.last_hit[0] == agent_index:
                        totals['self_hits'] += env.last_hit[1] == agent_index
                        totals['rebound_deaths'] += env.last_hit[2] > 0
                    for key in ('dodge_decisions', 'mode_switches', 'movement_reversals'):
                        totals[key] += getattr(env.dodger, key, 0)
                    break
    return totals


def watch(network, seed=2026, agent_index=1, legacy=False,
          planner_only=False):
    env = DuelEnv(network, seed, agent_index, legacy, planner_only)
    clock = pygame.time.Clock()
    paused = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                paused = not paused
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                env.reset()
        if not paused:
            _, done = env.step()
            if done:
                env.reset()
        env.game.draw()
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description='Evaluate the current dodge AI')
    parser.add_argument('--games', type=int, default=100, help='Games per seat')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--swap', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--legacy', action='store_true',
                      help='Use immediate switching with the same network')
    mode.add_argument('--planner-only', action='store_true',
                      help='Use only rule-based movement and shooting')
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    if not args.watch:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    network = None if args.planner_only else DodgeNetwork.load(args.model)
    if args.watch:
        watch(network, args.seed, int(not args.swap), args.legacy,
              args.planner_only)
    else:
        print(evaluate(network, args.games, args.seed, args.legacy,
                       args.planner_only))
        pygame.quit()


if __name__ == '__main__':
    main()
