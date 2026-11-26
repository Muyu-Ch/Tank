"""Visual demo of the hybrid tank agent. Run: python3 tank_demo.py"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pygame

from tank import BG, BLUE, HEIGHT, MUTED, TEXT, WIDTH, YELLOW
from tank_hybrid import HybridAgent
from tank_nn import MAP_SEED, MODEL, Network
from tankai import TankEnv


def make_env(fixed_map):
    env = TankEnv(fixed_map=fixed_map, map_seed=MAP_SEED, spawn_mode='anywhere')
    env.reset()
    return env


def draw(env, wins, losses, draws, fixed_map, using_network, paused, result):
    game = env.game
    screen = game.screen
    screen.fill(BG)
    game.maze.draw(screen)
    for bullet in game.bullets:
        bullet.draw(screen)
    for tank in game.tanks:
        tank.draw(screen)

    game.label(f'混合 AI  {wins}', game.font, YELLOW, (220, 43))
    game.label('VS', game.small, MUTED, (WIDTH // 2, 43))
    game.label(f'规则 AI  {losses}', game.font, BLUE, (900, 43))
    stage = '固定地图' if fixed_map else '随机地图'
    status = '神经网络接管移动' if using_network else '规划器寻路与瞄准'
    game.label(f'{stage}  ·  {status}', game.small, TEXT, (WIDTH // 2, 82))
    game.label('Esc 退出   P 暂停   N 下一局   F 切换地图', game.small, MUTED,
               (WIDTH // 2, HEIGHT - 13))
    if paused or result:
        veil = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((255, 255, 255, 180))
        screen.blit(veil, (0, 0))
        game.label('已暂停' if paused else result, game.big, TEXT,
                   (WIDTH // 2, HEIGHT // 2))
        if not paused:
            game.label(f'总计 {wins} 胜 / {losses} 负 / {draws} 平', game.font, TEXT,
                       (WIDTH // 2, HEIGHT // 2 + 64))
    pygame.display.flip()


def run(model_path, fixed_map, seed):
    random.seed(seed)
    network = Network.load(model_path)
    env = make_env(fixed_map)
    agent = HybridAgent(network)
    clock = pygame.time.Clock()
    wins = losses = draws = 0
    paused = False
    result = ''
    result_until = 0
    using_network = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_n:
                    env.reset()
                    agent.reset()
                    result = ''
                    using_network = False
                elif event.key == pygame.K_f:
                    fixed_map = not fixed_map
                    env = make_env(fixed_map)
                    agent.reset()
                    wins = losses = draws = 0
                    result = ''
                    using_network = False
                    paused = False
        if not running:
            break
        if not paused:
            if result:
                if pygame.time.get_ticks() >= result_until:
                    env.reset()
                    agent.reset()
                    result = ''
                    using_network = False
            else:
                before = agent.network_decisions
                action = agent.act(env.game)
                using_network = agent.network_decisions > before
                _, reward, terminated, truncated = env.step(action)
                if terminated or truncated:
                    if reward > 0:
                        wins += 1
                        result = '混合 AI 获胜'
                    elif reward < 0:
                        losses += 1
                        result = '规则 AI 获胜'
                    else:
                        draws += 1
                        result = '平局'
                    result_until = pygame.time.get_ticks() + 1000
        draw(env, wins, losses, draws, fixed_map, using_network, paused, result)
        clock.tick(30)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Watch the hybrid tank AI in Pygame')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--random-map', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f'Model not found: {args.model}. Train tank_nn.py first.')
    run(args.model, fixed_map=not args.random_map, seed=args.seed)


if __name__ == '__main__':
    main()
