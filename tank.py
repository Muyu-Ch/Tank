"""Tank Trouble inspired local Pygame game. Run with: python3 tank.py"""
from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pygame

WIDTH, HEIGHT = 1120, 760
CELL, COLS, ROWS = 88, 12, 7
OX, OY = 32, 112
FIELD_W, FIELD_H = CELL * COLS, CELL * ROWS
FPS = 60
WALL = 10
BG = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
TEXT = (0, 0, 0)
MUTED = (95, 95, 95)
YELLOW = (255, 194, 73)
BLUE = (71, 185, 255)


def cell_center(cell):
    col, row = cell
    return pygame.Vector2(OX + (col + .5) * CELL, OY + (row + .5) * CELL)


def unit(angle):
    return pygame.Vector2(math.cos(angle), math.sin(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


class Maze:
    def __init__(self, rng=None):
        rng = rng or random
        self.openings = {(x, y): set() for y in range(ROWS) for x in range(COLS)}
        seen = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            x, y = stack[-1]
            choices = []
            for dx, dy, name, reverse in ((1, 0, 'E', 'W'), (-1, 0, 'W', 'E'),
                                          (0, 1, 'S', 'N'), (0, -1, 'N', 'S')):
                nxt = (x + dx, y + dy)
                if 0 <= nxt[0] < COLS and 0 <= nxt[1] < ROWS and nxt not in seen:
                    choices.append((nxt, name, reverse))
            if not choices:
                stack.pop()
                continue
            nxt, name, reverse = rng.choice(choices)
            self.openings[(x, y)].add(name)
            self.openings[nxt].add(reverse)
            seen.add(nxt)
            stack.append(nxt)
        # Add loops so players have multiple routes through the arena.
        for _ in range(20):
            x, y = rng.randrange(COLS), rng.randrange(ROWS)
            directions = []
            if x + 1 < COLS:
                directions.append(((x + 1, y), 'E', 'W'))
            if y + 1 < ROWS:
                directions.append(((x, y + 1), 'S', 'N'))
            if directions:
                nxt, name, reverse = rng.choice(directions)
                self.openings[(x, y)].add(name)
                self.openings[nxt].add(reverse)
        self.walls = []
        for y in range(ROWS):
            for x in range(COLS):
                px, py = OX + x * CELL, OY + y * CELL
                if 'E' not in self.openings[(x, y)]:
                    self.walls.append(pygame.Rect(px + CELL - WALL // 2, py - WALL // 2,
                                                  WALL, CELL + WALL))
                if 'S' not in self.openings[(x, y)]:
                    self.walls.append(pygame.Rect(px - WALL // 2, py + CELL - WALL // 2,
                                                  CELL + WALL, WALL))
        self.walls += [
            pygame.Rect(OX - WALL, OY - WALL, FIELD_W + 2 * WALL, WALL),
            pygame.Rect(OX - WALL, OY + FIELD_H, FIELD_W + 2 * WALL, WALL),
            pygame.Rect(OX - WALL, OY, WALL, FIELD_H),
            pygame.Rect(OX + FIELD_W, OY, WALL, FIELD_H),
        ]

    def route(self, start, goal):
        if start == goal:
            return [start]
        queue = deque([start])
        parent = {start: None}
        offsets = {'E': (1, 0), 'W': (-1, 0), 'S': (0, 1), 'N': (0, -1)}
        while queue:
            here = queue.popleft()
            for direction in ('E', 'W', 'S', 'N'):
                if direction not in self.openings[here]:
                    continue
                dx, dy = offsets[direction]
                nxt = (here[0] + dx, here[1] + dy)
                if nxt not in parent:
                    parent[nxt] = here
                    if nxt == goal:
                        path = [goal]
                        while path[-1] != start:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    queue.append(nxt)
        return [start]

    def draw(self, screen):
        for rect in self.walls:
            pygame.draw.rect(screen, WALL_COLOR, rect)


def hits_wall(pos, radius, walls):
    for rect in walls:
        nearest_x = clamp(pos.x, rect.left, rect.right)
        nearest_y = clamp(pos.y, rect.top, rect.bottom)
        if (pos.x - nearest_x) ** 2 + (pos.y - nearest_y) ** 2 < radius ** 2:
            return True
    return False


def grid_cell(pos):
    return (clamp(int((pos.x - OX) / CELL), 0, COLS - 1),
            clamp(int((pos.y - OY) / CELL), 0, ROWS - 1))


@dataclass
class Tank:
    pos: pygame.Vector2
    angle: float
    color: tuple
    name: str
    score: int = 0
    cooldown: float = 0
    alive: bool = True

    def update(self, dt, forward, turn, walls):
        if not self.alive:
            return
        self.angle += turn * 2.85 * dt
        velocity = unit(self.angle) * forward * 185 * dt
        for axis in ('x', 'y'):
            candidate = self.pos.copy()
            setattr(candidate, axis, getattr(candidate, axis) + getattr(velocity, axis))
            if not hits_wall(candidate, 17, walls):
                self.pos = candidate
        self.cooldown = max(0, self.cooldown - dt)

    def draw(self, screen):
        if not self.alive:
            return
        direction = unit(self.angle)
        side = pygame.Vector2(-direction.y, direction.x)
        center = self.pos
        body = [center + direction * 18 + side * 14,
                center + direction * 18 - side * 14,
                center - direction * 18 - side * 14,
                center - direction * 18 + side * 14]
        barrel = [center + direction * 8 + side * 5,
                  center + direction * 31 + side * 5,
                  center + direction * 31 - side * 5,
                  center + direction * 8 - side * 5]
        # The body covers the base of the barrel, so it always appears attached
        # to the front of the tank at every rotation angle.
        pygame.draw.polygon(screen, self.color, barrel)
        pygame.draw.polygon(screen, TEXT, barrel, 2)
        pygame.draw.polygon(screen, self.color, body)
        pygame.draw.polygon(screen, TEXT, body, 2)
        pygame.draw.circle(screen, TEXT, center, 8)
        pygame.draw.circle(screen, self.color, center, 5)


@dataclass
class Bullet:
    pos: pygame.Vector2
    velocity: pygame.Vector2
    owner: int
    life: float = 7.0
    bounces: int = 0
    age: float = 0

    def update(self, dt, walls):
        self.life -= dt
        self.age += dt
        # Small steps prevent fast shots from passing through thin walls.
        steps = max(1, math.ceil(self.velocity.length() * dt / 4))
        step = dt / steps
        for _ in range(steps):
            next_x = self.pos + pygame.Vector2(self.velocity.x * step, 0)
            if hits_wall(next_x, 5, walls):
                self.velocity.x *= -1
                self.bounces += 1
            else:
                self.pos = next_x
            next_y = self.pos + pygame.Vector2(0, self.velocity.y * step)
            if hits_wall(next_y, 5, walls):
                self.velocity.y *= -1
                self.bounces += 1
            else:
                self.pos = next_y
        return self.life > 0 and self.bounces <= 9

    def draw(self, screen):
        pygame.draw.circle(screen, TEXT, self.pos, 5)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('坦克动荡 · Tank Trouble')
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = self.load_font(27)
        self.small = self.load_font(19)
        self.big = self.load_font(58)
        self.mode = 'menu'
        self.solo = False
        self.strong_ai = False
        self.hybrid_ai = None
        self.ai_decision_clock = 0.0
        self.ai_decision = (0, 0, False)
        self.menu_error = ''
        self.maze = None
        self.tanks = []
        self.bullets = []
        self.round_delay = 0
        self.message = ''
        self.ai_shoot_timer = 0
        self.last_hit = None

    @staticmethod
    def load_font(size):
        font_files = [
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            'C:/Windows/Fonts/msyh.ttc',
        ]
        for font_file in font_files:
            if Path(font_file).is_file():
                return pygame.font.Font(font_file, size)
        candidates = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
        for name in candidates:
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont(None, size)

    def start(self, solo, strong=False):
        if strong:
            # When this file is run directly, the AI modules should reuse its
            # existing Game and Tank classes rather than import a second copy.
            sys.modules.setdefault('tank', sys.modules[__name__])
            try:
                from tank_dodge_walls import OLD_MODEL, DodgeNetwork, dodge_frame
                from tank_ai import ACTIONS, TankAI15
                if not OLD_MODEL.is_file():
                    raise FileNotFoundError(OLD_MODEL)
                self.hybrid_ai = TankAI15(DodgeNetwork.load(OLD_MODEL),
                                          player_index=1, frame_fn=dodge_frame)
                self.strong_actions = ACTIONS
            except (ImportError, FileNotFoundError, ValueError) as error:
                self.menu_error = f'AI 1.5 无法启动：{error}'
                return
        else:
            self.hybrid_ai = None
        self.menu_error = ''
        self.solo = solo
        self.strong_ai = strong
        self.tanks = [Tank(pygame.Vector2(), 0, YELLOW, 'P1'),
                      Tank(pygame.Vector2(), math.pi, BLUE,
                           'AI 1.5' if strong else 'AI' if solo else 'P2')]
        self.new_round()

    def new_round(self, maze=None, left=None, right=None):
        self.maze = maze if maze is not None else Maze()
        if self.strong_ai and left is None and right is None:
            cells = [(x, y) for y in range(ROWS) for x in range(COLS)]
            while True:
                left, right = random.sample(cells, 2)
                if abs(left[0] - right[0]) + abs(left[1] - right[1]) >= 6:
                    break
        left = left if left is not None else (1, random.randrange(ROWS))
        right = right if right is not None else (COLS - 2, random.randrange(ROWS))
        self.tanks[0].pos = cell_center(left)
        self.tanks[0].angle = 0
        self.tanks[1].pos = cell_center(right)
        self.tanks[1].angle = math.pi
        for tank in self.tanks:
            tank.alive = True
            tank.cooldown = 0
        self.bullets = []
        self.round_delay = 0
        self.message = ''
        self.mode = 'playing'
        self.ai_shoot_timer = 0
        self.ai_decision_clock = 0.0
        self.ai_decision = (0, 0, False)
        self.last_hit = None
        if self.hybrid_ai is not None:
            self.hybrid_ai.reset(self)

    def shoot(self, index):
        tank = self.tanks[index]
        if not tank.alive or tank.cooldown > 0 or sum(b.owner == index for b in self.bullets) >= 5:
            return
        direction = unit(tank.angle)
        muzzle = tank.pos + direction * 28
        if not hits_wall(muzzle, 5, self.maze.walls):
            self.bullets.append(Bullet(muzzle, direction * 420, index))
            tank.cooldown = .34

    def ai_update(self, dt):
        ai, target = self.tanks[1], self.tanks[0]
        route = self.maze.route(grid_cell(ai.pos), grid_cell(target.pos))
        destination = cell_center(route[1] if len(route) > 1 else route[0])
        if len(route) <= 2:
            destination = target.pos
        vector = destination - ai.pos
        desired = math.atan2(vector.y, vector.x)
        difference = (desired - ai.angle + math.pi) % (2 * math.pi) - math.pi
        turn = clamp(difference * 3, -1, 1)
        forward = 1 if abs(difference) < .85 else 0
        ai.update(dt, forward, turn, self.maze.walls)
        self.ai_shoot_timer -= dt
        toward_player = target.pos - ai.pos
        aim_difference = (math.atan2(toward_player.y, toward_player.x) - ai.angle + math.pi) % (2 * math.pi) - math.pi
        if abs(aim_difference) < .22 and self.ai_shoot_timer <= 0:
            # Ray sample ensures the AI fires only along a clear corridor.
            clear = all(not hits_wall(ai.pos + toward_player * (i / 24), 5, self.maze.walls)
                        for i in range(2, 24))
            if clear:
                self.shoot(1)
                self.ai_shoot_timer = .8

    def advance_bullets(self, dt):
        """Apply the same bullet and hit rules in the game and headless training."""
        self.last_hit = None
        for bullet in self.bullets[:]:
            if not bullet.update(dt, self.maze.walls):
                self.bullets.remove(bullet)
                continue
            for index, tank in enumerate(self.tanks):
                if tank.alive and (index != bullet.owner or bullet.age > .42) and tank.pos.distance_to(bullet.pos) < 20:
                    tank.alive = False
                    self.last_hit = (index, bullet.owner, bullet.bounces)
                    self.tanks[1 - index].score += 1
                    self.message = f'{self.tanks[1 - index].name}  得分！'
                    self.round_delay = 2.0
                    self.bullets.clear()
                    return self.last_hit
        return None

    def step_actions(self, actions, dt):
        """Advance one physics frame from two (forward, turn, shoot) actions."""
        for tank, (forward, turn, _) in zip(self.tanks, actions):
            tank.update(dt, forward, turn, self.maze.walls)
        for index, (_, _, shoot) in enumerate(actions):
            if shoot:
                self.shoot(index)
        return self.advance_bullets(dt)

    def update(self, dt):
        if self.mode != 'playing':
            return
        keys = pygame.key.get_pressed()
        if self.round_delay > 0:
            self.round_delay -= dt
            if self.round_delay <= 0:
                self.new_round()
            return
        player_action = (int(keys[pygame.K_w]) - int(keys[pygame.K_s]),
                         int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
                         bool(keys[pygame.K_SPACE]))
        if self.strong_ai:
            self.ai_decision_clock += dt
            if self.ai_decision_clock >= 1 / 30:
                self.ai_decision_clock -= 1 / 30
                self.ai_decision = self.strong_actions[self.hybrid_ai.act(self, 1 / 30)]
            self.step_actions((player_action, self.ai_decision), dt)
        elif self.solo:
            self.tanks[0].update(dt, player_action[0], player_action[1], self.maze.walls)
            self.ai_update(dt)
            if player_action[2]:
                self.shoot(0)
            self.advance_bullets(dt)
        else:
            second_action = (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN]),
                             int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
                             bool(keys[pygame.K_RETURN] or keys[pygame.K_KP_ENTER]))
            self.step_actions((player_action, second_action), dt)

    def label(self, text, font, color, center):
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=center))

    def draw(self):
        self.screen.fill(BG)
        if self.mode == 'menu':
            self.label('坦克动荡', self.big, TEXT, (WIDTH // 2, 165))
            pygame.draw.rect(self.screen, TEXT, (380, 314, 360, 62), 2)
            self.label('1   单人对战 AI', self.font, TEXT, (WIDTH // 2, 345))
            pygame.draw.rect(self.screen, TEXT, (380, 393, 360, 62), 2)
            self.label('2   本地双人对战', self.font, TEXT, (WIDTH // 2, 424))
            pygame.draw.rect(self.screen, TEXT, (380, 472, 360, 62), 2)
            self.label('3   挑战 AI 1.5', self.font, TEXT, (WIDTH // 2, 503))
            self.label('P1: W A S D 移动 · 空格射击    P2: 方向键移动 · 回车射击',
                       self.small, MUTED, (WIDTH // 2, 603))
            self.label('AI 1.5 模式每局生成随机地图', self.small, MUTED, (WIDTH // 2, 643))
            if self.menu_error:
                self.label(self.menu_error, self.small, TEXT, (WIDTH // 2, 690))
        else:
            self.maze.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            for tank in self.tanks:
                tank.draw(self.screen)
            self.label(f'{self.tanks[0].name}  {self.tanks[0].score}', self.font, TEXT, (440, 48))
            self.label('—', self.font, MUTED, (WIDTH // 2, 48))
            self.label(f'{self.tanks[1].name}  {self.tanks[1].score}', self.font, TEXT, (680, 48))
            self.label('Esc 暂停', self.small, MUTED, (1010, 48))
            if self.message:
                self.label(self.message, self.big, TEXT, (WIDTH // 2, HEIGHT // 2))
            if self.mode == 'paused':
                shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                shade.fill((255, 255, 255, 210))
                self.screen.blit(shade, (0, 0))
                self.label('已暂停', self.big, TEXT, (WIDTH // 2, 310))
                self.label('Esc 继续   ·   R 重开   ·   M 返回菜单', self.font, MUTED,
                           (WIDTH // 2, 385))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000, .05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.mode == 'menu':
                        if event.key == pygame.K_1:
                            self.start(True)
                        elif event.key == pygame.K_2:
                            self.start(False)
                        elif event.key == pygame.K_3:
                            self.start(True, strong=True)
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                    elif event.key == pygame.K_ESCAPE:
                        self.mode = 'paused' if self.mode == 'playing' else 'playing'
                    elif self.mode == 'paused':
                        if event.key == pygame.K_r:
                            for tank in self.tanks:
                                tank.score = 0
                            self.new_round()
                        elif event.key == pygame.K_m:
                            self.mode = 'menu'
            if self.mode == 'menu' and pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                if 380 <= mx <= 740 and 314 <= my <= 376:
                    self.start(True)
                elif 380 <= mx <= 740 and 393 <= my <= 455:
                    self.start(False)
                elif 380 <= mx <= 740 and 472 <= my <= 534:
                    self.start(True, strong=True)
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == '__main__':
    Game().run()
