import pygame
import random

SCREAN_WIDTH=800

SCREAN_LENGTH=1000

class Field:
    def __init__(self):
        self.width=600
        self.length=800
        self.field_color=(255,255,255)

    def draw_field(self,screen):
        screen.fill(self.field_color)

    def draw_tanks(self,tanks):
        return 

class Tank:
    def __init__(self):
        self.x=0
        self.y=0
        self.color='black'
        self.angle=0

    def draw(self):
        return 
        
field=Field()
pygame.init()
screen = pygame.display.set_mode((SCREAN_LENGTH,SCREAN_WIDTH))
field.draw_field(screen)

isRunning = True 

while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning=False
    pygame.display.flip()

