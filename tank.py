import pygame
import random
import math
Pi=3.1415926
BLACK=(0,0,0)

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
        self.x=200
        self.y=200
        self.color=BLACK
        self.angle=10
        self.size=50
    
    def draw(self,screen):
        points=[] 
        for i in (35,145,-145,-35):
            point=(
                self.x+self.size*math.cos(Pi*(i+self.angle)/180),
                self.y+self.size*math.sin(Pi*(i+self.angle)/180)
            ) 
            points.append(point)
        pygame.draw.polygon(screen,self.color,points,0)
                
field=Field()
tank=Tank()
pygame.init()
screen = pygame.display.set_mode((SCREAN_LENGTH,SCREAN_WIDTH))
field.draw_field(screen)
tank.draw(screen)
isRunning = True 

while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning=False
    pygame.display.flip()

