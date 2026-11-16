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
        body_points=[] 
        for i in (35,145,-145,-35):
            point=(
                self.x+self.size*math.cos(Pi*(i+self.angle)/180),
                self.y+self.size*math.sin(Pi*(i+self.angle)/180)
            ) 
            body_points.append(point)
        pygame.draw.polygon(screen,self.color,body_points,0)

        fire_points=[]
        fire_points.append((
            self.x+self.size*(0.5*math.cos(Pi*self.angle/180) + 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(0.5*math.sin(Pi*self.angle/180) - 0.2*math.cos(Pi*self.angle/180))
        ))
        fire_points.append((
            self.x+self.size*(0.5*math.cos(Pi*self.angle/180) - 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(0.5*math.sin(Pi*self.angle/180) + 0.2*math.cos(Pi*self.angle/180))
        ))
        fire_points.append((
            self.x+self.size*(1.5*math.cos(Pi*self.angle/180) - 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(1.5*math.sin(Pi*self.angle/180) + 0.2*math.cos(Pi*self.angle/180))
        ))
        fire_points.append((
            self.x+self.size*(1.5*math.cos(Pi*self.angle/180) + 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(1.5*math.sin(Pi*self.angle/180) - 0.2*math.cos(Pi*self.angle/180))
        ))       
        pygame.draw.polygon(screen,self.color,fire_points,0)
                
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

