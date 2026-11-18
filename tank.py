import pygame
import random
import math
import time
Pi=3.1415926

BLACK=(0,0,0)

RED=(255,0,0)

GREEN=(0,255,0)

BLUE=(0,0,255)

BULLET_COLOR=(100,100,100)

SCREAN_WIDTH=800

SCREAN_LENGTH=1000

class Field:
    def __init__(self):
        self.width=600
        self.length=800
        self.field_color=(255,255,255)

    def draw_field(self,screen):
        screen.fill(self.field_color)

    def draw_tanks(self,screen,tanks):
        for t in tanks:
            t.draw(screen)

class Tank:
    def __init__(self,color):
        self.x=200
        self.y=200
        self.color=color
        self.angle=0
        self.size=30
        self.speed=100 
        self.angle_speed=180

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
            self.x+self.size*(1.3*math.cos(Pi*self.angle/180) - 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(1.3*math.sin(Pi*self.angle/180) + 0.2*math.cos(Pi*self.angle/180))
        ))
        fire_points.append((
            self.x+self.size*(1.3*math.cos(Pi*self.angle/180) + 0.2*math.sin(Pi*self.angle/180)),
            self.y+self.size*(1.3*math.sin(Pi*self.angle/180) - 0.2*math.cos(Pi*self.angle/180))
        ))       
        pygame.draw.polygon(screen,self.color,fire_points,0)

    def forward(self,fps):
        self.x+=self.speed*math.cos(Pi*self.angle/180)/fps
        self.y+=self.speed*math.sin(Pi*self.angle/180)/fps

    def back(self,fps):
        self.x+=self.speed*math.cos(Pi*self.angle/180)/fps
        self.y+=self.speed*math.sin(Pi*self.angle/180)/fps

    def turn_left(self,fps):
        self.angle-=self.angle_speed/fps

    def turn_right(self,fps):
        self.angle+=self.angle_speed/fps

class Bullet:
    def __init__(self):
        self.x=300
        self.y=300
        self.size=10.0
    def draw(self,screen):
        pygame.draw.circle(screen,BULLET_COLOR,(self.x,self.y),self.size,0)

field=Field()
tank1=Tank(GREEN)
tank2=Tank(BLUE)
tanks=[tank1,tank2]

pygame.init()
screen = pygame.display.set_mode((SCREAN_LENGTH,SCREAN_WIDTH))
fps=60.0
isRunning = True 

while isRunning:
    field.draw_field(screen)
    field.draw_tanks(screen,tanks)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning=False
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]:
            tank1.forward(fps)
        if keys[pygame.K_s]:
            tank1.back(fps)
        if keys[pygame.K_a]:
            tank1.turn_left(fps)
        if keys[pygame.K_d]:
            tank1.turn_right(fps)

        if keys[pygame.K_UP]:
            tank2.forward(fps)
        if keys[pygame.K_DOWN]:
            tank2.back(fps)
        if keys[pygame.K_LEFT]:
            tank2.turn_left(fps)
        if keys[pygame.K_RIGHT]:
            tank2.turn_right(fps)

    pygame.display.flip()
    time.sleep(1/fps)


