import pygame

SCREAN_WIDTH=800

SCREAN_LENGTH=1000

pygame.init()
screan = pygame.display.set_mode((SCREAN_WIDTH,SCREAN_LENGTH))

isRunning = True 

while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning=False
    pygame.display.flip()

