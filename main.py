import argparse
import pygame
from audio import AudioDriver
from random import seed, shuffle
from SortingAlgorithms import get_algorithm, Action
from typing import Tuple
from dataclasses import dataclass

parser = argparse.ArgumentParser(description='Sorting Visualiser')
parser.add_argument('--name', type=str, default="BubbleSort", help='Algorithm Name')
parser.add_argument('--number', type=int,default=10, help='Max number of numbers to sort')

# Parse the arguments
args = parser.parse_args()

@dataclass(frozen=True)
class DisplayRect:
    x:int
    y:int
    width:int
    height:int
    colour:Tuple[int,int,int]

    def render(this, screen: pygame.SurfaceType,margin:int=1) -> None:
        pygame.draw.rect(screen, this.colour, (this.x, this.y, this.width - margin, this.height))





# generate a sequence and scramble it

seed(47)
my_sequence = list(range(1,args.number+1))
shuffle(my_sequence)

sort = get_algorithm(args.name)(data=my_sequence)
sort_stepper = sort.make_step()

## audio stuff

low_freq = 220
high_freq = 880
sample_rate = 44100

minVal = min(my_sequence)
maxVal = max(my_sequence)

audio = AudioDriver(args.number,low_freq,high_freq,sample_rate)





# init gui

pygame.init()
# Set up the display
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height),pygame.RESIZABLE)
font = pygame.font.SysFont("Courier New", 25)

algorithm_name = sort.__class__.__name__

pygame.display.set_caption(algorithm_name)

running = True



headbar_size = 70

started = False
finished = False
finished_idx = 0
action = None
curr_time = 0


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_KP_ENTER, pygame.K_RETURN]:
                started=True


    w, h = screen.get_size()

    rect_max_height = h-headbar_size
    rect_width = int(w/len(sort))
    x_shift = (w - (rect_width * len(sort))) // 2

    # Fill the screen with black
    screen.fill((0, 0, 0))


    text = f"Comparisons: {sort.number_comparisons}     Swaps: {sort.number_swaps}"

    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.topleft=(10,10)
    screen.blit(text_surface, text_rect)

    text = algorithm_name

    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.topleft = (w-text_rect.width-10, 10)
    screen.blit(text_surface, text_rect)



    delta = pygame.time.get_ticks() - curr_time

    if (started):
        try:
            action = next(sort_stepper)
            audio.mix(sort[action[1]]-1, sort[action[2]]-1)

        except StopIteration:
            finished = True
            started = False
            action = None


    rectagles_to_render = [None] * len(sort)


    for i in range(len(sort)):

        hh = int((sort[i] - minVal+1) / (maxVal - minVal+1) * rect_max_height)
        xx = rect_width * i + x_shift
        yy = h-hh

        colour = (255,255,255)

        if (action is not None):

            if (action[3] is not None) and (action[3] == i):
                colour = (0,0,255)

            match (action[0]):
                case Action.COMPARE:
                    if (i==action[1]) or ((i==action[2])):
                        colour = (0,255,0)
                case Action.SWAP:
                    if (i == action[1]) or ((i == action[2])):
                        colour = (255, 0, 0)
        elif (finished==True):
            if (i<finished_idx):
                colour = (0,255,0)
            elif (i==finished_idx):
                colour = (255, 255, 0)


        rectagles_to_render[i] = DisplayRect(x=xx, y=yy,width=rect_width,height=hh, colour=colour)#.render(screen)


    if (action is not None) and (isinstance(action[3],tuple) ):
        left,right = action[3]

        DisplayRect(x=rectagles_to_render[left].x,
                    y=headbar_size,
                    width=rect_width * (right-left+1) ,
                    height=rect_max_height,
                    colour=(0,0,128)).render(screen)

    for r in rectagles_to_render:
        r.render(screen)




    if (finished):
        if (finished_idx<len(sort)):
            audio.mix(sort[finished_idx]-1)

            finished_idx += 1
        else:
            audio.stop()


    # if (audio is not None):
    #     sounddevice.play(audio,samplerate=sample_rate)
    #
    # else:
    #     sounddevice.stop()


    pygame.display.flip()
    curr_time = pygame.time.get_ticks()
    pygame.time.delay(50)


pygame.quit()