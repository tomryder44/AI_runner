"""
Runner game inspired by google offline game.
NEAT algorithm used to learn to play.
"""

import os
import pygame
import sys
import neat
from random import randint
import numpy as np
from neat.math_util import softmax
from matplotlib import pyplot as plt


pygame.init() # initialise pygame modules

# background image, dimensions: 1920 x 1200
bg_size = (1920//2, 1200//4)
screen_width = bg_size[0]
screen_height = bg_size[1]
game_window = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('GAME')
bg = pygame.transform.scale(pygame.image.load('background.jpg'), bg_size)

# player running images, dimensions: 415 x 507
run_size = (415//12, 507//12)
character_run_images = [pygame.transform.scale(pygame.image.load(os.path.join("Run__00" + str(x) + ".png")), run_size) for x in range(0,10)]

# player jumping images, dimensions: 407 x 536
jump_size = (407//12, 536//12)
character_jump_images = [pygame.transform.scale(pygame.image.load(os.path.join("Jump__00" + str(x) + ".png")), jump_size) for x in range(0,10)]

# player sliding images, dimensions:394 x 389
slide_size = (394//11, 389//16)
character_slide_images = [pygame.transform.scale(pygame.image.load(os.path.join("Slide__00" + str(x) + ".png")), slide_size) for x in range(0,10)]

# dinosaur running images, dimensions: 680 x 472
dino_size = (680//10, 472//9)
dino_images = [pygame.transform.flip(pygame.transform.scale(pygame.image.load(os.path.join("Run (" + str(x) + ").png")), dino_size), True, False) for x in range(1,9)]

# aeroplane flying images, dimensions: 443 x 302
plane_size = (443//6, 302//9)
plane_images = [pygame.transform.flip(pygame.transform.scale(pygame.image.load(os.path.join("Fly (" + str(x) + ").png")), plane_size), True, False) for x in range(1,3)]

# other stuff
ground = 20 # height of ground from bottom of screen
stats_font = pygame.font.SysFont('couriernew', 30) # font for score etc.
# colours
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# 3 different plane heights
low = screen_height - ground - plane_size[1] - 5 # jump over
medium = screen_height - ground - 35 # jump over
high = screen_height - ground - run_size[1] - plane_size[1] + 1 # duck under
highest = screen_height - ground - run_size[1] - plane_size[1] * 1.5 # can run under
plane_heights = [low, medium, high, highest]

class Player(object):
    '''
    player class, jumps or ducks under obstacles
    '''
    
    def __init__(self):
        self.width = run_size[0] # character initialised as running
        self.height = run_size[1]
        self.x = 100
        self.y = screen_height - ground - self.height
        self.is_running = True
        self.is_jumping = False
        self.gravity = 3
        self.is_sliding = False
        self.hitbox = (self.x, self.y, self.width, self.height) # for collision detection
        # image index for each type of movement
        self.jump_image_idx = 0
        self.run_image_idx = 0
        self.slide_image_idx = 0
        
    def draw(self):
        # 10 images make up run, jump and slide sequences
        # each image shown for 3 frames
        # reset index to zero when image sequence is complete
        if self.run_image_idx > (3*9):
            self.run_image_idx = 0
        if self.jump_image_idx > (3*9):
            self.jump_image_idx = 0
        if self.slide_image_idx > (3*9):
            self.slide_image_idx = 0
        
        # draw player running
        if self.is_running:
            game_window.blit(character_run_images[self.run_image_idx//3], (self.x, self.y))
            self.run_image_idx += 1
            
        # draw player jumping
        elif self.is_jumping:
            game_window.blit(character_jump_images[self.jump_image_idx//3], (self.x, self.y))
            self.jump_image_idx += 1
            
        # draw player sliding
        elif self.is_sliding:
            game_window.blit(character_slide_images[self.slide_image_idx//3], (self.x, self.y))
            self.slide_image_idx += 1
            
        # draw the hitbox
        self.hitbox = (self.x, self.y, self.width, self.height) # adjust hitbox for movement type
        if show_hitbox:
            pygame.draw.rect(game_window, red, self.hitbox, 2)
        
    def jump(self):
        self.is_jumping = True
        self.is_running = False
        self.is_sliding = False
        self.velocity = -20 # initial velocity 
        self.width = jump_size[0] # set size for image size
        self.height = jump_size[1]
            
    def slide(self):
        self.is_sliding = True
        self.is_jumping = False
        self.is_running = False
        self.width = slide_size[0]
        self.height = slide_size[1]
        
    def run(self):
        self.is_running = True
        self.is_jumping = False
        self.is_sliding = False
        self.width = run_size[0]
        self.height = run_size[1]
    
    def move(self):
        # if running or sliding, adjust y for height of image 
        if self.is_running or self.is_sliding:
            self.y = screen_height - ground - self.height
        
        # if jumping, jump
        if self.is_jumping: 
            self.y += self.velocity # move player 
            self.velocity += self.gravity # update velocity
            
            # limit maximum y of player, end jump and return to running
            if self.y + self.height >= screen_height - ground:
                self.y = screen_height - ground - self.height
                self.is_jumping = False
                self.is_running = True
        
class Dinosaur(object):
    '''
    dinosaur class, runs at players
    '''
    
    def __init__(self, x):
        self.width = dino_size[0]
        self.height = dino_size[1]
        self.x = x 
        self.y = screen_height - ground - self.height + 5 # need +5 for dinosaur to run along ground 
        self.speed = 14
        self.run_image_idx = 0
        self.hitbox = (self.x + self.width*0.3, self.y, self.width-self.width*0.3, self.height-5)
        self.show_nearest_ob = False
    
    def draw(self):
        if self.run_image_idx > (3*7): # 8 dino images, each image shown for 3 frames
            self.run_image_idx = 0
        
        # draw dinosaur running
        game_window.blit(dino_images[self.run_image_idx//3], (self.x, self.y))
        self.run_image_idx += 1
        
        # show hitbox
        self.hitbox = (self.x + self.width*0.3, self.y, self.width-self.width*0.3, self.height - 5)
        if show_hitbox:
            pygame.draw.rect(game_window, red, self.hitbox, 2)
            
        # box to show obstacle is nearest_obstacle used for network inputs
        self.ob_box = (self.x + self.width*0.3 - 5, self.y - 5, self.width-self.width*0.3 + 10, self.height + 5)
        if self.show_nearest_ob: 
            pygame.draw.rect(game_window, blue, self.ob_box, 2)
        
        
class Plane(object):
    '''
    plane class, flies at players
    '''
    
    def __init__(self, x, y):
        self.width = plane_size[0]
        self.height = plane_size[1]
        self.x = x
        self.y = y
        self.speed = 14
        self.fly_image_idx = 0
        self.hitbox = (self.x, self.y, self.width, self.height)
        self.show_nearest_ob = False
        
    def draw(self):        
        if self.fly_image_idx > (3*1): # 2 dino images, each image shown for 3 frames
            self.fly_image_idx = 0
        
        # draw plane flying
        game_window.blit(plane_images[self.fly_image_idx//3], (self.x, self.y))
        self.fly_image_idx += 1

        # obstacle hitbox
        self.hitbox = (self.x, self.y, self.width, self.height)
        if show_hitbox:
            pygame.draw.rect(game_window, red, self.hitbox, 2)
          
        # box to show obstacle is nearest_obstacle used for network inputs
        self.ob_box = (self.x - 5, self.y - 5, self.width + 10, self.height + 10)
        if self.show_nearest_ob: 
            pygame.draw.rect(game_window, blue, self.ob_box, 2)
              
def detect_collision(ob1, ob2):
    ''' 
    checks if any corner of the hitbox of ob1 is in the hitbox of ob2
    '''

    # hitbox of object one 
    hitbox_1_x = (ob1.hitbox[0], ob1.hitbox[0] + ob1.hitbox[2]) # x coords
    hitbox_1_y = (ob1.hitbox[1], ob1.hitbox[1] + ob1.hitbox[3]) # y coords
    
    # hitbox of object two
    hitbox_2_x = (ob2.hitbox[0], ob2.hitbox[0] + ob2.hitbox[2]) # x coords
    hitbox_2_y = (ob2.hitbox[1], ob2.hitbox[1] + ob2.hitbox[3]) # y coords
    
    collision = False
    
    for i in range(2):
        for j in range(2):
            if (hitbox_1_x[i] > hitbox_2_x[0] and hitbox_1_x[i] < hitbox_2_x[1]
                and hitbox_1_y[j] > hitbox_2_y[0] and hitbox_1_y[j] < hitbox_2_y[1]):
                collision = True
                    
    return collision

def redraw(players, obstacles, score, high_score, generation):
    '''
    draws everything in game window
    '''

    # draw background
    game_window.blit(bg, (0,0))
    
    # score
    score_text = stats_font.render('Score: ' + str(score), 1, black)
    game_window.blit(score_text, (screen_width - 10 - score_text.get_width(), 30))
    
    # high score
    high_score_text = stats_font.render('High score: ' + str(round(high_score,2)), 1, black)
    game_window.blit(high_score_text, (screen_width - 10 - high_score_text.get_width(), 60))
    
    # generation
    gen_text = stats_font.render('Generation: ' + str(generation), 1, black)
    game_window.blit(gen_text, (10, 30))
    
    # ground
    pygame.draw.line(game_window, black, (ground, screen_height-ground), 
                     (screen_width-ground, screen_height-ground))
    
    # legend
    if show_legend:
        pygame.draw.rect(game_window, red, (350, 20, 30, 30), 2) 
        game_window.blit(stats_font.render('hitbox', 1, black), (390, 20))
        pygame.draw.rect(game_window, blue, (350, 60, 30, 30), 2) 
        game_window.blit(stats_font.render('nearest', 1, black), (390, 60))
    
    # draw all players
    for player in players:
        player.draw()
    
    # draw all obstacles (dinosaurs and planes)
    for obstacle in obstacles:
        obstacle.draw()
    
    pygame.display.update()


generation_high_scores = [] # high score for each gen, used for plotting
generation = 0
high_score = 0

def main(genomes, config):
    
    global generation, high_score, show_hitbox, show_legend
    generation += 1
    show_hitbox = True
    show_legend = True
    
    nets = [] # list of neural networks for each player
    ge = [] # list of genomes
    players = [] # list of players
    
    for _, genome in genomes: 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Player())
        genome.fitness = 0 # each genome starts with fitness of 0
        ge.append(genome)
    
    obstacles = [] # list to append dinosaurs and planes to
    # initialise first obstacles
    obstacles.append(Dinosaur(1000))
    obstacles.append(Dinosaur(1500))
    ob_idx = 0 # object index for nearest_obstacle    
    nearest_obstacle = obstacles[ob_idx] # nearest obstacle whose position etc is used as neural network inputs
    nearest_obstacle.show_nearest_ob = True # check correct obstacle is being used for inputs
 
    running = True
    clock = pygame.time.Clock()
    score = 0 # score is time alive 

    while running:
        
        clock.tick(30) # 30 fps
        
        # check hitboxes are working, turn off after 10s
        if score > 10:
            show_hitbox = False
            show_legend = False
            nearest_obstacle.show_nearest_ob = False
        
        # if quit button on game window is pressed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                sys.exit()
        
        # Adding new obstacles
        if len(players) > 0: # if there are players still alive 
            if players[-1].x > (nearest_obstacle.x + nearest_obstacle.width): # if players pass nearest_obstacle
                
                # first update nearest_obstacle
                ob_idx += 1
                nearest_obstacle = obstacles[ob_idx]
                if score < 10:
                    nearest_obstacle.show_nearest_ob = True
                    obstacles[ob_idx-1].show_nearest_ob = False
                
                # create new obstacle 
                i = randint(1,10)
                if i < 7:
                    obstacles.append(Dinosaur(obstacles[-1].x + 500 + 10*score + 
                                              randint(-100, 100))) # append new dinosaur after last obstacle in list
                else:
                    obstacles.append(Plane(obstacles[-1].x + 500 + 10*score + randint(-100, 100), plane_heights[randint(0,3)]))
                obstacles[-1].speed = obstacles[-2].speed # give correct speed to new obstacle
                

        else: # if all players are dead
            # plot generation vs score
            generation_high_scores.append(score)
            plt.plot(range(1, generation+1), generation_high_scores)
            plt.xlabel('Generation')
            plt.ylabel('Score')
            plt.show()
            break

        # collision checks with obstacles
        for i, player in enumerate(players):    
            # if collision between player and obstacle, remove player, and corresponding network and genome 
            if detect_collision(player, nearest_obstacle):
                players.pop(i)
                nets.pop(i)
                ge.pop(i)

        # update fitness of each player for staying alive
        for i, player in enumerate(players):
            ge[i].fitness += 1
        
        # make run, slide or jump decision
        for i, player in enumerate(players):
            if player.is_jumping is False: # need to finish jump before making decision
                
                # network inputs                
                input_vector = (nearest_obstacle.hitbox[0] - (player.hitbox[0] + player.hitbox[2]), # distance to obstacle
                                nearest_obstacle.speed*30, # speed of obstacle
                                nearest_obstacle.hitbox[1], # top of obstacle
                                nearest_obstacle.hitbox[1] + nearest_obstacle.hitbox[3], # bottom of obstacle
                                player.hitbox[1], # top of player
                                player.hitbox[1] + player.hitbox[3]) # bottom of player 
                
                output = nets[i].activate(input_vector) # raw output
                softmax_result = softmax(output) # probabilities
                class_output = np.argmax(((softmax_result / np.max(softmax_result)) == 1).astype(int)) # class decision
                    
                if class_output == 0:
                    player.run()
                elif class_output == 1:
                    player.slide()
                elif class_output == 2:
                    player.jump()
            player.move() # corresponds to movement decision 
           
        for i, obstacle in enumerate(obstacles):
            obstacle.x -= obstacle.speed # move obstacles
            obstacle.speed += 0.005 # increase speed each frame
            if obstacle.x < -1000: # delete obstacles when off screen
                obstacles.pop(i)
                ob_idx -= 1 # update index
                nearest_obstacle = obstacles[ob_idx] # update nearest_obstacle
                   
        # update score and high score
        score += 1/30 # score = time
        score = round(score,2)
        if score > high_score:
            high_score = score
        
        # draw everything 
        redraw(players, obstacles, score, high_score, generation)

# run the evolution       
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    
    pop = neat.Population(config)
    pop.run(main, 500) # run for 500 generations

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)