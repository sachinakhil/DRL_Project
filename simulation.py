import pygame
import random
import sys
import numpy as np
from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)

# Define screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60

# Car dimensions
CAR_WIDTH = 30
CAR_HEIGHT = 15

class Car:
    def __init__(self, x, y, direction, speed):
        self.x = x
        self.y = y
        self.direction = direction  # 'NS' or 'EW'
        self.speed = speed

    def move(self):
        if self.direction == 'NS':
            self.y += self.speed  # Move vertically
        elif self.direction == 'EW':
            self.x += self.speed  # Move horizontally

class TrafficSimulation:
    def __init__(self, env, agent1, agent2):
        self.env = env
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic Control Simulation")
        self.clock = pygame.time.Clock()

        self.cars_NS = []  # List of cars moving in the North-South direction
        self.cars_EW = []  # List of cars moving in the East-West direction
        self.agent1 = agent1  # For controlling NS traffic lights
        self.agent2 = agent2  # For controlling EW traffic lights

    def draw_intersection(self):
        # Draw roads
        pygame.draw.rect(self.screen, GRAY, (200, 0, 200, 600))  # Vertical road
        pygame.draw.rect(self.screen, GRAY, (0, 200, 600, 200))  # Horizontal road

        # Draw traffic lights
        if self.env.signal_NS == 1:
            pygame.draw.circle(self.screen, GREEN, (300, 180), 20)  # NS Green
            pygame.draw.circle(self.screen, RED, (300, 420), 20)    # EW Red
        else:
            pygame.draw.circle(self.screen, RED, (300, 180), 20)    # NS Red
            pygame.draw.circle(self.screen, GREEN, (300, 420), 20)  # EW Green

    def add_car(self, direction):
        if direction == 'NS':
            self.cars_NS.append(Car(290, random.randint(-300, -100), 'NS', random.randint(1, 3)))
        elif direction == 'EW':
            self.cars_EW.append(Car(random.randint(-300, -100), 290, 'EW', random.randint(1, 3)))

    def update_cars(self):
        for car in self.cars_NS:
            car.move()
        for car in self.cars_EW:
            car.move()

        # Remove cars that are out of the screen
        self.cars_NS = [car for car in self.cars_NS if car.y < SCREEN_HEIGHT]
        self.cars_EW = [car for car in self.cars_EW if car.x < SCREEN_WIDTH]

    def draw_cars(self):
        for car in self.cars_NS:
            pygame.draw.rect(self.screen, YELLOW, (car.x, car.y, CAR_WIDTH, CAR_HEIGHT))
        for car in self.cars_EW:
            pygame.draw.rect(self.screen, YELLOW, (car.x, car.y, CAR_HEIGHT, CAR_WIDTH))

    def step(self):
        # Get the current state of the environment
        state = self.env.get_state()  # Assuming this returns the environment state for both directions

        # Agent 1 (NS traffic lights) takes an action
        action1 = self.agent1.act(state)
        self.env.update_traffic_lights(action1, direction='NS')  # Update NS traffic lights based on agent1's action

        # Agent 2 (EW traffic lights) takes an action
        action2 = self.agent2.act(state)
        self.env.update_traffic_lights(action2, direction='EW')  # Update EW traffic lights based on agent2's action

        # Proceed with the environment’s step (vehicles moving, etc.)
        action = [action1, action2]  # Both actions passed as a list
        self.env.step(action)

    def run(self):
        while True:
            self.screen.fill(WHITE)  # Clear screen
            self.draw_intersection()  # Draw the intersection
            self.draw_cars()  # Draw the cars

            # Call step to update traffic lights and environment state based on agent actions
            self.step()  # Run a step in the simulation

            self.update_cars()  # Update car movements based on traffic flow

            # Handle pygame events (for quitting the game, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()  # Update the display with new information
            self.clock.tick(FPS)  # Control the frame rate
