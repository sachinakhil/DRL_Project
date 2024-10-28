import tkinter as tk
import random
from simulation import TrafficSimulation
from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

def start_simulation():
    num_cars = int(entry.get())
    print(f'Starting simulation with {num_cars} cars...')

    # Initialize the environment and agents
    env = TrafficEnv()
    
    # Fix: Use nvec to get the action space sizes for NS and EW
    agent1 = DQNAgent(env.observation_space.shape[0], env.action_space.nvec[0])  # NS action space size
    agent2 = DQNAgent(env.observation_space.shape[0], env.action_space.nvec[1])  # EW action space size

    # Load the trained models
    agent1.load("agent1_model.pt")
    agent2.load("agent2_model.pt")
    print("Successfully loaded pre-trained models.")
    

    # Start the simulation
    sim = TrafficSimulation(env, agent1, agent2)
    for _ in range(num_cars):
        sim.add_car('NS')
        sim.add_car('EW')
        

    sim.run()

# Tkinter UI setup
root = tk.Tk()
root.title("Traffic Simulation Control")

label = tk.Label(root, text="Number of Cars:")
label.pack()

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Start Simulation", command=start_simulation)
button.pack()

root.mainloop()
