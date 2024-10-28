# Cell 4: Main Execution Cell (main.py)
from agents.dqn_agent import DQNAgent
from env.traffic_env import TrafficEnv
from simulation import TrafficSimulation
import numpy


if __name__ == '__main__':
    # Create the traffic environment
    env = TrafficEnv()

    # Create DQN agents for NS and EW traffic signals
    agent1 = DQNAgent(state_size=4, action_size=2)
    agent2 = DQNAgent(state_size=4, action_size=2)

    # Train the agents
    # agent1.train(env, episodes=500)
    # agent2.train(env, episodes=500)

    # # Save the models to .pt files
    # agent1.save('agent1_model.pt')
    # agent2.save('agent2_model.pt')

    # Optionally, load the models (if you want to run simulation after loading pre-trained models)
    agent1.load('agent1_model.pt')
    agent2.load('agent2_model.pt')


    # Create the traffic simulation
    simulation = TrafficSimulation(env, agent1, agent2)
    num_cars = 10  # You can set this to however many cars you want to add
    for _ in range(num_cars):
        direction = 'NS' if numpy.random.choice([True, False]) else 'EW'
        simulation.add_car(direction)

    # Run the simulation
    simulation.run()
