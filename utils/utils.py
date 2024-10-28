import matplotlib.pyplot as plt

def plot_results(rewards, episodes):
    plt.plot(range(episodes), rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Time")
    plt.show()
