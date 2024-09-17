import matplotlib.pyplot as plt
import json
filename = "counter_circuit_o_1order_3_hidden_layers.json"

with open(filename, 'r') as openfile:
    episode_soups = json.load(openfile)

episode_soups = episode_soups[0:25000]

plt.plot(episode_soups)
plt.xlabel('Num Episodes') 
plt.ylabel('Num Soups Delivered') 
plt.title('Counter Circuit O 1order Training')
plt.show()
