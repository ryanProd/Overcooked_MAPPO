import matplotlib.pyplot as plt
import json
import numpy as np
filename = "counter_circuit_o_1orderother_metrics.json"

with open(filename, 'r') as openfile:
    l = json.load(openfile)

dish_drop = l[0]

soup_drop = l[1]
print(np.mean(soup_drop))

plt.plot(soup_drop)
plt.xlabel('Num Episodes') 
plt.ylabel('Num Soups Dropped') 
plt.title('Counter Circuit O 1Order Soups Dropped')
plt.show()