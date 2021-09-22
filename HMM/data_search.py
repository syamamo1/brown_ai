import numpy as np

data = []  # paste ugly ass matrix representation here

easy_data = []
for data_tuple in data:
       state = data_tuple[0]
       observation = data_tuple[1]
       s = np.nonzero(state.flatten())[0].item()
       o = np.nonzero(observation.flatten())[0].item()
       easy_data.append((s, o))

dict = {}
for i in range(400):
       dict[i] = []

for state, observable in easy_data:
       dict[state].append(observable)

print(dict)