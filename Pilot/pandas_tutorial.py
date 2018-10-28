import numpy as np
import pandas as pd

# The Pandas Series Object
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)

data = pd.Series({2: 'a', 1: 'b', 3: 'c'})
print(data)

# DataFrame
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)

area_dict = {   'California': 423967, 
                'Texas': 695662,
                'New York': 141297,
                'Florida': 170312,
                'Illinois': 149995}
area = pd.Series(area_dict)

states = pd.DataFrame({'population': population, 'area': area})
print(states)

print(pd.DataFrame(population, columns=['population']))

# random
data  = pd.DataFrame(np.random.rand(3, 2),
                    columns=['foo', 'bar'],
                    index=['a', 'b', 'c'])

print(data)