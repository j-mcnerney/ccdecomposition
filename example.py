#%%
"""
An example of use of the CostProblem class using an interactive .py script.
"""

import numpy as np
import pandas as pd
from ccdecomp import CostModel

#%% 1: Specify the cost problem

# Initialize
cp = CostModel(
    title = 'Made up cost problem.',
    equation = 'a1 * x1 x2+a2 * x3 *x4_cubed x2 * x1 + x5 + a3 a4 CF * x2'
)

# Rename cost components [optional]
cp.name_cost_components(['Materials','Labor','Equipment','O&M'])

# Specify which symbols are fixed parameters [optional]
cp.identify_parameters(['a1','a2','a3','a4'])

# Check that equation was correctly parsed
print('\n\nCost problem construction:')
print(cp)


#%% 2: Bind to data

# Enter data
time = [1980, 1985, 1993]
data = pd.DataFrame(index=time)
data['a1'] = 1 * np.ones((len(time), 1))
data['a2'] = 2.5 * np.ones((len(time), 1))
data['a3'] = 10 * np.ones((len(time), 1))
data['a4'] = 0.03 * np.ones((len(time), 1))
data['x1'] = [1, 1, 1.2]
data['x2'] = [100, 120, 150]
data['x3'] = [0.5, 0.6, 0.55]
data['x4_cubed'] = [5, 5.2, 5.7]
data['x5']= [10, 80, 120]
data['CF'] = [30, 40, 41]

# Bind data to symbols
cp.bind_data(data)

# Verify that data was bound correctly.  Examine costs in each period.
print('\n\nUpon binding the problem to variable data, cost components are automatically computed:')
cp.display_data()


#%% 3: Compute cost change decompositions

# Specify desire time spans to study
time_spans = [
    (1980, 1985),
    (1985, 1993),
    (1980, 1993)
]

# Get change decomposition over these spans
cp.cost_change_decomposition(time_spans)

# Report results
print('\n\nAfter asking for change decompositions over particular time spans:')
cp.display_contributions()

# Show auxiliary quantities used in the computation [optional]
print('\n\nRepresentative values of cost components and log changes to variables:')
cp.display_change_data()



# %%
