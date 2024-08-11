# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:32:52 2023

@author: Francisco
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#section 4.1.1
speed = ctrl.Antecedent(np.arange(0, 151, 1), 'Speed')
temperature = ctrl.Antecedent(np.arange(0, 151, 1), 'Temperature')
injection = ctrl.Consequent(np.arange(0, 101, 1), 'Injection', defuzzify_method='centroid')

#section 4.1.2
speed['low'] = fuzz.trapmf(speed.universe, [0, 0, 25, 60])
speed['medium'] = fuzz.trimf(speed.universe, [25, 75, 125])
speed['high'] = fuzz.trapmf(speed.universe, [60, 100, 150, 150])

temperature['low'] = fuzz.trapmf(temperature.universe, [0, 0, 25, 60])
temperature['medium'] = fuzz.trimf(temperature.universe, [25, 75, 125])
temperature['high'] = fuzz.trapmf(temperature.universe, [60, 100, 150, 150])

injection['low'] = fuzz.trimf(injection.universe, [0, 0, 30])
injection['medium'] = fuzz.trimf(injection.universe, [10, 50, 90])
injection['high'] = fuzz.trimf(injection.universe, [70, 100, 100])

#section 4.1.3
speed['medium'].view()
temperature.view()
injection.view()

#section 4.1.4 rules definition
rule1 = ctrl.Rule(speed['medium'] & temperature['high'], injection['low']) #rule6
rule2 = ctrl.Rule(speed['low'] & temperature['low'], injection['high']) #rule1
rule3 = ctrl.Rule(speed['high'] & temperature['medium'], injection['low']) #rule8

#section 4.1.7
# rule1 = ctrl.Rule(speed['low'] & temperature['low'], injection['high'])
# rule2 = ctrl.Rule(speed['low'] & temperature['medium'], injection['high'])
# rule3 = ctrl.Rule(speed['low'] & temperature['high'], injection['medium'])

# rule4 = ctrl.Rule(speed['medium'] & temperature['low'], injection['high'])
# rule5 = ctrl.Rule(speed['medium'] & temperature['medium'], injection['medium'])
# rule6 = ctrl.Rule(speed['medium'] & temperature['high'], injection['low'])

# rule7 = ctrl.Rule(speed['high'] & temperature['low'], injection['medium'])
# rule8 = ctrl.Rule(speed['high'] & temperature['medium'], injection['low'])
# rule9 = ctrl.Rule(speed['high'] & temperature['high'], injection['low'])

#section 4.1.5 create controller and control simulator
injection_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
# Uncomment for section 4.1.7
# injection_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

injection_sim = ctrl.ControlSystemSimulation(injection_ctrl)

#section 4.1.6
injection_sim.input['Speed'] = 50
injection_sim.input['Temperature'] = 10
injection_sim.compute()

# Results and visualization.
print(injection_sim.output['Injection'])
injection.view(sim=injection_sim)

injection_sim.input['Speed'] = 70
injection_sim.input['Temperature'] = 100
injection_sim.compute()

print(injection_sim.output['Injection'])
injection.view(sim=injection_sim)


#section 4.1.8
"""
upsampled = np.linspace(0, 150, 16)
x, y = np.meshgrid(upsampled, upsampled)
z = np.zeros_like(x)

#collect injections for the control surface
for i in range(15):
    for j in range(15):
        injection_sim.input['Speed'] = x[i, j]
        injection_sim.input['Temperature'] = y[i, j]
        injection_sim.compute()
        z[i, j] = injection_sim.output['Injection']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)

ax.view_init(20, 100)
"""
