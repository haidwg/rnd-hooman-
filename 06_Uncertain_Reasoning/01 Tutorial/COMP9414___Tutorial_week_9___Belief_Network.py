# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 00:11:51 2023

@author: Francisco
"""

import pomegranate as pg

#section 3.1.1
fo = pg.DiscreteDistribution({'family out': 0.15, 'family home': 0.85})
bp = pg.DiscreteDistribution({'bowel problem': 0.01, 'no bowel problem': 0.99})

#section 3.1.2
lo = pg.ConditionalProbabilityTable(
        [[ 'family home', 'light off', 0.95 ],
         [ 'family home', 'light on', 0.05 ],
         [ 'family out', 'light off', 0.4 ],
         [ 'family out', 'light on', 0.6 ]],[fo]) 

do = pg.ConditionalProbabilityTable(
        [[ 'family home', 'no bowel problem', 'dog out', 0.3 ],
         [ 'family home', 'bowel problem', 'dog out', 0.97 ],
         [ 'family out', 'no bowel problem', 'dog out', 0.9 ],
         [ 'family out', 'bowel problem', 'dog out', 0.99 ],
         [ 'family home', 'no bowel problem', 'dog in', 0.7 ],
         [ 'family home', 'bowel problem', 'dog in', 0.03 ],
         [ 'family out', 'no bowel problem', 'dog in', 0.1 ],
         [ 'family out', 'bowel problem', 'dog in', 0.01 ]],[fo,bp]) 

hb = pg.ConditionalProbabilityTable(
        [[ 'dog in', 'no hear bark', 0.99 ],
         [ 'dog in', 'hear bark', 0.01 ],
         [ 'dog out', 'no hear bark', 0.3 ],
         [ 'dog out', 'hear bark', 0.7 ]],[do]) 

#section 3.1.3
model = pg.BayesianNetwork("MyBN")

#section 3.1.4
FO = pg.State(fo, name="FO")
BP = pg.State(bp, name="BP")
LO = pg.State(lo, name="LO")
DO = pg.State(do, name="DO")
HB = pg.State(hb, name="HB")

model.add_states(FO, BP, LO, DO, HB)

#section 3.1.5
model.add_edge(FO, LO)
model.add_edge(FO, DO)
model.add_edge(BP, DO)
model.add_edge(DO, HB)

#section 3.1.6
model.bake()

#section 3.1.7
#P(~FO,BP,~LO,DO,HB)
print(model.probability([["family home", "bowel problem", "light off", "dog out", "hear bark"]]))

#P(DO|BP)
print(model.predict_proba([{'BP': 'bowel problem'}])[0][3].parameters[0]['dog out'])

#P(~BP|~DO)
print(model.predict_proba([{'DO': 'dog in'}])[0][1].parameters[0]['no bowel problem'])