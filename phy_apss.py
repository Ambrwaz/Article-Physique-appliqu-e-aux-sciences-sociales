import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#Paramètres de la simulation
step = 20000#nombre d'itérations
N = 100#nombres d'agents de la population
c = 0.05#facteur d'apprentissage
mu = 0.02#taux de mortalité des agents
s = 1/2 #prestige de la langue A
q = 1/2 #préférence des bilingues à parler A
pa = 1/2 #proportion initiale de A
pb = 1/2 #proportion initiale de B
pab = 1 - pa - pb #proportion initiale de AB
PA = [pa]
PB = [pb]
PAB = [pab]
NA = int(pa*N)
NB = int(pb*N)
NAB = int(pab*N)

for k in tqdm(range(step)) :
    NAs,NBs,NABs = NA,NB,NAB
    for a in range(NA):
        prob_a_to_ab = c*(1-mu)*(1-s)*(pb + (1-q)*pab)
        change = np.random.choice([True,False],p = [prob_a_to_ab,1-prob_a_to_ab])
        if change :
            NAs -= 1
            NABs += 1
    for b in range(NB) :
        prob_b_to_ab = c*(1-mu)*s*(pa + q*pab)
        change = np.random.choice([True,False],p = [prob_b_to_ab,1-prob_b_to_ab])
        if change :
            NBs -= 1
            NABs += 1
    for ab in range(NAB) :
        prob_ab_to_a = mu*s*(pa + q*pab)
        prob_ab_to_b = mu*(1-s)*(pb + (1-q)*pab)
        change = np.random.choice(['a','b','c'],p=[prob_ab_to_a,prob_ab_to_b,1 - prob_ab_to_a - prob_ab_to_b])
        if change == 'a' :
            NAs +=1
            NABs -= 1
        elif change == 'b' :
            NBs +=1
            NABs -=1
    NA = NAs
    NB = NBs
    NAB = NABs
    pa,pb,pab = NA/N,NB/N,NAB/N
    PA.append(pa)
    PB.append(pb)
    PAB.append(pab)

X = np.linspace(0,step,step+1)
plt.plot(X,PA,color ='red',label='Pa')
plt.plot(X,PB,color = 'blue',label = 'Pb')
plt.plot(X,PAB,color = 'green', label = 'Pab')

plt.legend()
plt.show()

