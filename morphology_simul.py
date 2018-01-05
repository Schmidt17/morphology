import pylab as p
import numpy as n
from percolation import SitePercolator2D
#import time

def get_neighbors(x, y):
    neigh_list = [((x+1)%N, y), ((x-1)%N, y), (x, (y+1)%N), (x, (y-1)%N)]
    return neigh_list

def energy(x, y, neighbors):
    #neighbors = get_neighbors(x, y)
    en = 0.
    for neigh in neighbors:
        en += lattice[neigh]*lattice[x,y]*u11 + u00*((not lattice[neigh])*(not lattice[x,y])) + u10*(lattice[neigh] + lattice[x,y])*(not lattice[neigh]*lattice[x,y])
    return en

# create 2d square lattice (NxN)
N = 100
lattice = p.zeros(N*N)
# populate initially randomly with given number fraction
x1 = 0.5 # x1: fraction of ones
N_ones = int(x1*len(lattice)) # actual number of ones
lattice[list(n.random.permutation(p.arange(len(lattice)))[:N_ones])] = 1
lattice = lattice.reshape((N,N))
init_lattice = p.copy(lattice)

percolator = SitePercolator2D(N, N, 0.)  # prob is not needed, since we take the geometry from here

# let neighbors switch place with thermal activation (periodic boundary conditions)
# Metropolis algorithm for updating
maxsteps = 1000000
kT = 0.025
u11 = -1.3 # in eV
u00 = -1.3 # in eV
u10 = -0. # in eV
d_utransf = 0.00 # energy barrier for place exchange, in eV
#p.ion()
# p.figure()
# p.imshow(lattice, interpolation='none')
# p.suptitle("Initial configuration")
# p.show()
#p.draw()
#time.sleep(.5)
printflag = -1
for i in range(maxsteps):
    a, b = n.random.randint(N), n.random.randint(N) # pick random point to update
    neighbors = get_neighbors(a, b)
    current_energy = energy(a, b, neighbors)
    energies = p.zeros(4) # 4 neighbors for square lattice
    for j, neigh in enumerate(neighbors):
        # exchange with neigh and calc new energy
        val = lattice[a,b]
        lattice[a,b] = lattice[neigh]
        lattice[neigh] = val
        energies[j] = energy(a, b, neighbors)
        # exchange back
        val = lattice[a, b]
        lattice[a, b] = lattice[neigh]
        lattice[neigh] = val
    # calculate probabilities for each transition:
    probs = p.ones(4) # in case of 5: last one is prob. of doing nothing
    for j, en in enumerate(energies):
        if en <= current_energy:
            probs[j] = p.exp(-d_utransf/kT)
        else:
            probs[j] = p.exp(-(en - current_energy + d_utransf)/kT)
    probs /= sum(probs) # norm to make actual probabilities
    intervals = n.cumsum(probs)
    dice_roll = n.random.random() # create random number between 0 and 1
    process_id = n.where(intervals > dice_roll)[0][0] # chosen neighbor to exchange with
    if process_id < 4: # if not "do nothing" ...
        # ... exchange with chosen neigh
        val = lattice[a, b]
        lattice[a, b] = lattice[neighbors[process_id]]
        lattice[neighbors[process_id]] = val
    percentage_finished = round(100 * i / maxsteps)
    if (i % (maxsteps // 10) == 0) and (printflag != percentage_finished):
        print("{}% ... ".format(percentage_finished), end="", flush=True)
        printflag = percentage_finished
        if percentage_finished == 50:
            p.figure()
            p.imshow(lattice, interpolation='none')
            p.suptitle("Intermediate configuration")
            p.draw()
    #p.gca().clear()
    #p.imshow(lattice, interpolation='none')
    #p.draw()
    #time.sleep(.5)

percolator.is_occupied = lattice
spanning, N_cl = percolator.spanning_cluster_exists(leave_cluster_marked=True)
if spanning:
    print("Spanning cluster exists and consists of {0}% of total sites.".format(100*float(N_cl)/N/N))
else:
    print("No spanning cluster found.")

print("Calculating border distance distribution")
# i) put border site positions (i.e. sites of 1 or 2 with at least one 0 neighbor) in an array and mark them with -1
borderPositions1 = []
for a in range(N):
    for b in range(N):
        if lattice[a, b] > 0:
            neighbors = get_neighbors(a, b)
            border = False
            for neigh in neighbors:
                if lattice[neigh] == 0:
                    borderPositions1.append([a, b])
                    lattice[a, b] = -1
                    break
borderPositions1 = p.array(borderPositions1)
# ii) find closest distance to border site for each 1 and 2 (type one and spanning cluster or not)
distances = []
for a in range(N):
    for b in range(N):
        if lattice[a, b] > 0:
            diffs = borderPositions1 - p.array([a, b])
            minBorderDist = p.amin(p.sum((diffs*diffs).T, axis=0))
            distances.append(p.sqrt(minBorderDist))
# print(distances)
p.figure()
p.imshow(init_lattice, interpolation='none')
p.suptitle("Initial configuration")
p.figure()
p.imshow(lattice, interpolation='none')
p.suptitle("Final configuration")
p.figure()
p.hist(distances)
p.suptitle("Distance distribution")
p.show()