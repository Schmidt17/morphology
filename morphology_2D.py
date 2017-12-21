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

def delta_E(pos):
    neighbors = (pos + neighbor_kernel) % N
    tpos = tuple(pos)
    dE = n.zeros(4)  # calc. dE for each neighbor exchange, return array
    for k, neigh in enumerate(neighbors):
        tneigh = tuple(neigh)
        if lattice[tneigh] != lattice[tpos]:
            n_center = [ne for j, ne in enumerate(neighbors) if j!=k]
            n_neigh = ((neighbors + neighbor_kernel[k])%N)
            n_neigh = [ne for j, ne in enumerate(n_neigh) if j!=reverse_neigh_indices[k]]
            for temp in n_center:
                temp = tuple(temp)
                dE[k] -= lattice[temp]*lattice[tpos]*u11 + u00*((not lattice[temp])*(not lattice[tpos])) + u10*(lattice[temp] + lattice[tpos])*(not lattice[temp]*lattice[tpos])
                dE[k] += lattice[temp]*lattice[tneigh]*u11 + u00*((not lattice[temp])*(not lattice[tneigh])) + u10*(lattice[temp] + lattice[tneigh])*(not lattice[temp]*lattice[tneigh])
            for temp in n_neigh:
                temp = tuple(temp)
                dE[k] += lattice[temp]*lattice[tpos]*u11 + u00*((not lattice[temp])*(not lattice[tpos])) + u10*(lattice[temp] + lattice[tpos])*(not lattice[temp]*lattice[tpos])
                dE[k] -= lattice[temp]*lattice[tneigh]*u11 + u00*((not lattice[temp])*(not lattice[tneigh])) + u10*(lattice[temp] + lattice[tneigh])*(not lattice[temp]*lattice[tneigh])
    return dE

# create 2d square lattice (NxN)
N = 100
lattice = p.zeros(N*N)
# populate initially randomly with given number fraction
x1 = 0.5 # x1: fraction of ones
N_ones = int(x1*len(lattice)) # actual number of ones
lattice[list(n.random.permutation(p.arange(len(lattice)))[:N_ones])] = 1
lattice = lattice.reshape((N,N))
init_lattice = p.copy(lattice)

neighbor_kernel = n.array([[1,0], [-1,0], [0,1], [0,-1]])
reverse_neigh_indices = {0:1, 1:0, 2:3, 3:2}

percolator = SitePercolator2D(N, N, 0.)  # prob is not needed, since we take the geometry from here

# let neighbors switch place with thermal activation (periodic boundary conditions)
# Metropolis algorithm for updating
maxsteps = 1000000
kT = 0.025
u11 = -5.3 # in eV
u00 = -2.3 # in eV
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
    curr_pos = n.random.randint(N, size=2)  # pick random coordinates to update
    dEs = delta_E(curr_pos)
    # calculate probabilities for each transition:
    probs = p.ones(4)  # in case of 5: last one is prob. of doing nothing
    for j, den in enumerate(dEs):
        if den <= 0:
            probs[j] = p.exp(-d_utransf / kT)
        else:
            probs[j] = p.exp(-(den + d_utransf) / kT)
    probs /= sum(probs)  # norm to make actual probabilities
    intervals = n.cumsum(probs)
    dice_roll = n.random.random()  # create random number between 0 and 1
    process_id = n.where(intervals > dice_roll)[0][0]  # chosen neighbor to exchange with
    if process_id < 4:  # if not "do nothing" ...
        # ... exchange with chosen neigh
        tpos = tuple(curr_pos)
        npos = tuple((curr_pos + neighbor_kernel[process_id]) % N)
        lattice[tpos], lattice[npos] = lattice[npos], lattice[tpos]
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

p.figure()
p.imshow(init_lattice, interpolation='none')
p.suptitle("Initial configuration")
p.figure()
p.imshow(lattice, interpolation='none')
p.suptitle("Final configuration")
p.show()