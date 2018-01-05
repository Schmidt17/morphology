import pylab as p
import numpy as n
from percolation import SitePercolator3D
import MC
import time

def get_neighbors(x, y, z):
    neigh_list = [((x+1)%N, y, z), ((x-1)%N, y, z), (x, (y+1)%N, z), (x, (y-1)%N, z), (x, y, (z+1)%N), (x, y, (z-1)%N)]
    return neigh_list

def energy(x, y, z, neighbors):
    en = 0.
    for neigh in neighbors:
        en += lattice[neigh]*lattice[x,y,z]*u11 + u00*((not lattice[neigh])*(not lattice[x,y,z])) + u10*(lattice[neigh] + lattice[x,y,z])*(not lattice[neigh]*lattice[x,y,z])
    return en

def delta_E(pos):
    t0 = time.time()
    neighbors = (pos + neighbor_kernel) % N
    tpos = tuple(pos)
    dE = n.zeros(6)  # calc. dE for each neighbor exchange, return array
    for k, neigh in enumerate(neighbors):
        tneigh = tuple(neigh)
        if lattice[tneigh] != lattice[tpos]:
            n_center = [neighbors[j] for j in p.delete(p.arange(6), k)]
            n_neigh = ((pos + neighbor_neighbors[k]) % N)
            # n_neigh = [ne for j, ne in enumerate(n_neigh) if j != reverse_neigh_indices[k]]
            for temp in n_center:
                temp = tuple(temp)
                dE[k] -= lattice[temp]*lattice[tpos]*u11 + u00*((not lattice[temp])*(not lattice[tpos])) #+ u10*(lattice[temp] + lattice[tpos])*(not lattice[temp]*lattice[tpos])
                dE[k] += lattice[temp]*lattice[tneigh]*u11 + u00*((not lattice[temp])*(not lattice[tneigh])) #+ u10*(lattice[temp] + lattice[tneigh])*(not lattice[temp]*lattice[tneigh])
            for temp in n_neigh:
                temp = tuple(temp)
                dE[k] += lattice[temp]*lattice[tpos]*u11 + u00*((not lattice[temp])*(not lattice[tpos])) #+ u10*(lattice[temp] + lattice[tpos])*(not lattice[temp]*lattice[tpos])
                dE[k] -= lattice[temp]*lattice[tneigh]*u11 + u00*((not lattice[temp])*(not lattice[tneigh])) #+ u10*(lattice[temp] + lattice[tneigh])*(not lattice[temp]*lattice[tneigh])
    t1 = time.time()
    return dE, t1-t0

# create 3d cubic lattice (NxNxN)
N = 40
lattice = p.zeros(N*N*N, dtype=n.int)
# populate initially randomly with given number fraction
x1 = 0.5 # x1: fraction of ones
N_ones = int(x1*len(lattice)) # actual number of ones
lattice[list(n.random.permutation(p.arange(len(lattice)))[:N_ones])] = 1
lattice = lattice.reshape((N,N,N))
init_lattice = p.copy(lattice)

percolator = SitePercolator3D(N, N, N, 0., 0., 0.)  # parameters are not needed, since we take the geometry from here

# let neighbors switch place with thermal activation (periodic boundary conditions)
# Metropolis algorithm for updating
maxsteps = 10000000
kT = 0.031
u11 = -.5 # in eV
u00 = -.5 # in eV
u10 = 0. # in eV
d_utransf = 0.00 # energy barrier for place exchange, in eV

neighbor_kernel = n.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])

neighbor_neighbors = n.array([[[2,0,0], [1,1,0], [1,-1,0], [1,0,1], [1,0,-1]],
                              [[-2,0,0], [-1,1,0], [-1,-1,0], [-1,0,1], [-1,0,-1]],
                              [[0,2,0], [1,1,0], [-1,1,0], [0,1,1], [0,1,-1]],
                              [[0,-2,0], [1,-1,0], [-1,-1,0], [0,-1,1], [0,-1,-1]],
                              [[0,0,2], [1,0,1], [-1,0,1], [0,1,1], [0,-1,1]],
                              [[0,0,-2], [1,0,-1], [-1,0,-1], [0,1,-1], [0,-1,-1]]
                              ])
reverse_neigh_indices = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}

# call Cython implementation
print("Start Cython Metropolis with {0} steps".format(maxsteps))
tstart = time.time()
lattice = MC.metropolis(maxsteps, N, lattice, u11, u00, kT)
tend = time.time()
print("Took {0} s".format(tend-tstart))

"""
#p.ion()
# p.figure()
# p.imshow(lattice, interpolation='none')
# p.suptitle("Initial configuration")
# p.show()
#p.draw()
#time.sleep(.5)
printflag = -1
tstart = time.time()
tdelta = 0.
for i in range(maxsteps):
    curr_pos = n.random.randint(N, size=3) # pick random coordinates to update
    dEs, dt = delta_E(curr_pos)
    tdelta += dt
    # calculate probabilities for each transition:
    probs = p.ones(6) # in case of 7: last one is prob. of doing nothing
    for j, den in enumerate(dEs):
        if den <= 0:
            probs[j] = p.exp(-d_utransf/kT)
        else:
            probs[j] = p.exp(-(den + d_utransf)/kT)
    probs /= sum(probs) # norm to make actual probabilities
    intervals = n.cumsum(probs)
    dice_roll = n.random.random() # create random number between 0 and 1
    process_id = n.where(intervals > dice_roll)[0][0] # chosen neighbor to exchange with
    if process_id < 6: # if not "do nothing" ...
        # ... exchange with chosen neigh
        tpos = tuple(curr_pos)
        npos = tuple((curr_pos + neighbor_kernel[process_id]) % N)
        lattice[tpos], lattice[npos] = lattice[npos], lattice[tpos]
    percentage_finished = round(100 * i / maxsteps)
    if (i % (maxsteps // 10) == 0) and (printflag != percentage_finished):
        print("{}% ... ".format(percentage_finished), end="", flush=True)
        printflag = percentage_finished
        # if percentage_finished == 50:
        #     p.figure()
        #     p.imshow(lattice, interpolation='none')
        #     p.suptitle("Intermediate configuration")
        #     p.draw()
    #p.gca().clear()
    #p.imshow(lattice, interpolation='none')
    #p.draw()
    #time.sleep(.5)
t_end = time.time()
print("Took {0} s, of that {1} s in deltaE.".format(t_end-tstart, tdelta))
"""

percolator.is_occupied = lattice #*(-lattice) + 1 #- for passing ones and zeros inverted
spanning, N_cl = percolator.spanning_cluster_exists(leave_cluster_marked=False)
if spanning:
    print("Spanning cluster exists and consists of {0}% of total sites.".format(100*float(N_cl)/N/N/N))
else:
    print("No spanning cluster found.")

print("Calculating border distance distribution")
# i) put border site positions (i.e. sites of 1 or 2 with at least one 0 neighbor) in an array and mark them with -1
borderPositions1 = []
for a in range(N):
    for b in range(N):
        for c in range(N):
            if lattice[a, b, c] > 0:
                neighbors = get_neighbors(a, b, c)
                border = False
                for neigh in neighbors:
                    if lattice[neigh] == 0:
                        borderPositions1.append([a, b, c])
                        lattice[a, b, c] = -1
                        break
borderPositions1 = p.array(borderPositions1)
# ii) find closest distance to border site for each 1 and 2 (type one and spanning cluster or not)
distances = []
for a in range(N):
    for b in range(N):
        for c in range(N):
            if lattice[a, b, c] > 0:
                diffs = borderPositions1 - p.array([a, b, c])
                minBorderDist = p.amin(p.sum((diffs*diffs).T, axis=0))
                distances.append(p.sqrt(minBorderDist))

# p.figure()
# p.imshow(init_lattice, interpolation='none')
# p.suptitle("Initial configuration")
p.figure()
p.imshow(p.sum(lattice[N//2:N//2+1], axis=0), interpolation='none', cmap=p.get_cmap('afmhot'))
p.suptitle("Final configuration")
p.figure()
p.hist(distances)
p.suptitle("Distance distribution")
p.show()