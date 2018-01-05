import numpy as np
cimport numpy as np

DTYPE = np.float
DTYPE_int = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_int_t

cdef double get_dE(int cent, int neighk, int neighc, double u11, double u00):
    return u11*(neighk*neighc - cent*neighc) + u00*((not neighk)*(not neighc) - (not cent)*(not neighc))

def metropolis(int nsteps, int N, np.ndarray[DTYPE_int_t, ndim=3] lattice, double u11, double u00, double kT):
    cdef np.ndarray[DTYPE_int_t, ndim=2] neighbor_kernel = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] reverse_neigh_indices = np.array([1, 0, 3, 2, 5, 4], dtype=DTYPE_int)

    printflag = -1
    for stepcounter in range(nsteps):
        pos = np.random.randint(N, size=3) # pick random coordinates to update
        cent = lattice[pos[0], pos[1], pos[2]]
        dE = np.zeros(6, dtype=DTYPE) # initialize deltaE
        neighbors = (pos + neighbor_kernel) % N
        for k in range(6):
            nn = (neighbors[k] + neighbor_kernel) % N
            neighk = lattice[neighbors[k,0],neighbors[k,1],neighbors[k,2]]
            # center loop
            for c in range(6):
                if c != k:
                    dE[k] += get_dE(cent,
                                    neighk,
                                    lattice[neighbors[c,0],neighbors[c,1],neighbors[c,2]], u11, u00)
            # neighbor loop
            for c in range(6):
                if c != reverse_neigh_indices[k]:
                    dE[k] += get_dE(cent,
                                    neighk,
                                    lattice[neighbors[c, 0], neighbors[c, 1], neighbors[c, 2]], u11, u00)
        probs = np.ones(6, dtype=DTYPE)
        for j in range(6):
            if dE[j] <= 0:
                probs[j] = 1.
            else:
                probs[j] = np.exp(-dE[j]/kT)
        probs /= sum(probs) # norm to make actual probabilities
        intervals = np.cumsum(probs)
        dice_roll = np.random.random() # create random number between 0 and 1
        process_id = np.where(intervals > dice_roll)[0][0] # chosen neighbor to exchange with
        # exchange with chosen neigh
        npos = (pos + neighbor_kernel[process_id]) % N
        lattice[pos[0],pos[1],pos[2]], lattice[npos[0],npos[1],npos[2]] = lattice[npos[0],npos[1],npos[2]], cent

        percentage_finished = round(100 * stepcounter / nsteps)
        if (stepcounter % (nsteps // 10) == 0) and (printflag != percentage_finished):
            print("{}% ... ".format(percentage_finished))
            printflag = percentage_finished

    return lattice