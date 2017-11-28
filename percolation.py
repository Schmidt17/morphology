# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:55:36 2017

@author: ahofacker
"""

from __future__ import print_function
from __future__ import with_statement

#import matplotlib
#matplotlib.use('Agg')

#import pylab as p
import numpy as np
import numpy.random as r

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

#import scipy.optimize as opt
# from my_brent import my_brent
# import cconnections

#import os
#from multiprocessing import Pool
#import pp
#from functools import partial
from itertools import combinations, product

# import potential_3D

# Vector routines ------------
def norm(x):
    return np.sqrt(sum(x*x))

def dist(a,b):
    return norm(a-b)
#-----------------------------
    
# Methods for automated search for percolation thresholds ---- 
def realisation(percolator, proc_nr, pr, kT, Nt, n, Nlin, Ef, lamb, sigma, Ec, n_lat):
#    percolator = BondVRHPercolator3D(prob = pr, kT = kT, alpha = 1., lamb = lamb, ef = Ef, eps_r = 3., L = float(Nlin), start_width = 1.01, filename = "energies"+str(proc_nr)+".dat")
#    os.system("python3 potential_3D.py " + str(percolator.L) + " " + str(n) + " " + str(proc_nr)) # create new positions and energies, save them in energies.dat
    pos_en = potential_3D.do_calculation(percolator.L, Nt, n, sigma, Ec, n_lat, pr_nr=proc_nr)
    percolator.generate_configuration(pos_en)
    spanning, sp_cl_size = percolator.spanning_cluster_exists() # leaves sites of spanning cluster set to 2. Pass False to turn this off.
    if spanning:
        return 1
    else:
        return 0
    
def find_P_minus_half(pr, kT, Nt, n, Nlin,Ef, lamb, sigma, Ec, n_lat):
    print("Starting p = {0:4.2f}".format(pr))
    N = 24
#    pool = Pool(N)
    
#    partial_real = partial(realisation, pr = pr, kT = kT, n=n, Nlin=Nlin, Ef=Ef, lamb=lamb)
    joblist = []
    samplelist = [BondVRHPercolator3D(prob = pr, kT = kT, alpha = float(Nlin)/float(Nt)**(1./3.), lamb = lamb, ef = Ef, eps_r = 3., L = float(Nlin), shrink_by = 0.1, start_width = 1.01, filename = "energies"+str(j)+".dat") for j in range(N)]
    for j in range(N):
        joblist.append(job_server.submit(realisation, (samplelist[j], j, pr, kT, Nt, n, Nlin, Ef, lamb, sigma, Ec, n_lat), depfuncs=(dist,norm), modules=("cconnections","numpy as np", "from numpy import random as r", "potential_3D", "pp")))
#    acc = sum(pool.map(partial_real, range(1,N+1)))
    reslist = [j() for j in joblist]
    acc = sum(reslist)
#    global Ps0 # for debugging
#    Ps0.append([pr, acc/float(N)])
    print("Ps = " + str(acc/float(N)))
    return acc/float(N) - 0.5 # return relative frequency of spanning clusters for each p minus 0.5, to find point where P is 0.5 =: percolation threshold

def find_pc(kT, Nt, n, Nlin, Ef, lamb, sigma, Ec, n_lat): # takes a percolator instance to work with
    # use Brent's root finding algorithm
#    return opt.brentq(find_P_minus_half, 1, 25, args=(kT,n,Nlin,Ef,lamb), xtol=0.1)
    return my_brent(find_P_minus_half, 15, 35, args=(kT,Nt,n,Nlin,Ef,lamb,sigma,Ec, n_lat), xtol=0.1)

# ------------------------------------------------------------

#Ps0 = [] # for debugging
#job_server = pp.Server(secret = "02559", ncpus=4)
#print("Starting pp with {0} workers".format(job_server.get_ncpus()))

def main():
    
#    r.seed(4)
#    Nlin = 21.5 # linear dimension for cubic sample. 22 -> about 10^4 sites
#    Nx = Nlin
#    Ny = Nlin
#    Nz = Nlin
#    e_fermi = -0.4
#    percolator = BondVRHPercolator3D(prob = 12., kT = 0.025, alpha = 1., ef = e_fermi, eps_r = 3., L = float(Nlin), start_width = 1.01, filename = "energies0.dat")
    
#    os.system("python3 3D_potential.py")
#    percolator.generate_configuration()
#    spanning, sp_cl_size = percolator.spanning_cluster_exists()
#    percolator.plot_configuration()  
    
# for lambda variation ---------------------------------------------    
#    Nls = [21.5, 21.5, 21.5, 21.5, 21.5]
#    Efs = [-0.42, -0.42, -0.42, -0.42, -0.42]
#    lambs = [0.08, 0.1, 0.15, 0.2, 0.25]
#    ns = [1000, 1000, 1000, 1000, 1000]
#    iterate_over = lambs
#    folder = "Ea_vs_lambda_10pc_sig_200meV"
        
# for Nlin variation -----------------------------------------------
#    Nls = [21.5, 19.875, 18.25, 16.625, 15.]
#    Efs = [-0.42, -0.42, -0.47, -0.48, -0.50]
#    lambs = [0.08, 0.08, 0.08, 0.08, 0.08]
#    ns = [1000, 1000, 1000, 1000, 1000]
#    iterate_over = Nls
#    folder = "Ea_vs_Nlin_"
        
# for Nd variation -------------------------------------------------
    Nls = 18.34*np.ones(5) # ZnPc: 18.34, F4ZnPc: 19.07, F8ZnPc: 19.76
    Efs = [-0.58, -0.55, -0.56, -0.48, -0.41]
    lambs = 0.184*np.ones(5) # ZnPc: 0.17, F4ZnPc: 0.174, F8ZnPc: 0.184
    ns = [100, 200, 450, 1000, 2222]
    iterate_over = ns
    folder = "Ea_vs_Nd_ZnPc_shrinked"
    
    Nts = [9212, 9800, 9800, 9800, 9800]
    
    sigma = 0.08
    E_coulomb = 0.868 # eV - Coulomb potential on states closer than n_lat "lattice distances" to charged donor
                        # for ZnPc: 0.868, for F4ZnPc: 0.749, for F8ZnPc: 0.633
    n_lat = 1.1

    kTs = 1./np.array([35., 36.65, 38.3, 40., 41.7, 43.35, 45.]) # in order to calculate 1st derivative of delta p_c at kT**-1 = 40
    
    percolator = BondVRHPercolator3D(prob = 14., kT = 1./40., alpha = 0.5*float(18.34)/float(9212)**(1./3.), lamb = 0.17, ef = -0.36, eps_r = 3., L = float(18.34), shrink_by = 0, start_width = 1.01, filename = "energies.dat")
#    sigmas_stds = []
#    for n in np.logspace(2,3.35,20):    
#        sigmas_temp = []    
#        for i in range(10):
#            pos_en = potential_3D.do_calculation(percolator.L, 9212, int(n), 0.08, 0.868, 1.1, pr_nr=0)
#            percolator.load_positions(pos_en)
#            sig = percolator.find_sigma()
#            sigmas_temp.append(sig)
#        sigmas_temp = np.array(sigmas_temp)
#        sig = np.nanmean(sigmas_temp)
#        err = np.nanstd(sigmas_temp)
#        sigmas_stds.append([sig, err])
#        print("n = {0}: sigma = {1} +- {2}".format(int(n), sig, err))
#    sigmas_stds = np.array(sigmas_stds)
#    np.savetxt("sigmas_errs_ZnPc_min_random_noEc.dat",sigmas_stds)
#    plt.errorbar(np.logspace(2,3.35,20), sigmas_stds[:,0], yerr = sigmas_stds[:,1], fmt = "s")
    
    Nt = Nts[0]
    Ec = E_coulomb
    proc_nr = 0
    n = 2*921
    pos_en, CT_ids = potential_3D.do_calculation(percolator.L, Nt, n, sigma, Ec, n_lat, pr_nr=proc_nr)
    percolator.generate_configuration(pos_en)
    spanning, sp_cl_size = percolator.spanning_cluster_exists() # leaves sites of spanning cluster set to 2. Pass False to turn this off.   
    percolator.plot_configuration(CT_ids)
    plt.show()
    
#    Eas = []
    
#    for nr, it in enumerate(iterate_over):
#        pcs = []
#        for kT in kTs:
#            pc = find_pc(kT, Nts[nr], ns[nr], Nls[nr], Efs[nr], lambs[nr], sigma, E_coulomb, n_lat)
#            pcs.append(pc)
#            print("Saving to " + folder + "/pcs_"+str(it)+".dat")
#            np.savetxt(folder + "/pcs_"+str(it)+".dat",np.array([[1./kTs[i], pcrit] for i, pcrit in enumerate(pcs)]))
            
#        Eas.append((pcs[1] - pcs[0])/10.)
#        np.savetxt("Eas_"+str(nr)+".dat",np.array(Eas))
#    p.plot(1./kTs, pcs, 's')
#    p.plot(lambs, Eas, 's')
#    for i,e in enumerate(Eas):
#        print("lambda = {0}\tEa = {1}".format(lambs[i],e))
        
#    global Ps0 # for debugging
#    Ps0 = np.array(Ps0)
#    np.savetxt("Ps.dat", Ps0)
#    p.plot(Ps0[:,0], Ps0[:,1], color="blue")
        
#    # Monte Carlo: Find and plot probability of spanning cluster for several ps, average over N configurations
#    N = 5
#    ps = np.linspace(8, 13, 5) # scan a range of occupation probabilities p
#    Ps = [] # list for saving the probability of a spanning cluster P_s for each p
#    for pr in ps:
#        print("Starting p = {0}".format(pr))
#        acc = 0.
#        percolator.set_p(pr)
#        for i in range(N):
#            os.system("python3 3D_potential.py " + str(Nlin)) # create new positions and energies, save them in energies.dat #TODO: Ns could be done in parallel.
#            percolator.generate_configuration()
#            spanning, sp_cl_size = percolator.spanning_cluster_exists() # leaves sites of spanning cluster set to 2. Pass False to turn this off.
#            if spanning:
#                acc += 1 # count number of configurations with spanning cluster for each p
##                acc += float(sp_cl_size)/float(Nx)/float(Ny) # prob. of random site belonging to spanning cluster
#        Ps.append(acc/float(N)) # remember relative frequency of spanning clusters for each p
    
#    p.figure()
#    p.plot(ps, Ps, 'o-', label=r"$\varepsilon_\mathrm{F}" +  "= {0:.2f},$".format(e_fermi) + r" $N_\mathrm{lin}" +  "= {0}$".format(Nlin))
#    p.ylim((-0.1, 1.1))
#    p.xlabel(r"$\mathrm{Cutoff \, Exponent} \, \xi$", size=20)
#    p.ylabel("$\mathrm{Probability \, of \, Spanning \, Cluster} \, P_\mathrm{s}$", size=20)
#    p.legend(loc="best")
#    
#    p.show()

class Percolator:
    """
    Base class for all percolators
    """
    
    def __init__(self, Nx, Ny, prob):
        raise NotImplementedError()
    
    def set_p(self, prob):
        self.prob = prob
        
    def set_kT(self, kT):
        self.kT = kT

    def neighbor_list(self, x, y):
        raise NotImplementedError()
    
    def generate_configuration(self):
        raise NotImplementedError()
    
    def spanning_cluster_exists(self, leave_cluster_marked = True):
        raise NotImplementedError()
        
    def plot_configuration(self):
        raise NotImplementedError()
        
class Percolator2D(Percolator):
    """
    Models percolation on a square 2D lattice
    """
    
    def __init__(self, Nx, Ny, prob):
        self.is_occupied = np.zeros((Nx, Ny))
        self.Nx = Nx
        self.Ny = Ny
        self.prob = prob
        
        self.ef = -0.6  # Fermi level in eV
        self.eps_r = 3. # Relative permittivity
    
    def neighbor_list(self, x, y):
        """ Neighbors on a square lattice with free boundaries """
        neigh_list = []
        if x > 0:
            neigh_list.append((x - 1, y))
        if x < (self.Nx - 1):
            neigh_list.append((x + 1, y))
        if y > 0:
            neigh_list.append((x, y - 1))
        if y < (self.Ny - 1):
            neigh_list.append((x, y + 1))
        return neigh_list


class Percolator3D(Percolator):
    
    def __init__(self, Nx, Ny, Nz, prob, ef, sigma, eps_r=3.):
        self.is_occupied = np.zeros((Nx, Ny, Nz))
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.prob = prob
        
        self.ef = ef        # Fermi level in eV
        self.sigma = sigma  # Width of intrinsic Gaussian DOS
        self.eps_r = eps_r  # Relative permittivity, dimensionless
    
    def neighbor_list(self, x, y, z):
        """ Neighbors on a cubic lattice with free boundaries """
        neigh_list = []
        if x > 0:
            neigh_list.append((x - 1, y, z))
        if x < (self.Nx - 1):
            neigh_list.append((x + 1, y, z))
        if y > 0:
            neigh_list.append((x, y - 1, z))
        if y < (self.Ny - 1):
            neigh_list.append((x, y + 1, z))
        if z > 0:
            neigh_list.append((x, y, z - 1))
        if z < (self.Nz - 1):
            neigh_list.append((x, y, z + 1))
        return neigh_list 
        
class SitePercolator2D(Percolator2D):
    
    def generate_configuration(self):
        self.is_occupied = np.zeros((self.Nx, self.Ny))
        for xi in range(self.Nx):
            for yi in range(self.Ny):
                random_number = r.random()
                if random_number < self.prob:
                    self.is_occupied[xi, yi] = 1
                    
    def spanning_cluster_exists(self, leave_cluster_marked = True):
        """ Check for spanning cluster from y=0 to Ny-1 using flood fill """
        for startx in range(self.Nx):
            x = startx
            y = 0
            if self.is_occupied[x, y]:
                self.is_occupied[x, y] = 2
                new_sites = [(x, y)]
                sp_cluster = False  # existence flag of spanning cluster
                cluster_sites = [(x, y)]
                while (len(new_sites) > 0): # and (sp_cluster == False): # if only detection of interest, add second condition for speeding up. Then, not the complete cluster may be set to is_occupied = 2.
                    neighs = []
                    for site in new_sites:
                        neighs += self.neighbor_list(site[0], site[1])
                    neighs = list(set(neighs))
                    new_sites = [pos for pos in neighs if self.is_occupied[pos[0], pos[1]] == 1]
                    for pos in new_sites:
                        self.is_occupied[pos[0], pos[1]] = 2
                        cluster_sites.append((pos[0], pos[1]))
                        if pos[1] == (self.Ny-1):
                            sp_cluster = True
                if sp_cluster:
                    if not leave_cluster_marked:
                        for site in cluster_sites:
                            self.is_occupied[site] = 1
                    return True, len(cluster_sites)
                else:
                    for site in cluster_sites:
                        self.is_occupied[site] = 1
        return False, 0
"""
    def plot_configuration(self):
        p.figure()
        p.imshow(self.is_occupied.T, interpolation='none', origin='lower') # transpose array so that the plot's x and y equal the first and second index of the array, resp.
"""        
class BondPercolator2D(Percolator2D):
    
    def connect(self, xi, yi, direction):
        initial = (xi, yi)
        if direction == 0:
            final = (xi + 1, yi)
        else:
            final = (xi, yi + 1)
        fermi_flag = (self.energy[initial] - self.ef)*(self.energy[final] - self.ef)
        if fermi_flag < 0:
            e_ij = abs(self.energy[final] - self.energy[initial]) - 1. / self.eps_r / 0.69508 # - potential energy in eV, to account for self-interaction
        else:
            e_ij = max(abs(self.energy[initial] - self.ef), abs(self.energy[final] - self.ef))
        #e_ij = 0.5*(abs(self.energy[final] - self.energy[initial]) + abs(self.energy[initial] - self.ef) + abs(self.energy[final] - self.ef))
        if e_ij < self.prob:
            return True
        else:
            return False
        
    
    def generate_configuration(self):
        self.energy = r.normal(0, 0.5, (self.Nx, self.Ny))
        don1 = (r.randint(self.Nx) + 0.5, r.randint(self.Ny) + 0.5)
        don2 = (r.randint(self.Nx) + 0.5, r.randint(self.Ny) + 0.5)
        el1 = (r.randint(self.Nx) + 0.5, r.randint(self.Ny) + 0.5)
        el2 = (r.randint(self.Nx) + 0.5, r.randint(self.Ny) + 0.5)
        for xi in range(self.Nx):
            for yi in range(self.Ny):
                self.energy[xi, yi] +=  10./np.sqrt((xi - el1[0])**2 + (yi - el1[1])**2) # fixed electron 1
                self.energy[xi, yi] +=  10./np.sqrt((xi - el2[0])**2 + (yi - el2[1])**2) # fixed electron 2
                self.energy[xi, yi] -=  10./np.sqrt((xi - don1[0])**2 + (yi - don1[1])**2) # fixed hole 1
                self.energy[xi, yi] -=  10./np.sqrt((xi - don2[0])**2 + (yi - don2[1])**2) # fixed hole 2
                #self.energy[xi, yi] -=  yi/self.Ny # electric field, positive @ y = Ny
        self.is_occupied = np.zeros((self.Nx, self.Ny, 2)) # 2nd dimension: 0: bond to the right, 1: bond up        
        for xi in range(self.Nx - 1):
            for yi in range(self.Ny - 1):                
                self.is_occupied[xi, yi, 0] = self.connect(xi, yi, 0) # bond to the right                
                self.is_occupied[xi, yi, 1] = self.connect(xi, yi, 1) # bond to the top
    
    def are_connected(self, pos1, pos2):
        """ Checks is two site positions pos1 and pos2 are connected via an open bond which was not already marked,
        returns flag and the connecting bond"""
        if (pos1[0] - pos2[0] == 1) and (pos1[1] == pos2[1]) and (self.is_occupied[pos2[0], pos2[1], 0] == 1):
            return True, (pos2[0], pos2[1], 0)
        elif (pos2[0] - pos1[0] == 1) and (pos1[1] == pos2[1]) and (self.is_occupied[pos1[0], pos1[1], 0] == 1):
            return True, (pos1[0], pos1[1], 0)
        elif (pos1[1] - pos2[1] == 1) and (pos1[0] == pos2[0]) and (self.is_occupied[pos2[0], pos2[1], 1] == 1):
            return True, (pos2[0], pos2[1], 1)
        elif (pos2[1] - pos1[1] == 1) and (pos1[0] == pos2[0]) and (self.is_occupied[pos1[0], pos1[1], 1] == 1):
            return True, (pos1[0], pos1[1], 1)
        else:
            return False, None
    
                   
    def spanning_cluster_exists(self, leave_cluster_marked = True):
        
        # choose starting site
        for xstart in range(self.Nx):
            sp_cluster = False
            done_site = np.zeros((self.Nx, self.Ny))
            cluster_bonds = []
            new_sites = [(xstart, 0)]
            while (len(new_sites) > 0) and (sp_cluster == False): # if there are no new sites left, terminate
                # collect neighbors
                neighs = []
                for site in new_sites:
                    neighs += self.neighbor_list(site[0], site[1])
                # check connections to neighbors
                #   -> if disconnected, dismiss
                #   -> if already marked bond, dismiss
                #   -> if unmarkedely connected, but already visited as starting site, mark bond, but don't keep site
                #   -> if unmarkedely connected, but not part of the cluster, mark bond and add site to new starting sites
                # check, if any of the new sites are at the top of the lattice -> found spanning cluster
                new_sites_temp = []
                for neighbor in neighs:
                    for site in new_sites:
                        conn_flag, conn_bond = self.are_connected(neighbor, site)
                        if conn_flag:
                            self.is_occupied[conn_bond] = 2
                            cluster_bonds.append(conn_bond)
                            if not done_site[neighbor]:
                                new_sites_temp.append(neighbor)
                                done_site[neighbor] = 1
                                if neighbor[1] == (self.Ny - 1):
                                    sp_cluster = True
                new_sites = list(new_sites_temp)    
            if not sp_cluster:
                for bond in cluster_bonds: # remove marks if cluster was not spanning
                    self.is_occupied[bond] = 1
            else:
                return True, len(cluster_bonds)
                
            # start again with new starting sites
        return False, 0
                   
"""                   
    def plot_configuration(self):
#        X, Y = p.meshgrid(range(self.Nx), range(self.Ny))
#        p.scatter(X, Y)
        
        colorcode = [None, "blue", "red"]
        for xi in range(self.Nx - 1):
            for yi in range(self.Ny - 1):
                if self.is_occupied[xi, yi, 0]:
                    p.plot([xi, xi + 1], [yi, yi], linewidth = 2, color = colorcode[int(self.is_occupied[xi, yi, 0])])
                if self.is_occupied[xi, yi, 1]:
                    p.plot([xi, xi], [yi, yi + 1], linewidth = 2, color = colorcode[int(self.is_occupied[xi, yi, 1])])
                    
        p.xlim((-1, self.Nx))
        p.ylim((-1, self.Ny))
"""


class SitePercolator3D(Percolator3D):

    def spanning_cluster_exists(self, leave_cluster_marked=True):
        """ Check for spanning cluster from y=0 to Ny-1 using flood fill """
        for startx in range(self.Nx):
            for starty in range(self.Ny):
                x = startx
                y = starty
                z = 0
                if self.is_occupied[x, y, z]:
                    self.is_occupied[x, y, z] = 2
                    new_sites = [(x, y, z)]
                    sp_cluster = False  # existence flag of spanning cluster
                    cluster_sites = [(x, y, z)]
                    while (len(new_sites) > 0):  # and (sp_cluster == False): # if only detection of interest, add second condition for speeding up. Then, not the complete cluster may be set to is_occupied = 2.
                        neighs = []
                        for site in new_sites:
                            neighs += self.neighbor_list(site[0], site[1], site[2])
                        neighs = list(set(neighs))
                        new_sites = [pos for pos in neighs if self.is_occupied[pos[0], pos[1], pos[2]] == 1]
                        for pos in new_sites:
                            self.is_occupied[pos[0], pos[1], pos[2]] = 2
                            cluster_sites.append((pos[0], pos[1], pos[2]))
                            if pos[2] == (self.Nz - 1):
                                sp_cluster = True
                    if sp_cluster:
                        if not leave_cluster_marked:
                            for site in cluster_sites:
                                self.is_occupied[site] = 1
                        return True, len(cluster_sites)
                    else:
                        for site in cluster_sites:
                            self.is_occupied[site] = 1
        return False, 0

class BondPercolator3D(Percolator3D):
    """ Bond percolation on a cubic lattice """
    
    def __init__(self, Nx, Ny, Nz, prob, ef, sigma, eps_r=3.):
        self.is_occupied = np.zeros((Nx, Ny, Nz))
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.prob = prob
        
        self.ef = ef        # Fermi level in eV
        self.sigma = sigma  # Width of intrinsic Gaussian DOS in eV
        self.eps_r = eps_r  # Relative permittivity, dimensionless
        
        # Load energy landscape from file generated by 3D_potential.py
        self.energy = r.normal(0, self.sigma, (self.Nx, self.Ny, self.Ny)) # empty DOS - load values from files
        print("Loading energy landscape ...")
        pos_energies = np.loadtxt("energies.dat")
        for entry in pos_energies:
            self.energy[int(entry[0]), int(entry[1]), int(entry[2])] = entry[3]
        print("Done.")
    
    def connect(self, xi, yi, zi, direction): # direction, 0: x++, 1: y++, 2: z++
        """ Decide, if bond from (xi, yi, zi) into 'direction' is connecting. If yes, return True. """
        initial = (xi, yi, zi)
        if direction == 0:
            final = (xi + 1, yi, zi)
        elif direction == 1:
            final = (xi, yi + 1, zi)
        elif direction == 2:
            final = (xi, yi, zi + 1)
        fermi_flag = (self.energy[initial] - self.ef)*(self.energy[final] - self.ef)
        if fermi_flag < 0:
            e_ij = abs(self.energy[final] - self.energy[initial]) - 1. / self.eps_r / 0.69508 # - potential energy in eV with r = 1 nm, to account for self-interaction (Efros, Shklovskii (10.1.17))
        else:
            e_ij = max(abs(self.energy[initial] - self.ef), abs(self.energy[final] - self.ef))
        #e_ij = 0.5*(abs(self.energy[final] - self.energy[initial]) + abs(self.energy[initial] - self.ef) + abs(self.energy[final] - self.ef))
        if e_ij < self.prob:
            return True
        else:
            return False
        
    
    def generate_configuration(self):
#        self.energy = r.normal(0, self.sigma, (self.Nx, self.Ny, self.Ny)) # Gaussian intrinsic DOS
        
#        for xi in range(self.Nx):
#            for yi in range(self.Ny):
#                for zi in range(self.Nz):
#                    for dpos in donor_positions:
#                        if (xi, yi, zi) == (dpos[0], dpos[1], dpos[2]):
#                            self.energy == - 10.
#                        else:
#                            self.energy[xi, yi, zi] -=  1./np.sqrt((xi - dpos[0])**2 + (yi - dpos[1])**2 + (yi - dpos[2])**2) / self.eps_r / 0.69508 # potential energy in eV, fixed electron
#                    for elpos in electron_positions:
#                        if (xi, yi, zi) == (dpos[0], dpos[1], dpos[2]):
#                            self.energy == 10.
#                        else:
#                            self.energy[xi, yi, zi] +=  1./np.sqrt((xi - elpos[0])**2 + (yi - elpos[1])**2 + (yi - elpos[2])**2) / self.eps_r / 0.69508 # potential energy in eV, fixed electron
                        
        self.is_occupied = np.zeros((self.Nx, self.Ny, self.Nz, 3)) # 3nd dimension: 0: bond to x++, 1: bond to y++, 2: bond to z++
        for xi in range(self.Nx - 1):
            for yi in range(self.Ny - 1): 
                for zi in range(self.Nz - 1):
                    self.is_occupied[xi, yi, zi, 0] = self.connect(xi, yi, zi, 0) # bond to x++                
                    self.is_occupied[xi, yi, zi, 1] = self.connect(xi, yi, zi, 1) # bond to y++
                    self.is_occupied[xi, yi, zi, 2] = self.connect(xi, yi, zi, 2) # bond to z++
    
    def are_connected(self, pos1, pos2):
        """ Checks is two site positions pos1 and pos2 are connected via an open bond which was not already marked,
        returns flag and the connecting bond"""
        if (pos1[0] - pos2[0] == 1) and (pos1[1] == pos2[1]) and (pos1[2] == pos2[2]) and (self.is_occupied[pos2[0], pos2[1], pos2[2], 0] == 1):
            return True, (pos2[0], pos2[1], pos2[2], 0)
        elif (pos2[0] - pos1[0] == 1) and (pos1[1] == pos2[1]) and (pos1[2] == pos2[2]) and (self.is_occupied[pos1[0], pos1[1], pos1[2], 0] == 1):
            return True, (pos1[0], pos1[1], pos1[2], 0)
        elif (pos1[1] - pos2[1] == 1) and (pos1[0] == pos2[0]) and (pos1[2] == pos2[2]) and (self.is_occupied[pos2[0], pos2[1], pos2[2], 1] == 1):
            return True, (pos2[0], pos2[1], pos2[2], 1)
        elif (pos2[1] - pos1[1] == 1) and (pos1[0] == pos2[0]) and (pos1[2] == pos2[2]) and (self.is_occupied[pos1[0], pos1[1], pos1[2], 1] == 1):
            return True, (pos1[0], pos1[1], pos1[2], 1)
        elif (pos2[2] - pos1[2] == 1) and (pos1[0] == pos2[0]) and (pos1[1] == pos2[1]) and (self.is_occupied[pos1[0], pos1[1], pos1[2], 2] == 1):
            return True, (pos1[0], pos1[1], pos1[2], 2)
        elif (pos1[2] - pos2[2] == 1) and (pos1[0] == pos2[0]) and (pos1[1] == pos2[1]) and (self.is_occupied[pos2[0], pos2[1], pos2[2], 2] == 1):
            return True, (pos2[0], pos2[1], pos2[2], 2)
        else:
            return False, None
    
                   
    def spanning_cluster_exists(self, leave_cluster_marked = True):
        """ Checks for spanning cluster between z = 0 and z = self.Nz-1 """
        # choose starting site from z=0-plane
        for xstart in range(self.Nx):
            for ystart in range(self.Ny):
                if self.is_occupied[xstart, ystart, 0, 2]: # don't waste time with starting sites that don't connect in z++ direction
                    sp_cluster = False
                    done_site = np.zeros((self.Nx, self.Ny, self.Nz))
                    cluster_bonds = []
                    new_sites = [(xstart, ystart, 0)]
                    while (len(new_sites) > 0) and (sp_cluster == False): # if there are no new sites left, terminate
                        # collect neighbors
                        neighs = []
                        for site in new_sites:
                            neighs += self.neighbor_list(site[0], site[1], site[2])
                        # check connections to neighbors
                        #   -> if disconnected, dismiss
                        #   -> if already marked bond, dismiss
                        #   -> if unmarkedely connected, but already visited as starting site, mark bond, but don't keep site
                        #   -> if unmarkedely connected, but not part of the cluster, mark bond and add site to new starting sites
                        # check, if any of the new sites are at the top of the lattice -> found spanning cluster
                        new_sites_temp = []
                        for neighbor in neighs:
                            for site in new_sites:
                                conn_flag, conn_bond = self.are_connected(neighbor, site)
                                if conn_flag:
                                    self.is_occupied[conn_bond] = 2
                                    cluster_bonds.append(conn_bond)
                                    if not done_site[neighbor]:
                                        new_sites_temp.append(neighbor)
                                        done_site[neighbor] = 1
                                        if neighbor[2] == (self.Nz - 1): # check for percolation in z-direction
                                            sp_cluster = True
                        new_sites = list(new_sites_temp)    
                    if not sp_cluster:
                        for bond in cluster_bonds: # remove marks if cluster was not spanning
                            self.is_occupied[bond] = 1
                    else:
                        return True, len(cluster_bonds)
                    
                    # start again with new starting sites
        return False, 0
                   
"""                   
    def plot_configuration(self):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
#        X, Y = p.meshgrid(range(self.Nx), range(self.Ny))
#        p.scatter(X, Y)
        
        colorcode = [None, "blue", "red"]
        for xi in range(self.Nx - 1):
            for yi in range(self.Ny - 1):
                for zi in range(self.Nz - 1):
                    if self.is_occupied[xi, yi, zi, 0]:
                        ax.plot3D([xi, xi + 1], [yi, yi], [zi, zi], linewidth = 2, color = colorcode[int(self.is_occupied[xi, yi, zi, 0])])
                    if self.is_occupied[xi, yi, zi, 1]:
                        ax.plot3D([xi, xi], [yi, yi + 1], [zi, zi], linewidth = 2, color = colorcode[int(self.is_occupied[xi, yi, zi, 1])])
                    if self.is_occupied[xi, yi, zi, 2]:
                        ax.plot3D([xi, xi], [yi, yi], [zi, zi + 1], linewidth = 2, color = colorcode[int(self.is_occupied[xi, yi, zi, 2])])
        
        ax.set_xlabel("x in nm")
        ax.set_ylabel("y in nm")
        ax.set_zlabel("z in nm")
#        p.xlim((-1, self.Nx))
#        p.ylim((-1, self.Ny))
"""
        
class BondVRHPercolator3D(Percolator):
    
    def __init__(self, prob, kT, alpha, lamb, ef, eps_r, L, shrink_by, start_width, filename):
        self.prob = prob  
        self.kT = kT                # in eV
        self.alpha = alpha          # localization radius in nm
        self.reorg_lambda = lamb    # single-molecule reorganization energy in eV
        self.ef = ef                # Fermi level in eV
        self.eps_r = eps_r          # Relative permittivity, dimensionless
        self.L = L                  # linear sample size (sample is LxLxL cube)
        self.shrink_by = shrink_by  # fraction of L that should be cut away at the border
        self.zmin = self.L*shrink_by/2. # renormalization of zero z-position due to cutting out a cube
        self.start_width = start_width # width of starting and target sites near z = 0 and z = L face
        self.spanning_cluster_bonds = []
        
        self.filename = filename # Name of file containing positions and energies
        
    def load_positions(self, pos_energies=None):
        """ Load energy landscape from file generated by 3D_potential.py """
#        print("Loading sites and energy landscape ...")
        if pos_energies == None:
            pos_energies = np.loadtxt(self.filename)
        if self.shrink_by > 0: # option to take only a subvolume of the sample for the percolation analysis to diminish border effects
            pos_energies = pos_energies[np.where( np.all( np.abs(pos_energies[:,:3]-self.L * np.array([0.5, 0.5, 0.5])) < (self.L*(1.-self.shrink_by)/2.),  axis=1))[0] ]
            self.L = self.L*(1.-self.shrink_by)
        self.positions = pos_energies[:,:3] # 0, 1, 2: coordinates, 3: energy
        self.energy = pos_energies[:,3]
        self.N = len(self.energy) # number of sites
        self.start_sites = []
        self.target_sites = []
        for i in range(self.N):
            if self.positions[i,2] < (self.zmin + self.start_width):
                self.start_sites.append(i)
            if self.positions[i,2] > (self.zmin + self.L - self.start_width): # elif to only put them in one of the two lists
                self.target_sites.append(i)
        print("{0} starting sites, {1} target sites".format(len(self.start_sites), len(self.target_sites)))
#        print("Done.")
        
    def connect_MA(self, initial, final):
        """ Decide, if bond between site initial and final is connecting. If yes, return True. 
            Uses Miller-Abrahams hopping rate. """ 
        d = dist(self.positions[initial], self.positions[final])
        fermi_flag = (self.energy[initial] - self.ef)*(self.energy[final] - self.ef)
        if fermi_flag < 0:
            e_ij = abs(self.energy[final] - self.energy[initial]) - 1. / d / self.eps_r / 0.69508 # - potential energy in eV, if dist() is in nm, to account for self-interaction (Efros, Shklovskii (10.1.17))
        else:
            e_ij = max(abs(self.energy[initial] - self.ef), abs(self.energy[final] - self.ef))
        if (2.*d/self.alpha + e_ij/self.kT) < self.prob:
            return True
        else:
            return False
            
    def connect(self, initial, final):
        """ Decide, if bond between site initial and final is connecting. If yes, return True. 
            Uses Marcus hopping rate. """ 
#        d = dist(self.positions[initial], self.positions[final])
        d = self.positions[final] - self.positions[initial]
        d = np.sqrt(sum(d*d))
        fermi_flag = (self.energy[initial] - self.ef)*(self.energy[final] - self.ef)
        if fermi_flag < 0:
            e_ij = abs(self.energy[final] - self.energy[initial]) - 1. / d / self.eps_r / 0.69508 # - potential energy in eV, if dist() is in nm, to account for self-interaction (Efros, Shklovskii (10.1.17))
        else:
            e_ij = max(abs(self.energy[initial] - self.ef), abs(self.energy[final] - self.ef))
        if (2.*d/self.alpha + (e_ij + self.reorg_lambda)**2/(self.kT*4.*self.reorg_lambda)) < self.prob:
            return True
        else:
            return False
    
    def conn_list(self, i):
        """ Finds all sites connected to site i that were not previously tested. """
        for j in range(i+1, self.N):
            if self.connect(i, j):
                self.connected[i].append(j)
                self.connected[j].append(i)
   
    def generate_configuration(self, pos_en=None):
        self.load_positions(pos_en)
#        N4 = int(self.N/4)
        self.connected = [[] for j in range(self.N)] # construct lists of connected sites for each site
        print("Constructing bond configuration ...")
        #perc_flag = -1
        self.connected = cconnections.get_connections(self.N, self.positions, self.energy, self.ef, self.eps_r, self.alpha, self.reorg_lambda, self.kT, self.prob)

        """
        def conn_list_parallel(obj, istart, istop):
             Finds all sites connected to sites istart to istop-1 that were not previously tested. 
            conne = [[] for j in range(obj.N)]
            for i in range(istart,istop):
                for j in range(i+1, obj.N):
                    if obj.connect(i, j):
                        conne[i].append(j)
                        conne[j].append(i)
            return conne
        """    
#        job_server_loc = pp.Server(secret = "02559", ncpus=4)
#        job_list_conn = [job_server_loc.submit(conn_list_parallel, args=(self,ist, self.N - (3-count)*N4), modules=("numpy as np",)) for count, ist in enumerate([0, N4, 2*N4, 3*N4])]
#        result_list = [j() for j in job_list_conn]
#        for res in result_list:
#            self.connected = [self.connected[j] + res[j] for j in range(self.N)]
        # for i in range(self.N):
        #     self.conn_list(i)
        #     percent = round(100*i/self.N)
        #     if percent % 10 == 0 and percent > perc_flag:
        #         print("{0} % finished".format(percent))
        #         perc_flag = percent
        print("Done.")
    
    def spanning_cluster_exists_cy(self, leave_cluster_marked = True):
        """ Checks for spanning cluster between z = [0,self.start_width] and z = [L-self.start_width, L] """
        return cconnections.spanning_cluster_exists(self.N, self.connected, np.array(self.start_sites, dtype=np.int), np.array(self.target_sites, dtype=np.int))
    
    def spanning_cluster_exists(self, leave_cluster_marked = True):
        """ Checks for spanning cluster between z = [0,self.start_width] and z = [L-self.start_width, L] """
        # choose starting site from starting list
        for istart in self.start_sites:
#            print("Checking starting site {0}".format(istart))
            sp_cluster = False
            done_site = np.zeros(self.N)
            cluster_bonds = []
            new_sites = [istart]
            while len(new_sites) > 0:# and (sp_cluster == False): # if there are no new sites left, terminate
                # check connections to neighbors
                #   -> if disconnected, dismiss
                #   -> if already marked bond, dismiss
                #   -> if unmarkedely connected, but already visited as starting site, mark bond, but don't keep site
                #   -> if unmarkedely connected, but not part of the cluster, mark bond and add site to new starting sites
                # check, if any of the new sites are at the top of the lattice -> found spanning cluster
                    
                new_sites_temp = []
                for site in new_sites:
                    for temp_conn in self.connected[site]:
                        if not ((temp_conn, site) in cluster_bonds):
                            cluster_bonds.append((site, temp_conn))
                            if not done_site[temp_conn]:
                                new_sites_temp.append(temp_conn)
                                done_site[temp_conn] = 1                                
                                if temp_conn in self.target_sites:
                                    sp_cluster = True
                new_sites = list(new_sites_temp)
#                print("New sites: {0}".format(new_sites))
            if sp_cluster:
                self.spanning_cluster_bonds = list(cluster_bonds)
                return True, len(cluster_bonds)
            # start again with new starting site
                
        return False, 0


    def plot_configuration(self, CT_ids):
        mi = min(self.energy)
        ma = max(self.energy)
        normalized_energies = (self.energy - mi)/(ma-mi)
#        normalized_energies = 1./normalized_energies
#        mi = min(normalized_energies)
#        ma = max(normalized_energies)
#        normalized_energies = (normalized_energies - mi)/(ma-mi)
        colors = np.array([cm.seismic(1.-x) for x in normalized_energies])
        
        cluster_set = set([i for (i,j) in self.spanning_cluster_bonds] + [j for (i,j) in self.spanning_cluster_bonds])
        n_CT_cluster_sites = len(cluster_set.intersection(set(CT_ids)))
        print("Fraction of CTs in transport sites: {0}".format(float(n_CT_cluster_sites)/len(cluster_set)))
        cluster_sites= list(cluster_set)
        pos = np.array(self.positions)        

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        
        # draw sample cube
        r = [0., self.L]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="black")        
        
#        ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], 'o', s=100, c = colors)
        ax.scatter(pos[cluster_sites,0], pos[cluster_sites,1], pos[cluster_sites,2], 'o', s=100, c = colors[cluster_sites])
        for i, j in self.spanning_cluster_bonds:
            ax.plot3D([self.positions[i, 0], self.positions[j, 0]], [self.positions[i, 1], self.positions[j, 1]], [self.positions[i, 2], self.positions[j, 2]], linewidth = 2, color = 'red')
        if len(self.spanning_cluster_bonds) > 0:
            kmax = 0
            max_e = 0
            for l, (i,j) in enumerate(self.spanning_cluster_bonds):
                if abs(self.energy[j] - self.energy[i]) > max_e:
                    max_e = abs(self.energy[j] - self.energy[i])
                    kmax = l
            print("Max. jump energy = {0} eV".format(max_e))
            i = self.spanning_cluster_bonds[kmax][0]
            j = self.spanning_cluster_bonds[kmax][1]
            ax.plot3D([self.positions[i, 0], self.positions[j, 0]], [self.positions[i, 1], self.positions[j, 1]], [self.positions[i, 2], self.positions[j, 2]], linewidth = 5, color = 'green')
        
        scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
        
        ax.set_xlabel("x in nm")
        ax.set_ylabel("y in nm")
        ax.set_zlabel("z in nm")
        
        fig, (ax3, ax4, ax5) = plt.subplots(3, 1, sharex=True)
        plt.suptitle("Energetic distributions")
        # get histogram of occupied + unoccupied states
        ax3.hist(self.energy, bins=(ma-mi)/0.01+1, normed=False)
        ax3.set_ylabel("Total count of states (normed)")
        plt.tight_layout()
        
        if len(self.spanning_cluster_bonds) > 0:
            start_sites =  []
            target_sites =  []
            for i, j in self.spanning_cluster_bonds:
                if self.energy[i] < self.energy[j]:
                    start_sites.append(i)
                    target_sites.append(j)
                else:
                    start_sites.append(j)
                    target_sites.append(i)
            target_sites = list(set(target_sites))
            start_sites = list(set(start_sites))
    #        cluster_sites = set([i for i in self.spanning_cluster_bonds[:,0]] + [i for i in self.spanning_cluster_bonds[:,1]])
            ax4.hist(self.energy[start_sites], bins = (ma-mi)/0.01 +1)
#            ax5.hist(self.energy[target_sites], bins = (ma-mi)/0.01 +1)
            ax5.hist(self.energy[start_sites + target_sites], bins = (ma-mi)/0.01 +1)
        ax4.set_ylabel("Start states")
#        ax5.set_ylabel("Target states")
        ax5.set_ylabel("All transport states")
        ax5.set_xlabel("Energy in eV")
        
        
        plt.tight_layout()
        plt.show()
#        p.xlim((-1, self.Nx))
#        p.ylim((-1, self.Ny))

    def find_sigma(self):
        mi = min(self.energy)
        ma = max(self.energy)
        DOS, es = np.histogram(self.energy, bins=(ma-mi)/0.01+1, normed=True)
        maxDOSind = np.argmax(DOS)
        maxDOS_2 = DOS[maxDOSind] / 2. # Half maximum value
        i = maxDOSind
        while DOS[i] > maxDOS_2:
            i -= 1 # search half maximum to the left, where transport happens
        if DOS[i-1] == DOS[i]:
            halfmpos = es[i]
        else:
            halfmpos = es[i] + (maxDOS_2 - DOS[i])*(es[i+1] - es[i]) / (DOS[i-1] - DOS[i]) # interpolate beween the bracketing points
        sig = (es[maxDOSind] - halfmpos) / (np.sqrt(2.*np.log(2.))) # get standard deviation from half width at half maximum
        return sig
        

if __name__ == '__main__':
    main()