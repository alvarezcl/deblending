## Simulate the ising Model in any dimension using the Metropolis Algorithm
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

# First, create the spin arrays in a given dimension
def create_coord(n,dim):
    # Error checking for faulty inputs
    assert n > 0 and dim > 0, "n and d must be greater than 0."
    # Total array size is (n-1)**dim x dim
    return fill_coordinates(n,dim)
    
# Create the coordinate array for a given length and dimension
def fill_coordinates(n,d):
    # Fill array according to length and dimension
    if (n > 255): raise ValueError("n above resolution limit.")
    result = np.zeros((n**d,d),dtype=np.int8)
    count = 0
    # Store the coordinates of any spin in a 2D array
    # Any nearest neighbor in k dimensions consists of the spin with coordinates
    # (a,b,...,k) is the set of spins at {(a+/-1,b,...k),...,(a,b,...k+/-1)}
    if d == 1:
        for i in xrange(0,n):
            result[count][0] = i
            count = count + 1
    elif d == 2:
        for i in xrange(0,n):
            for j in xrange(0,n):
                result[count][0] = i
                result[count][1] = j
                count = count + 1
    elif d == 3:
        for i in xrange(0,n):
            for j in xrange(0,n):
                for k in xrange(0,n):
                    result[count][0] = i
                    result[count][1] = j
                    result[count][2] = k
                    count = count + 1
    elif d == 4:
        for i in xrange(0,n):
            for j in xrange(0,n):
                for k in xrange(0,n):
                    for l in xrange(0,n):
                        result[count][0] = i
                        result[count][1] = j
                        result[count][2] = k
                        result[count][3] = l
                        count = count + 1
    elif d == 5:
        for i in xrange(0,n):
            for j in xrange(0,n):
                for k in xrange(0,n):
                    for l in xrange(0,n):
                        for m in xrange(0,n):
                            result[count][0] = i
                            result[count][1] = j
                            result[count][2] = k
                            result[count][3] = l
                            result[count][4] = m
                            count = count + 1
    elif d > 5:
        raise ValueError("Dimension cannot be greater than 5")
    
    return result
    
# Assign spins to an array by using uniform sampling
def assign_spins(n,all_ones=False,seed=0):
    np.random.seed(seed=seed)
    p = np.random.random_sample(size=(n,))
    p[p >= 0.5] = 1
    p[p < 0.5] = -1
    if all_ones == True:
        return np.ones((n,),dtype=np.int8)
    return np.array(p,dtype=np.int8)
   
# Given a lattice site, return all nearest neighbors coordinates and
# spins on each site. Which nearest neighbor coupling (1st,2nd..) is 
# noted by "neighbor_coupling"
def calc_neighbors(site_i,coord,spins,n,dim,neighbor_coupling):
    # Extract all nearest neighbors
    # Obtain the coordinates of each nearest neighbor
    num_NN = 2*dim
    # Store the results in a result array 
    result_coord = np.zeros((num_NN,dim))
    result_spins = np.zeros((num_NN,1))
    # Get the coordinates of the ith site
    site_coord = coord[site_i]
    # Run through the + and - for each scalar value in the vector in site_coord
    count = 0
    for i in range(0,dim):
        assert count <= num_NN, "Accessing more than nearest neighbors values."
        site_coord_i = site_coord[i]
        plus = site_coord_i + neighbor_coupling
        minus = site_coord_i - neighbor_coupling

        # Implement periodic boundaries
        if (plus > (n-1)): plus = plus - n
        if (minus < 0): minus = n - np.abs(minus)
        
        # Store the coordinates
        result_coord[count] = site_coord
        result_coord[count][i] = minus
        # Store the spin value
        spin_index = np.where(np.all(result_coord[count]==coord,axis=1))[0][0]
        result_spins[count] = spins[spin_index]
        count = count + 1
        
        # Store the coordinates
        result_coord[count] = site_coord
        result_coord[count][i] = plus
        # Store the spin value
        spin_index = np.where(np.all(result_coord[count]==coord,axis=1))[0][0]
        result_spins[count] = spins[spin_index]
        count = count + 1

        
    return np.array(result_coord,dtype=np.int8), np.array(result_spins,dtype=np.int8)

# Calculate the total internal energy of the system and return the total
# magnetization per spin
def calc_energy_config(k,h,
                       coord,spins,n,dim,
                       neighbor):
                
    # Neighbor energy
    energy_NN = 0               
    # For each spin, calculate the neighbor coordinates and spins and sum
    for i,spin in np.ndenumerate(spins):
        NearNeigh, Near_spins = calc_neighbors(i,coord,spins,n,dim,neighbor)
        # Perform sum
        energy_i = np.sum(spin*Near_spins)
        # Add energy contribution from sum
        energy_NN = energy_NN + energy_i
        # Divide by two for double counting
        #TODO Check if double counting is correct to do
    energy_NN = energy_NN/2 
    
    # Magnetic energy
    energy_mag = h*np.sum(spins)
    
    return (-k*energy_NN - energy_mag), np.sum(spins)/(n**dim)
    
# Calculate the energy change of flipping one spin at site_i
def calc_energy_change(site_i,k,h,
                       coord,spins,n,dim,
                       neighbor):
                           
    # Copy from the previous array of spins
    new_spins = np.copy(spins)
    old_spin = spins[site_i]
    # Flip the spin
    new_spins[site_i] = -old_spin
    new_spin = new_spins[site_i]
    # Calculate the energy change
    NearNeigh, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,neighbor)
    # A component in the neighbor term with magnetic field term
    energy_change = -2*k*(np.sum(new_spins[site_i]*Near_spins)) - h*(new_spin-old_spin)
    
    return energy_change, new_spins

if __name__ == '__main__':
    
    # ---------------------- Initialization --------------------------- #

    # Independent Parameters    
    kb = 1 # 1.3806503 * 10**(-23)
    # Temp
    T = 1
    # Beta
    B = 1/(kb*T)
    # Coupling Constant
    J = 1
    # Magnetic field
    H = 0
    # Chem Potential
    mu = 1
    # Size of Lattice
    n = 20
    # Dimension
    dim = 2
    # Number of spins
    spin_num = n**dim
    # Nearest neighbor coupling
    neighbor = 1
    # Integer for seed
    seed_int_one = 1
    seed_int_two = 2
    # Number of MC trials
    MC_trials = 10000
    
    # Effective values
    k = J*B
    h = mu*H*B    
    

    # ---------------------- Spins and Coordinates -------------------- #
    # Create Spins with Coordinate Array
    coord = create_coord(n,dim)
    # Assign spins to array
    spins = assign_spins(spin_num,all_ones=True,seed=seed_int_one)

    # ---------------------- Initial Magnetization and Energy --------- #
    
    init_energy, tot_mag_per_spin = calc_energy_config(k,h,
                                                       coord,spins,n,dim,
                                                       neighbor)                                                  
    
    T = np.linspace(0.001,5,100)
    mag_whole = []
    for t in T:
        mag = []
        B = 1/(kb*t)
        k = J*B
        h = mu*H*B
        for i in xrange(0,MC_trials):
            site_i = np.random.randint(0,spin_num)
            energy_change, new_spins = calc_energy_change(site_i,k,h,
                                                          coord,spins,n,dim,
                                                          neighbor)
            if energy_change <= 0:
                spins = new_spins
            elif (np.exp(-B*energy_change) > np.random.random()):
                spins = new_spins
                
            mag.append((np.sum(spins)/spin_num))
        mean_mag = np.sum(mag)/MC_trials                                             
        mag_whole.append(mean_mag)
    
    mag_whole = pd.DataFrame(mag_whole)
    plt.plot(T,mag_whole)
    plt.show()