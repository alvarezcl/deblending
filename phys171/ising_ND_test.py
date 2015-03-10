## Simulate the ising Model in any dimension using the Metropolis Algorithm

import numpy as np
import pandas as pd
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
def assign_spins(n):
    p = np.random.random_sample(size=(n,))
    p[p >= 0.5] = 1
    p[p < 0.5] = -1
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
        if plus > (n-1): plus = plus - n
        if minus < 0: minus = n - np.abs(minus)
        
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

# Calculate the total internal energy of the system
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
    
    return (-k*energy_NN - energy_mag), np.sum(spins)
    
# Calculate the energy change of flipping one spin
def calc_energy_change(site_i,k,h,
                       coord,spins,n,dim,
                       neighbor):
                           
    # Copy from the previous array of spins
    new_spins = np.copy(spins)
    # Flip the spin
    new_spins[site_i] = -new_spins[site_i]
    # Calculate the energy change
    NearNeigh, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,neighbor)
    energy_change = -2*k*(np.sum(new_spins[site_i]*Near_spins))
    
    return energy_change, new_spins
    
if __name__ == '__main__':
    
    # ---------------------- Initialization ------------------------ #
    # Parameters
    kb = 1                # kb = 1.3806503 * 10**(-23)
    # Coupling Constant
    J = 2
    # Magnetic field
    h = 0
    # Temperature
    T = 1
    # Effective Coupling
    k = J/(kb*T)
    # Size of Lattice
    n = 10
    # Dimension
    dim = 3
    # Number of Spins
    spin_num = n**dim
    # Nearest neighbor coupling
    neighbor = 1
    # RNG
    rng = np.random.seed(seed=1)
    
    # ---------------------- Spins and Coordinates -------------------- #
    # Create Spins with Coordinate Array
    coord = create_coord(n,dim)
    # Assign spins to array
    spins = assign_spins(spin_num)
    
    # ---------------------- Initial Magnetization and Energy --------------#
    site_i = (n)**2 - 2
    NN, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,neighbor)    
    
    
    init_energy, total_spin = calc_energy_config(k,h,
                                                 coord,spins,n,dim,
                                                 neighbor)
                                                 
    energy_change, new_spins = calc_energy_change(site_i,k,h,
                                                  coord,spins,n,dim,
                                                  neighbor)
                                                  
    new_energy, new_tot_spin = calc_energy_config(k,h,
                                                  coord,new_spins,n,dim,
                                                  neighbor)
                                                  
                    