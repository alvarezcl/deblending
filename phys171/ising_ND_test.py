## Simulate the ising Model in any dimension using the Metropolis Algorithm
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    if (n > 1000): raise ValueError("n above resolution limit.")
    result = np.zeros((n**d,d),dtype=np.int16)
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

        
    return np.array(result_coord,dtype=np.int16), np.array(result_spins,dtype=np.int16)

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

# Given the spins and coordinates, calculate the spin-spin correlation by 
# choosing s0 to be the center, and running through the lattice axes.
def calc_spin_spin(coord,spins,n,dim,
                   neighbor):
    center_coord = np.floor(n/2)*np.ones(dim,dtype=np.int16)
    # Obtain the spin value of the center. This will be s_0
    spin_index_center = np.where(np.all(center_coord==coord,axis=1))[0][0] 
    s0 = spins[spin_index_center]
    # Run through all spins at the lattice axes up to some 
    # distance r to store s0*sr. Let r = n/2 so we reach the boundary
    spin_spin = []
    for i in xrange(1,np.int(n/2)):
        # Obtain the spin values of the ith neighbor (this will have the effect
        # of running through the lattice axes).
        NearNeigh, Near_spins = calc_neighbors(spin_index_center,coord,spins,n,dim,
                                               i)
        # Average over all the neighbor spins
        s0si = s0*np.mean(Near_spins)
        spin_spin.append(s0si)
    spin_spin = np.array(spin_spin)
    return spin_spin, s0

# Run the metropolis algorithm 
def run_Metropolis(T,kb,J,H,mu,
                   coord,spins,n,dim,
                   neighbor,
                   MC_trials,Equib_trials,interval,
                   seed_int):
    print "Beginning Metropolis..."
    # Set the seed
    np.random.seed(seed=seed_int)
    # For magnetization and spin_spin data
    mag_whole = []
    spin_spin_temp = {}
    s0_arr = []
    # Magnetic Susceptibility
    chi = {}
    # Heat Capacity
    Hc = {}
    # For the temperatures of interest
    for t in T:
        print "Temperature =",t
        mag = []
        energy = []
        B = 1/(kb*t)
        k = J*B
        h = mu*H*B
        count = 0
        spin_spin_whole = np.zeros(len(xrange(1,np.int(n/2))))
        # Calculate MC trials and take data for each trial
        for i in xrange(0,MC_trials + Equib_trials):
            if (np.mod(i,(MC_trials)/10)==0): print i
            # Choose a random site
            site_i = np.random.randint(0,spin_num)
            # Calculate the energy change
            energy_change, new_spins = calc_energy_change(site_i,k,h,
                                                          coord,spins,n,dim,
                                                          neighbor)
            # Accept if energy change is negative, else check boltzmann weight
            if energy_change <= 0:
                spins = new_spins
            elif (np.exp(-B*energy_change) > np.random.random()):
                spins = new_spins
            # If equilibrium has been reached at t
            if (i >= Equib_trials and np.mod(i,interval)==0):
                # Obtain the magnetization per site
                mag.append((np.sum(spins)/spin_num))
                energy_config, tot_spin_per_site = calc_energy_config(k,h,
                                                                      coord,spins,n,dim,
                                                                      neighbor)
                energy.append(energy_config)

                # Obtain the spin-spin correlation
                spin_spin, s0 = calc_spin_spin(coord,spins,n,dim,
                                           neighbor)
                count = count + 1
                spin_spin_whole = (spin_spin_whole + spin_spin)/count
                s0_arr.append(s0)

        spin_spin_whole = spin_spin_whole/np.mean(np.array(s0_arr)**2)        
        spin_spin_temp[str(t)] = spin_spin_whole
        # Obtain heat capacity
        energy = np.array(energy)
        mag = np.array(mag)
        Hc[str(t)] = np.mean((1/np.sqrt(t*n**dim)*energy)**2) - (np.mean((1/np.sqrt(t*n**dim)*energy)))**2        
        # Obtain chi 
        chi[str(t)] = np.mean((1/np.sqrt(t*n**dim)*mag)**2) - (np.mean((1/np.sqrt(t*n**dim)*np.abs(mag))))**2
        # Obtain the mean magnetization per site
        mag_whole.append(np.sum(mag)/len(mag))
    # Convert to Pandas Object                          
    mag_whole = pd.DataFrame(mag_whole,columns=['mag_whole_met'])
    spin_spin_temp = pd.DataFrame(spin_spin_temp)
    chi = pd.Series(chi)
    Hc = pd.Series(Hc)
    print "Ending Metropolis."    
    return mag_whole, spin_spin_temp, chi, Hc

# Run the Wolff Cluster algorithm
def run_Wolff(T,kb,J,H,mu,
              coord,spins,n,dim,
              neighbor,
              MC_trials,Equib_trials,interval,
              seed_int):
    print "Beginning Wolff..."
    # Set the seed
    np.random.seed(seed=seed_int)
    # For magnetization and spin_spin data
    mag_whole = []
    spin_spin_temp = {}
    s0_arr = []
    # Magnetic Susceptibility
    chi = {}
    # Heat Capacity
    Hc = {}
    # For temperatures of interest
    for t in T:
        print "Temperature =",t
        mag = []
        energy = []
        B = 1/(kb*t)
        k = J*B
        h = mu*H*B
        count = 0
        spin_spin_whole = np.zeros(len(xrange(1,np.int(n/2))))
        # Keep track of the cluster and the stack
        stack = []
        cluster = []
        # Keep track of spins that have interacted
        interaction_matrix = np.zeros((n**dim,n**dim),dtype=np.int32)
        
        for i in xrange(0,MC_trials + Equib_trials):
            if (np.mod(i,(MC_trials)/10)==0): print i
            # Choose a random site
            site_i = np.random.randint(0,spin_num)
            # Obtain the neighbors and spins
            NearNeigh, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,
                                                   neighbor)
            # Probability of inclusion in cluster for neighbors
            p = 1 - np.exp(-2*k)
            # See if each spin is parallel to that at site_i and if they have
            # interacted
            # For all neighbor spins
            for (m,z),val in np.ndenumerate(Near_spins):
                # Calculate the spin index
                val_site = np.where(np.all(NearNeigh[m]==coord,axis=1))[0][0]
                # Check if the spins are parallel and haven't interacted
                if ((val == spins[site_i]) and ((interaction_matrix[val_site,site_i] == 0) or
                (interaction_matrix[site_i,val_site] == 0))):
                    # Update interaction matrix
                    interaction_matrix[val_site,site_i] = 1
                    interaction_matrix[site_i,val_site] = 1
                    # TODO check if less than or greater than
                    # Add to cluster and stack then invert spin
                    if (np.random.random() > p):
                        cluster.append(val_site)
                        stack.append(val_site)
                        spins[val_site] = -spins[val_site]

            # Now while the stack is not empty
            while (len(stack) > 0):
                # Remove from the stack
                site_i = stack.pop()
                # Redo for each spin in the stack, as long as the stack is
                # not empty
                # Obtain the neighbors and spins
                NearNeigh, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,
                                                       neighbor)
                # Probability of inclusion in cluster for neighbors
                p = 1 - np.exp(-2*k)
                # See if each spin is parallel to that at site_i and if they have
                # interacted
                # For all neighbor spins
                for (a,b),val in np.ndenumerate(Near_spins):
                    # Calculate the spin index 
                    val_site = np.where(np.all(NearNeigh[a]==coord,axis=1))[0][0]
                    # Check if the spins are parallel and haven't interacted
                    if ((val == spins[site_i]) and ((interaction_matrix[val_site,site_i] == 0) or
                    (interaction_matrix[site_i,val_site] == 0))):
                        # Update interaction matrix
                        interaction_matrix[val_site,site_i] = 1
                        interaction_matrix[site_i,val_site] = 1
                        # TODO check if less than or greater than
                        # Add to cluster and stack then invert spin
                        if (np.random.random() > p):
                            cluster.append(val_site)
                            stack.append(val_site)
                            spins[val_site] = -spins[val_site]
                        # If equilibrium has been reached at t
            if (i >= Equib_trials and np.mod(i,interval)==0):
                # Now the stack is empty
                # Calculate observable on spins
                mag.append((np.sum(spins)/spin_num))
                energy_config, tot_spin_per_site = calc_energy_config(k,h,
                                                                      coord,spins,n,dim,
                                                                      neighbor)
                energy.append(energy_config)

                # Obtain the spin-spin correlation
                spin_spin, s0 = calc_spin_spin(coord,spins,n,dim,
                                           neighbor)
                count = count + 1
                spin_spin_whole = (spin_spin_whole + spin_spin)/count
                s0_arr.append(s0)

        spin_spin_whole = spin_spin_whole/np.mean(np.array(s0_arr)**2)
        spin_spin_temp[str(t)] = spin_spin_whole
        # Done with MC trials now
        # Obtain heat capacity
        energy = np.array(energy)
        mag = np.array(mag)
        Hc[str(t)] = np.mean((1/np.sqrt(t*n**dim)*energy)**2) - (np.mean((1/np.sqrt(t*n**dim)*energy)))**2        
        # Obtain chi 
        chi[str(t)] = np.mean((1/np.sqrt(t*n**dim)*mag)**2) - (np.mean((1/np.sqrt(t*n**dim)*np.abs(mag))))**2
        # Obtain the mean magnetization per site
        mag_whole.append(np.sum(mag)/len(mag))
    # Convert to Pandas Object                                             
    mag_whole = pd.DataFrame(mag_whole,columns=['mag_whole_wolff'])
    spin_spin_temp = pd.DataFrame(spin_spin_temp)
    chi = pd.Series(chi)
    Hc = pd.Series(Hc)
    print "Ending Wolff."    
    return mag_whole, spin_spin_temp, chi, Hc

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
    seed_int_three = 3
    # Number of MC trials
    MC_trials = 2000
    # Number of Equilibrium Trials
    Equib_trials = 100
    # Effective values
    k = J*B
    h = mu*H*B    
    # Intervals at which to sample observable in MC iteration
    interval = 10

    # ---------------------- Spins and Coordinates -------------------- #
    # Create Spins with Coordinate Array
    coord = create_coord(n,dim)
    # Assign spins to array
    spins = assign_spins(spin_num,all_ones=True,seed=seed_int_one)                                                  
    
    # ---------------------- Metropolis ------------------------------- #
    T_end = 4
    num_interval = 8
    T = np.linspace(0.001,T_end,num_interval)

    mag_whole_met, two_point_met, chi_met, Hc_met = run_Metropolis(T,kb,J,H,mu,
                                                                   coord,spins,n,dim,
                                                                   neighbor,
                                                                   MC_trials,Equib_trials,interval,
                                                                   seed_int_two)
    
    # ---------------------- Wolff Cluster ---------------------------- #
    
    mag_whole_wolff, two_point_wolff, chi_w, Hc_w = run_Wolff(T,kb,J,H,mu,
                                                              coord,spins,n,dim,
                                                              neighbor,
                                                              MC_trials,Equib_trials,interval,
                                                              seed_int_two)
    pickle = False
    gs = gridspec.GridSpec(20,2)
    fig = plt.figure(figsize=(20,11))
    fs = 15
    
    ax1 = fig.add_subplot(gs[0:8,0])     
    plt.title('Magnetization Vs Temperature',fontsize=fs)
    plt.plot(T,mag_whole_met,T,mag_whole_wolff)
    plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\frac{<M>}{N}$',fontsize=fs)
    plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})
    plt.xlim([0,T_end])

    Temp = T[4]
    
    ax2 = fig.add_subplot(gs[10:18,0])
    plt.title('Two-Point Correlation',fontsize=fs)
    plt.plot(xrange(1,np.int(n/2)),np.abs(two_point_met[str(Temp)]),
             xrange(1,np.int(n/2)),np.abs(two_point_wolff[str(Temp)]))
    plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$<s_os_r>$',fontsize=fs)
    plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})

    ax3 = fig.add_subplot(gs[0:8,1])
    plt.title('Specific Heat',fontsize=fs)
    T_hc = np.linspace(0.001,T_end,len(Hc_met))
    plt.plot(T_hc,Hc_met,T_hc,Hc_w)
    plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$C$',fontsize=fs)
    plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})

    ax4 = fig.add_subplot(gs[10:18,1])
    plt.title('Magnetic Susceptibility',fontsize=fs)
    T_chi = np.linspace(0.001,T_end,len(chi_met))
    plt.plot(T_chi,chi_met,T_chi,chi_w)
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%.1f" % x,locs*1e6))
    plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\chi\/(1E-6)$',fontsize=fs)
    plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})

    plt.show()

    if pickle == True:
        mag_whole_met.to_pickle('data/mag_whole_met' + str(n**dim))    
        two_point_met.to_pickle('data/two_point_met' + str(n**dim))
        chi_met.to_pickle('data/chi_met' + str(n**dim))
        Hc_met.to_pickle('data/Hc_met' + str(n**dim))
    
        mag_whole_wolff.to_pickle('data/mag_whole_wolff' + str(n**dim))    
        two_point_wolff.to_pickle('data/two_point_wolff' + str(n**dim))
        chi_w.to_pickle('data/chi_w' + str(n**dim))
        Hc_w.to_pickle('data/Hc_w' + str(n**dim))
    else:
        mag_whole_met.to_pickle('data/mag_whole_met' + str(n**dim) + '.csv')    
        two_point_met.to_pickle('data/two_point_met' + str(n**dim) + '.csv')
        chi_met.to_pickle('data/chi_met' + str(n**dim) + '.csv')
        Hc_met.to_pickle('data/Hc_met' + str(n**dim) + '.csv')
    
        mag_whole_wolff.to_pickle('data/mag_whole_wolff' + str(n**dim) + '.csv')    
        two_point_wolff.to_pickle('data/two_point_wolff' + str(n**dim) + '.csv')
        chi_w.to_pickle('data/chi_w' + str(n**dim) + '.csv')
        Hc_w.to_pickle('data/Hc_w' + str(n**dim) + '.csv')
        
