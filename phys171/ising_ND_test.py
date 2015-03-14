## Simulate the ising Model in any dimension using the Metropolis Algorithm

from __future__ import division
import numpy as np
import pandas as pd
from scipy import sparse
import lmfit
import ipdb

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
    
    NearNeigh, Near_spins = calc_neighbors(site_i,coord,spins,n,dim,neighbor)
    # A component in the neighbor term with magnetic field term
    energy_change = -2*k*(np.sum(-spins[site_i]*Near_spins)) - h*(-spins[site_i]-spins[site_i])
    
    return energy_change

# Given the spins and coordinates, calculate the spin-spin correlation by 
# choosing s0 to be the center, and running through the lattice axes.
def calc_spin_spin(coord,spins,n,bound,dim,
                   neighbor):
    # Result
    spin_spin_whole = np.zeros(n-bound)
    spin_spin_whole[0] = 1 # Every spin is correlated with itself
    # Samples
    samples = np.zeros(n-bound)
    samples[0] = 1
    
    # Access each diagonal element spin
    diag_spins_sites = np.zeros(n,dtype=np.int16)
    diag_coord = np.zeros((n,dim),dtype=np.int16)
    diag_spin_val = np.zeros(n,dtype=np.int8)
    for i in xrange(0,n):
        diag_coord_i = i*np.ones(dim)
        diag_coord[i] = diag_coord_i
        spin_site_diag = np.where(np.all(diag_coord_i==coord,axis=1))[0][0]              
        diag_spins_sites[i] = np.int(spin_site_diag)
        diag_spin_val[i] = spins[spin_site_diag]
    
    # Now we have the diagonal spins and coordinates
    # For each diagonal spin
    r_spin_count = 0
    r_spin_sum = 0 
    for i,spin_site in np.ndenumerate(diag_spins_sites):
        i = i[0]
        # For each distance r from the spin
        samples_ith_spin = np.zeros(n-bound)
        for r in xrange(1,n-bound):
            # Locate the neighbors a distance r away
            NearNeigh, Near_spins = calc_neighbors(spin_site,coord,spins,n,dim,
                                                   r)
            # Only consider the spins that aren't influenced by
            # periodic boundary conditions
            # If the ith spin will access r greater or smaller than the boundary
                                                   
            if ((i + r > n-1) and (i - r < 0)): # Outside the larger & smaller boundary
                # Don't contribute from this value
                continue
            
            elif (i - r < 0): # Outside the smaller boundary
                # Access the (plus) values only (which have odd index values)
                indices = np.array(range(0,2*dim))
                indices = indices[indices%2==1]
                NearNeigh = NearNeigh[indices]
                Near_spins = Near_spins[indices]
            elif (i + r > n-1): # Outside the larger boundary only
                # Access the (minus) values only (which have even index values)
                indices = np.array(range(0,2*dim))
                indices = indices[indices%2==0]
                NearNeigh = NearNeigh[indices]
                Near_spins = Near_spins[indices]
            
            # Store the spins as "r"
            r_spin_sum = r_spin_sum + np.sum(Near_spins)
            r_spin_count = r_spin_count + len(Near_spins)
            # Calculate spin-spin contribution
            spin_spin_ith = spins[spin_site]*np.sum(Near_spins)
            samples_ith_spin[r] = samples_ith_spin[r] + len(Near_spins)
            # Add to the spin_spin array for each r
            spin_spin_whole[r] = spin_spin_whole[r] + spin_spin_ith
            
        # Update total sample array
        samples = samples + samples_ith_spin
            
    return spin_spin_whole/samples - (np.mean(diag_spin_val))*(r_spin_sum/r_spin_count)

# Run the metropolis algorithm 
def run_Metropolis(T,kb,J,H,mu,
                   coord,spins,n,bound,dim,
                   neighbor,
                   MC_trials,Equib_trials,interval,
                   seed_int,divisor):
    
    print "Beginning Metropolis..."
    # Set the seed
    np.random.seed(seed=seed_int)
    # For magnetization and spin_spin data
    mag_whole = {}
    spin_spin_temp = {}
    energy_whole = {}
    # Magnetic Susceptibility
    chi = {}
    # Heat Capacity
    Hc = {}
    # For the temperatures of interest
    for t in T:
        print "Temperature =",t
        mag = np.zeros(MC_trials/interval)
        energy = np.zeros(MC_trials/interval)
        B = 1/(kb*t)
        k = J*B
        h = mu*H*B
        count = 0
        spin_spin_whole = np.zeros(n-bound)
        # Calculate MC trials and take data for each trial
        energy_config, tot_spin_per_site = calc_energy_config(k,h,
                                                              coord,spins,n,dim,
                                                              neighbor)
        for i in xrange(0,MC_trials + Equib_trials):
            if (np.mod(i,(MC_trials)/divisor)==0): print i
            # Choose a random site
            site_i = np.random.randint(0,spin_num)
            # Calculate the energy change
            energy_change = calc_energy_change(site_i,k,h,
                                               coord,spins,n,dim,
                                               neighbor)
            # Flip spin if energy change is negative, else check boltzmann weight
            if energy_change <= 0:
                spins[site_i] = -spins[site_i]
                energy_config = energy_config + energy_change 
            elif (np.exp(-B*energy_change) > np.random.random()):
                spins[site_i] = -spins[site_i]
                energy_config = energy_config + energy_change
            # If equilibrium has been reached at t
            if (i >= Equib_trials and np.mod(i,interval)==0):
                # Obtain the magnetization per site
                mag[count] = (np.sum(spins)/spin_num)
                
                energy[count] = energy_config

                # Obtain the spin-spin correlation
                spin_spin = calc_spin_spin(coord,spins,n,bound,dim,
                                           neighbor)
                count = count + 1
                spin_spin_whole = (spin_spin_whole + spin_spin)
                
        spin_spin_whole = spin_spin_whole/count
        spin_spin_temp[str(t)] = spin_spin_whole
        # Obtain heat capacity
        energy_whole[str(t)] = np.mean(energy)
        Hc[str(t)] = (1/(t*n**dim))*np.var(energy)
        # Obtain chi 
        chi[str(t)] = (1/t)*np.var(mag)
        # Obtain the mean magnetization per site
        mag_whole[str(t)] = (np.sum(mag))/count
    # Convert to Pandas Object                   
    mag_whole = pd.Series(mag_whole)
    spin_spin_temp = pd.DataFrame(spin_spin_temp)
    chi = pd.Series(chi)
    Hc = pd.Series(Hc)
    energy_whole = pd.Series(energy_whole)
    print "Ending Metropolis."    
    return mag_whole, spin_spin_temp, chi, Hc, energy_whole

# Run the Wolff Cluster algorithm
def run_Wolff(T,kb,J,H,mu,
              coord,spins,n,bound,dim,
              neighbor,
              MC_trials,Equib_trials,interval,
              seed_int,divisor):
    print "Beginning Wolff..."
    # Set the seed
    np.random.seed(seed=seed_int)
    # For magnetization and spin_spin data
    mag_whole = {}
    spin_spin_temp = {}
    energy_whole = {}
    # Magnetic Susceptibility
    chi = {}
    # Heat Capacity
    Hc = {}
    # For temperatures of interest
    for t in T:
        print "Temperature =",t
        mag = np.zeros(MC_trials/interval)
        energy = np.zeros(MC_trials/interval)
        B = 1/(kb*t)
        k = J*B
        h = mu*H*B
        count = 0
        spin_spin_whole = np.zeros(n-bound)
        # Keep track of the cluster and the stack
        stack = []
        interaction_matrix = sparse.lil_matrix((n**dim,n**dim))
        for i in xrange(0,MC_trials + Equib_trials):
            if (np.mod(i,(MC_trials)/divisor)==0): print i
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
                if ((val == spins[site_i]) and ((interaction_matrix[val_site,site_i] == 0) and
                (interaction_matrix[site_i,val_site] == 0))):
                    # Update interaction matrix
                    interaction_matrix[val_site,site_i] = 1
                    interaction_matrix[site_i,val_site] = 1
                    # TODO check if less than or greater than
                    # Add to cluster and stack then invert spin
                    if (np.random.random() > p):
                        #cluster.append(val_site)
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
                    if ((val == spins[site_i]) and ((interaction_matrix[val_site,site_i] == 0) and
                    (interaction_matrix[site_i,val_site] == 0))):
                        # Update interaction matrix
                        interaction_matrix[val_site,site_i] = 1
                        interaction_matrix[site_i,val_site] = 1
                        # TODO check if less than or greater than
                        # Add to cluster and stack then invert spin
                        if (np.random.random() > p):
                            #cluster.append(val_site)
                            stack.append(val_site)
                            spins[val_site] = -spins[val_site]
                        # If equilibrium has been reached at t
            if (i >= Equib_trials and np.mod(i,interval)==0):
                # Now the stack is empty
                # Calculate observable on spins
                mag[count] = (np.sum(spins)/spin_num)
                energy_config, tot_spin_per_site = calc_energy_config(k,h,
                                                                      coord,spins,n,dim,
                                                                      neighbor)
                energy[count] = energy_config

                # Obtain the spin-spin correlation
                spin_spin = calc_spin_spin(coord,spins,n,bound,dim,
                                           neighbor)
                count = count + 1
                spin_spin_whole = (spin_spin_whole + spin_spin)
                
        spin_spin_whole = spin_spin_whole/count
        spin_spin_temp[str(t)] = spin_spin_whole
        # Done with MC trials now
        # Obtain heat capacity
        energy_whole[str(t)] = np.mean(energy)
        Hc[str(t)] = (1/(t*n**dim))*np.var(energy)        
        # Obtain chi 
        chi[str(t)] = (1/t)*np.var(mag)
        # Obtain the mean magnetization per site
        mag_whole[str(t)] = (np.sum(mag)/len(mag))
    # Convert to Pandas Object                                             
    mag_whole = pd.Series(mag_whole)
    spin_spin_temp = pd.DataFrame(spin_spin_temp)
    chi = pd.Series(chi)
    Hc = pd.Series(Hc)
    energy_whole = pd.Series(energy_whole)
    print "Ending Wolff."    
    return mag_whole, spin_spin_temp, chi, Hc, energy_whole

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
    n = 28
    bound = 2
    # Dimension
    dim = 3
    # Number of spins
    spin_num = n**dim
    # Nearest neighbor coupling
    neighbor = 1
    # Integer for seed
    seed_int_one = 0
    seed_int_two = 0
    seed_int_three = 0
    # Number of MC trials
    MC_trials = 1500
    # Number of Equilibrium Trials
    Equib_trials = 100
    # Effective values
    k = J*B
    h = mu*H*B    
    # Intervals at which to sample observable in MC iteration
    interval = 25
    divisor = 100
    assert np.mod(MC_trials,interval)==0, "Use proper interval"

    # ---------------------- Spins and Coordinates -------------------- #
    # Create Spins with Coordinate Array
    coord = create_coord(n,dim)
    # Assign spins to array
    spins = assign_spins(spin_num,all_ones=True,seed=seed_int_one)                                                  
    
    # ---------------------- Metropolis ------------------------------- #
    Tinit = 0.01    
    T_end = 4
    fir = np.linspace(Tinit,1,3)
    sec = np.linspace(1.1,1.8,6)
    thi = np.linspace(1.9,2.5,25)
    four = np.linspace(2.6,T_end,6)
    T = np.concatenate([fir,sec,thi,four])
   

    met = False
    w = True

    if met == True:
        mag_whole_met, two_point_met, chi_met, Hc_met, energy_met = run_Metropolis(T,kb,J,H,mu,
                                                                                   coord,spins,n,bound,dim,
                                                                                   neighbor,
                                                                                   MC_trials,Equib_trials,interval,
                                                                                   seed_int_two,divisor)
        
        save = True
        if save == True:
            
            info = 'size_' + str(n) + '_dim_' + str(dim) + '_MC_' + str(MC_trials) +'_inter_' + str(interval) +  '_T' +str(T_end-Tinit) + '.csv'
            
            mag_whole_met.to_csv('data/mag_whole_met_' + info)    
            two_point_met.to_csv('data/two_point_met_' + info)
            chi_met.to_csv('data/chi_met_' + info)
            Hc_met.to_csv('data/Hc_met_' + info)
            energy_met.to_csv('data/energy_met_' + info)    
    
    
    # ---------------------- Spins and Coordinates -------------------- #
    # Create new set of all up spins
    # Assign spins to array
    spins = assign_spins(spin_num,all_ones=True,seed=seed_int_one)    
    
    # ---------------------- Wolff Cluster ---------------------------- #

    if w == True:
        mag_whole_wolff, two_point_wolff, chi_w, Hc_w, energy_w = run_Wolff(T,kb,J,H,mu,
                                                                            coord,spins,n,bound,dim,
                                                                            neighbor,
                                                                            MC_trials,Equib_trials,interval,
                                                                            seed_int_two,divisor)                                                              
            
        if save == True:        
            mag_whole_wolff.to_csv('data/mag_whole_wolff_' + info)    
            two_point_wolff.to_csv('data/two_point_wolff_' + info)
            chi_w.to_csv('data/chi_w_' + info)
            Hc_w.to_csv('data/Hc_w_' + info)
            energy_w.to_csv('data/energy_w_' + info)
        
    
    plot = False
    if plot == True:    
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    
        gs = gridspec.GridSpec(20,2)
        fig = plt.figure(figsize=(15,11))
        fs = 15
        plt.suptitle('Metropolis and Wolff Monte Carlo Simulation on a\n' +
                     str(dim) + 'D Lattice of Size ' + str(n) +' With ' +
                     str(MC_trials) +' Monte Carlo Trials',fontsize=fs)
        
        ax1 = fig.add_subplot(gs[0:5,0])     
        plt.title('Magnetization',fontsize=fs)
        plt.plot(T,mag_whole_met,'--o',T,mag_whole_wolff,'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\frac{<M>}{N}$',fontsize=fs)
        plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})
    
        Temp = T[44]
        
        ax2 = fig.add_subplot(gs[7:12,0])
        model = 1/((np.linspace(1,n-bound,n-bound))**(1/4))
        plt.title('Two-Point Correlation',fontsize=fs)
        plt.plot(np.array(xrange(0,n-bound)),np.abs(two_point_met[str(Temp)])[0:n],'--o',
                 np.array(xrange(0,n-bound)),np.abs(two_point_wolff[str(Temp)][0:n]),'--o',
                 np.array(xrange(0,n-bound)),model,'-o')
        plt.xlabel(r'$|r_i - r_j|$',fontsize=fs); plt.ylabel(r'$<s_os_r>$',fontsize=fs)
        plt.legend(['Metropolis','Wolff','True'],prop={'size':fs-5})
    
        ax3 = fig.add_subplot(gs[0:5,1])
        plt.title('Specific Heat',fontsize=fs)
        #plt.plot(T,Hc_met,'--o')
        plt.plot(T,Hc_met,'--o',T,Hc_w,'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$C$',fontsize=fs)
        plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})
    
        ax4 = fig.add_subplot(gs[7:12,1])
        plt.title('Magnetic Susceptibility',fontsize=fs)
        plt.plot(T,chi_met,'--o',T,chi_w,'--o')
        locs,labels = plt.yticks()
        plt.yticks(locs, map(lambda x: "%.1f" % x,locs*1e6))
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel(r'$\chi\/(1E-6)$',fontsize=fs)
        plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})
        
        ax5 = fig.add_subplot(gs[14:19,0:])
        plt.title('Energy',fontsize=fs)
        plt.plot(T,energy_met,'--o',T,energy_w,'--o')
        plt.xlabel('Temperature',fontsize=fs); plt.ylabel('Energy',fontsize=fs)
        plt.legend(['Metropolis','Wolff'],prop={'size':fs-5})
            
        plt.show()

