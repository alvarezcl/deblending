## 1-D Implementation of D. Lang's algorithm in order to verify conservation
## of flux.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(argv):
    # Draw from two gaussians
    N = 100
    x1 = np.random.normal(loc=-2,scale=1,size=N)
    x1 = np.sort(x1)
    x2 = np.random.normal(loc=3,scale=1.5,size=N)
    x2 = np.sort(x2)
    sum = np.array(x1.tolist() + x2.tolist())
    sum = np.sort(sum)
    # Create a histogram of the two
    bin_size = 0.1
    min_value = -8; max_value = 8; bin_num = (max_value-min_value)/bin_size
    bins = np.linspace(min_value,max_value,bin_num)
    fig = plt.figure(figsize=(15,11))
    n_1, bin_edges_1, patches_1 = plt.hist(x1,bins=bins,histtype=u'step')
    n_2, bin_edges_1, patches_2 = plt.hist(x2,bins=bins,histtype=u'step')
    n_3, bin_edges_1, patches_3 = plt.hist(sum,bins=bins,histtype=u'step')
    plt.show()
    
def pivot(arr,loc):
    floored_arr = np.floor(arr)
    index = np.mean(floored_arr==loc)
    index = int(np.floor(index))
    rev_index = np.length(arr) - index
    rev_arr = arr[::-1]
    rev_part_arr = rev_arr[rev_index:]
    if rev_index > index:
        new = []
        num = rev_index - index
        zero_arr = np.zeros(num)
        for i in xrange(index,rev_index):
            new.append(arr[i])
        new = new + rev_part_arr.tolist() + zero_arr.tolist() 
        new = np.array(new)    
    else:
        new = []
        num = index - rev_index
        zero_arr = np.zeros(num)
        for i in xrange(rev_index,index):
            new.append(arr[i])
        new = new + rev_part_arr.tolist() + zero_arr.tolist()           
        new = np.array(new)    
    return new

if __name__ == '__main__':
    main(sys.argv)