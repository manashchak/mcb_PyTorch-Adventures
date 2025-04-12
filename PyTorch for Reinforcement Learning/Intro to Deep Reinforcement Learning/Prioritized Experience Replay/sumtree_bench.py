import numpy as np
import timeit
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def sample_dist(probabilities):
    cumulative_probs = np.cumsum(probabilities) 
    sample_val = random.uniform(0, 1.0) 

    for idx, p in enumerate(cumulative_probs):
        if sample_val < p:
            return idx
            
# class SumTree:

#     def __init__(self, capacity):

#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)
#         self.data = np.zeros(capacity, dtype=object) 

#         ### Write Pointer Is an Integer that indicated the next available slot ###
#         ### It ranges from 0 to capacity-1, and once we hit capacity, ###
#         ### we wrap back around to 0! ###
#         self.write_ptr = 0

#         ### Just a tracker to know how many samples we have, Once we hit capacity then ###
#         ### this wont really matter anymore ###
#         self.n_entries = 0

#     def total(self):
#         return self.tree[0]

#     def _propagate(self, tree_idx, change):
#         parent_idx = (tree_idx - 1)//2
#         self.tree[parent_idx] += change

#         if parent_idx != 0:
#             self._propagate(parent_idx, change)

#     def update(self, tree_idx, priority):
#         change = priority - self.tree[tree_idx]
#         self.tree[tree_idx] = priority
#         self._propagate(tree_idx, change)

#     def add(self, priority, data):
#         tree_idx = self.write_ptr + self.capacity - 1
#         self.data[self.write_ptr] = data
#         self.update(tree_idx, priority)

#         self.write_ptr += 1
#         if self.write_ptr >= self.capacity:
#             self.write_ptr = 0

#         if self.n_entries < self.capacity:
#             self.n_entries += 1

#     def _retrieve(self, node_idx, sample_value):
#         left_child_idx = 2 * node_idx + 1
#         right_child_idx = 2* node_idx + 2

#         if left_child_idx >= len(self.tree):
#             return node_idx

#         if sample_value <= self.tree[left_child_idx]:
#             return self._retrieve(left_child_idx, sample_value)
#         else:
#             return self._retrieve(right_child_idx, sample_value-self.tree[left_child_idx])

#     def get(self, sample_value):
#         leaf_tree_idx = self._retrieve(node_idx=0, sample_value=sample_value)
#         data_idx = leaf_tree_idx - self.capacity + 1
#         return leaf_tree_idx, self.tree[leaf_tree_idx], self.data[data_idx]
        
#     def __len__(self):
#         return self.n_entries
class SumTree:

    """
    Simple Array Implementation of SumTree
    """ 

    def __init__(self, N):

        ### Set Max Capacity N (max memories) ###
        self.N = N

        ### The tree will have 2 * N - 1 nodes ###
        self.tree = np.zeros(2 * N - 1)

        # ### Empty Array To Store Experiences ###
        # self.experiences = np.zeros(capacity, dtype=object) 

        ### Write Index Is an Integer that indicated the next available slot ###
        ### It ranges from 0 to N-1, and once we hit N, ###
        ### we wrap back around to 0! ###
        self.write_idx = 0

        ### Just a tracker to know how many samples we have, Once we hit N then ###
        ### this wont really matter anymore, as our entries will just be N ###
        self.n_entries = 0

    def total(self):
        """
        The total cumulative sum is stored in the root node
        """
        return self.tree[0]

    def _propagate(self, idx, delta):
        """
        If we added some Delta to our leaf node, we need to 
        propagate that Delta up the chain 
        """
        
        ### Use Our Logic to get from our current index to its parent ###
        parent_idx = (idx - 1)//2

        ### Add the Delta to our node ###
        self.tree[parent_idx] += delta

        ### If we arent at root node yet continue going up the chain ###
        if parent_idx != 0:
            self._propagate(parent_idx, delta)

    def update(self, idx, priority):

        """
        Method to compute our Delta between the new priority and existing priority
        and then propagate that change up the tree 
        """

        ### Compute the Change ###
        delta = priority - self.tree[idx]

        ### Set the New Value (basically adding our delta to the old value) ###
        self.tree[idx] = priority

        ### Propagate the delta up the tree ###
        self._propagate(idx, delta)

    def add(self, priority):

        """
        Method to add a new value to our tree at whatever the 
        write_ptr index is
        """

        ### Get the Tree Index. Our Write Index only goes from 0 -> N-1 ###
        ### But we can only write to our leaf nodes, and our leaf nodes are in ###
        ### indexes N-1 -> 2N-2. So just a simple shift is needed to get our indexes right:

        ### 0 + (N-1) -> (N-1)
        ### (N-1) + (N-1) -> (2N-2)
        
        tree_idx = self.write_idx + (self.N - 1)

        ### Update the rest of the tree accordingly ###
        self.update(tree_idx, priority)

        ### Update Write Index and Wrap ###
        self.write_idx += 1
        if self.write_idx >= self.N:
            self.write_idx = 0

        ### Update Number of Entries in our Tree So Far ###
        if self.n_entries < self.N:
            self.n_entries += 1

    def _retrieve(self, idx, sample_value):

        """
        Recursively goes down tree to find which region our 
        sample_value is inside
        """

        ### Index for Right and Left Child Nodes ###
        left_child_idx = 2 * idx + 1
        right_child_idx = 2* idx + 2

        ### If the Child Node Index is outside of our tree, we are on a leaf node ###
        if left_child_idx >= len(self.tree):
            return idx

        ### Logic for Left vs Right Decision As Explained Earlier ###
        if sample_value <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, sample_value)
        else:
            return self._retrieve(right_child_idx, sample_value-self.tree[left_child_idx])

    def get(self, sample_value):

        ### Get Our Leaf Index (N-1->2N-2) for whichever was selected based on the sample value ###
        leaf_tree_idx = self._retrieve(idx=0, sample_value=sample_value)

        ### Convert our Leaf Index Back to the original Index ###
        ### Our Leaf Index is (N-1->2N-2) but our data index just goes (0->N-1)
        ### Therefore adjust our index (reverse of what we had done earlier) 

        ### (N-1) - (N-1) -> 0
        ### (2N-2) - (N-1) -> N-1
        data_idx = leaf_tree_idx - (self.N - 1)

        return leaf_tree_idx, self.tree[leaf_tree_idx], data_idx
        
    def __len__(self):
        return self.n_entries
        
        
if __name__ == "__main__":

    buffer_sizes = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000] 
    num_samples_to_time = 1000 

    sumtree_times = []
    sample_dist_times = []
    
    def sumtree_sample(tree_obj):
        total_p = tree_obj.total()
        sample_value = random.uniform(0, total_p) 
        tree_obj.get(sample_value) # Perform the core operation
        
    for N in buffer_sizes:
        print(f"Benchmarking for N: {N}")
    
        priorities = np.random.uniform(0.01, 1.0, size=N).astype(np.float64)
        probabilities = priorities / priorities.sum() 
        
        st = SumTree(N)
        for i in range(N):
            st.add(priorities[i]) 
            
        st_time = timeit.timeit(lambda: sumtree_sample(st), number=num_samples_to_time)
        avg_time_st = st_time / num_samples_to_time
        sumtree_times.append(avg_time_st)
        print(f"SumTree Avg Time: {avg_time_st:.10f} sec/sample")
    
        sd_time = timeit.timeit(lambda p=probabilities: sample_dist(p), number=num_samples_to_time)
        avg_time_sd = sd_time / num_samples_to_time
        sample_dist_times.append(avg_time_sd)
        print(f"sample_dist Avg Time: {avg_time_sd:.10f} sec/sample")
        print("-" * 50)

    
    def smooth(x, y):
        x, y = np.array(x), np.array(y)
        x_smooth = np.linspace(x.min(), x.max(), 200) 
        spl = make_interp_spline(x, y, k=2) 
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth

    smooth_buffer_sizes, smooth_sumtree_times = smooth(buffer_sizes, sumtree_times)
    smooth_buffer_sizes, smooth_sample_dist_times = smooth(buffer_sizes, sample_dist_times)

    plt.plot(smooth_buffer_sizes, smooth_sumtree_times, label="SumTree")
    plt.plot(smooth_buffer_sizes, smooth_sample_dist_times, label="Default")
    plt.legend()
    plt.title("Log Time Comparison")
    plt.yscale('log')
    plt.xlabel("N")
    plt.ylabel("Log Sec/Sample")
    plt.tight_layout()
    plt.savefig("bench.png", dpi=200)
    