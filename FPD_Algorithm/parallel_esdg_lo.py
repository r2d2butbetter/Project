import cupy as cp
import numpy as np
import logging
import time
from collections import deque
from ESD_Graph.structures.esd_graph import ESD_graph

# --- Algorithm 3 Kernel: Level Order Traversal ---
# Implements the logic from Algorithm 3 in the paper[cite: 281].
# - Processes all nodes in a specific level range in parallel.
# - Propagates startTime (max departure) to neighbors using atomicMax.
LEVEL_ORDER_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void level_order_kernel(
    const int level_start_idx,      // Start index of this level in sorted arrays
    const int level_end_idx,        // End index (exclusive)
    const int* nodes_v,             // Dest vertex (right(x))
    const int* nodes_a,             // Arrival time (arr(x))
    const int* adj_indptr,          // CSR pointers
    const int* adj_indices,         // CSR indices
    int* start_times,               // Propagated Departure Times (startTime[x])
    int* status,                    // Active Status (0 or 1)
    unsigned long long* packed_results // High 32: time, Low 32: predecessor
) {
    // 1. Calculate global thread ID mapped to the reordered node index
    int idx = level_start_idx + blockDim.x * blockIdx.x + threadIdx.x;
    
    // Bounds check
    if (idx >= level_end_idx) return;
    
    // 2. Check if node is active
    if (status[idx] == 1) {
        
        int my_start_time = start_times[idx];
        int my_arrival = nodes_a[idx];
        int dest_vertex = nodes_v[idx];
        
        // 3. Compute Journey Time
        // journey = arr(x) - startTime[x]
        int journey = my_arrival - my_start_time;
        
        // 4. Update Global Fastest Time (Atomic Packed)
        unsigned long long new_val = ((unsigned long long)journey << 32) | idx;
        atomicMin(&packed_results[dest_vertex], new_val);
        
        // 5. Propagate to Neighbors
        int start_edge = adj_indptr[idx];
        int end_edge = adj_indptr[idx + 1];
        
        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor_idx = adj_indices[i];
            
            // atomicMax(startTime[y], startTime[x])
            atomicMax(&start_times[neighbor_idx], my_start_time);
            
            // status[y] = true
            // "Every thread performs the same action" so no atomic needed for status [cite: 379]
            status[neighbor_idx] = 1;
        }
    }
}
''', 'level_order_kernel')

class ParallelESDG_LO:
    """
    GPU-accelerated Solver implementing Algorithm 3 (Level Order Traversal).
    Optimized for small-to-medium datasets by removing phase overhead.
    """

    def __init__(self, esd_graph: ESD_graph):
        self.original_graph = esd_graph
        self.num_nodes = len(esd_graph.nodes)
        
        # Determine max vertex ID
        self.max_vertex_id = 0
        if self.num_nodes > 0:
            max_u = max(int(n.u) for n in esd_graph.nodes.values())
            max_v = max(int(n.v) for n in esd_graph.nodes.values())
            self.max_vertex_id = max(max_u, max_v)

        logging.info("Initializing Algorithm 3 (Level Order) Solver...")
        self._prepare_gpu_data()

    def _compute_levels(self):
        """
        Computes the topological level for every node in the ESDG.
        Level(x) = 1 + max(Level(parents))[cite: 183].
        """
        in_degree = {i: 0 for i in self.original_graph.nodes}
        for u in self.original_graph.nodes:
            for v in self.original_graph.adj.get(u, []):
                in_degree[v] = in_degree.get(v, 0) + 1
        
        queue = deque([u for u in self.original_graph.nodes if in_degree[u] == 0])
        levels = {u: 1 for u in self.original_graph.nodes} # Default level 1
        
        visited_count = 0
        max_level = 0
        
        # Topological Sort
        while queue:
            u = queue.popleft()
            current_lvl = levels[u]
            max_level = max(max_level, current_lvl)
            visited_count += 1
            
            for v in self.original_graph.adj.get(u, []):
                # Level propagation: Child level is at least Parent + 1
                if levels[v] < current_lvl + 1:
                    levels[v] = current_lvl + 1
                
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        return levels, max_level

    def _prepare_gpu_data(self):
        """
        Relabels nodes based on levels to ensure Memory Coalescing.
        Sorts nodes by Level, then builds CSR/SoA arrays.
        """
        t0 = time.perf_counter()
        
        # 1. Compute Levels
        node_levels, self.max_level = self._compute_levels()
        
        # 2. Sort nodes by Level (Primary) and ID (Secondary) for stability
        # "Nodes in the graph are relabeled such that all nodes in any level appear consecutively" 
        sorted_node_ids = sorted(self.original_graph.nodes.keys(), 
                                 key=lambda k: (node_levels[k], k))
        
        # Create Mapping: Original ID -> New Sorted GPU Index
        self.id_map = {original: i for i, original in enumerate(sorted_node_ids)}
        
        # 3. Build Level Offsets [cite: 381]
        # We need to know where Level 1 starts, Level 2 starts, etc.
        self.level_offsets = []
        current_level = 0
        for i, original_id in enumerate(sorted_node_ids):
            lvl = node_levels[original_id]
            while current_level < lvl:
                self.level_offsets.append(i) # Mark start of new level
                current_level += 1
        self.level_offsets.append(self.num_nodes) # End marker
        
        # 4. Build Arrays (SoA)
        nodes_v = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_a = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_u_cpu = np.zeros(self.num_nodes, dtype=np.int32)
        nodes_t_cpu = np.zeros(self.num_nodes, dtype=np.int32)
        
        for i, original_id in enumerate(sorted_node_ids):
            node = self.original_graph.nodes[original_id]
            nodes_v[i] = int(node.v)
            nodes_a[i] = int(node.a)
            nodes_u_cpu[i] = int(node.u)
            nodes_t_cpu[i] = int(node.t)
            
        # 5. Build CSR Adjacency (remapped to new indices)
        adj_indices_list = []
        adj_indptr = [0]
        
        for original_id in sorted_node_ids:
            neighbors = self.original_graph.adj.get(original_id, [])
            mapped_neighbors = [self.id_map[n] for n in neighbors if n in self.id_map]
            adj_indices_list.extend(mapped_neighbors)
            adj_indptr.append(len(adj_indices_list))
            
        # 6. Transfer to GPU
        self.d_nodes_v = cp.asarray(nodes_v)
        self.d_nodes_a = cp.asarray(nodes_a)
        self.d_adj_indices = cp.asarray(adj_indices_list, dtype=np.int32)
        self.d_adj_indptr = cp.asarray(adj_indptr, dtype=np.int32)
        
        # Keep source logic on CPU
        self.h_nodes_u = nodes_u_cpu
        self.h_nodes_t = nodes_t_cpu
        
        t1 = time.perf_counter()
        logging.info(f"Level-Order Data Prep (L={self.max_level}) complete in {t1-t0:.4f}s")

    def find_fastest_paths(self, source_vertex_s: str, reconstruct_paths: bool = False):
        """
        Executes Algorithm 3.
        Single pass through levels 1 to L.
        """
        source_vertex_int = int(source_vertex_s)
        t_start = time.perf_counter()
        
        # --- 1. Initialization ---
        # startTime[x] = -1 (infinity equivalent for max)
        d_start_times = cp.full(self.num_nodes, -1, dtype=cp.int32)
        # status[x] = 0 (false)
        d_status = cp.zeros(self.num_nodes, dtype=cp.int32)
        
        # Result arrays
        INF_TIME = 2147483647
        init_val = (INF_TIME << 32) | 0xFFFFFFFF
        d_packed_results = cp.full(self.max_vertex_id + 1, init_val, dtype=cp.uint64)
        
        # Set source journey to 0
        source_init = (0 << 32) | 0xFFFFFFFF
        d_packed_results[source_vertex_int] = source_init

        # --- 2. Activate Source Nodes ---
        # Find all ESDG nodes u where left(u) == s
        source_indices = np.where(self.h_nodes_u == source_vertex_int)[0]
        
        if len(source_indices) == 0:
            return {}, {}
            
        # Initialize sources on GPU
        # startTime[x] = dep(x); status[x] = true;
        d_start_times[source_indices] = cp.asarray(self.h_nodes_t[source_indices])
        d_status[source_indices] = 1

        # --- 3. Level Order Loop ---
        threads = 256
        
        # Iterate from Level 1 to L
        # Note: self.level_offsets indices correspond to levels 1..L
        for l in range(1, len(self.level_offsets)):
            start_idx = self.level_offsets[l-1]
            end_idx = self.level_offsets[l]
            
            num_nodes_in_level = end_idx - start_idx
            
            if num_nodes_in_level <= 0:
                continue
                
            blocks = (num_nodes_in_level + threads - 1) // threads
            
            LEVEL_ORDER_KERNEL(
                (blocks,), (threads,),
                (
                    int(start_idx),
                    int(end_idx),
                    self.d_nodes_v,
                    self.d_nodes_a,
                    self.d_adj_indptr,
                    self.d_adj_indices,
                    d_start_times,
                    d_status,
                    d_packed_results
                )
            )

        # --- 4. Unpack Results ---
        h_packed = cp.asnumpy(d_packed_results)
        times = h_packed >> 32
        
        result_times = {}
        for v in range(len(times)):
            t = int(times[v])
            if t < INF_TIME:
                result_times[str(v)] = t

        fastest_paths = {}
        # (Path reconstruction logic remains similar if needed, omitted for brevity/speed focus)

        t_end = time.perf_counter()
        logging.info(f"Algorithm 3 (LO) Complete: {t_end - t_start:.4f}s")
        
        return result_times, fastest_paths