import cupy as cp
import numpy as np
import logging
import time
from ESD_Graph.structures.esd_graph import ESD_graph

# --- Optimized CUDA Kernel ---
# Implements Algorithm 1 (MBFS) with optimizations.
# Key changes:
# 1. Atomic Packed Update: Packs journey_time (high 32) and predecessor (low 32) 
#    into a single 64-bit int to ensure atomic consistency.
# 2. Sparse Queue: Uses atomicAdd to build the next frontier sparsely, avoiding 
#    expensive dense scans.
BFS_KERNEL_OPTIMIZED = cp.RawKernel(r'''
extern "C" __global__
void bfs_kernel(
    const int frontier_size,        // Size of current frontier
    const int* frontier_nodes,      // Sparse list of current nodes
    const int* nodes_v,             // Dest vertex (right(x))
    const int* nodes_a,             // Arrival time (arr(x))
    const int source_departure,     // Source departure (dep(z))
    const int* adj_indptr,          // CSR pointers
    const int* adj_indices,         // CSR indices
    unsigned long long* packed_results, // High 32: time, Low 32: predecessor
    int* status,                    // Global visited status
    int* next_frontier,             // Next frontier sparse array
    int* next_frontier_count        // Counter for next frontier size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx >= frontier_size) return;
    
    int node_idx = frontier_nodes[idx];
    int dest_vertex = nodes_v[node_idx];
    int arrival_time = nodes_a[node_idx];
    
    // 1. Calculate Journey Time: arr(x) - dep(z)
    int journey_time = arrival_time - source_departure;
    
    // 2. Atomic Packed Update (Fixes Race Condition)
    // Pack time and predecessor into one 64-bit int to update atomically.
    // Time must be unsigned for comparison logic to work with ULL.
    // Assuming positive times.
    unsigned long long new_val = ((unsigned long long)journey_time << 32) | node_idx;
    atomicMin(&packed_results[dest_vertex], new_val);
    
    // 3. Explore Neighbors 
    int start_edge = adj_indptr[node_idx];
    int end_edge = adj_indptr[node_idx + 1];
    
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = adj_indices[i];
        
        // 4. Pruning via atomicCAS
        // "Ignore already visited nodes since they cannot offer a shorter journey time"
        if (atomicCAS(&status[neighbor_idx], 0, 1) == 0) {
            
            // 5. Sparse Queue Insertion 
            // "Reserve necessary space... using atomicAdd"
            int queue_pos = atomicAdd(next_frontier_count, 1);
            next_frontier[queue_pos] = neighbor_idx;
        }
    }
}
''', 'bfs_kernel')

class ParallelESDG_FPD:
    """
    GPU-accelerated FPD Solver implementing Algorithm 1 (MBFS).
    Includes optimizations for Sparse Queues and Atomic 64-bit updates.
    """

    def __init__(self, esd_graph: ESD_graph):
        self.original_graph = esd_graph
        self.num_nodes = len(esd_graph.nodes)
        
        # Determine max vertex ID for sizing arrays
        self.max_vertex_id = 0
        if self.num_nodes > 0:
            max_u = max(int(n.u) for n in esd_graph.nodes.values())
            max_v = max(int(n.v) for n in esd_graph.nodes.values())
            self.max_vertex_id = max(max_u, max_v)

        logging.info("Initializing Parallel FPD Solver...")
        self._prepare_gpu_data()

    def _prepare_gpu_data(self):
        """
        Prepares SoA (Structure of Arrays) and CSR format.
        Note: For full compliance, nodes should be relabeled by Level-Order
        here to enable Memory Coalescing.
        """
        t0 = time.perf_counter()
        
        # 1. Node Mapping (Arbitrary order for now)
        sorted_node_ids = list(self.original_graph.nodes.keys())
        self.id_map = {original: i for i, original in enumerate(sorted_node_ids)}
        
        # 2. Build Arrays
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
        
        # 3. CSR Adjacency
        adj_indices_list = []
        adj_indptr = [0]
        
        for original_id in sorted_node_ids:
            neighbors = self.original_graph.adj.get(original_id, [])
            mapped_neighbors = [self.id_map[n] for n in neighbors if n in self.id_map]
            adj_indices_list.extend(mapped_neighbors)
            adj_indptr.append(len(adj_indices_list))
            
        # 4. Transfer to GPU
        self.d_nodes_v = cp.asarray(nodes_v)
        self.d_nodes_a = cp.asarray(nodes_a)
        self.d_adj_indices = cp.asarray(adj_indices_list, dtype=np.int32)
        self.d_adj_indptr = cp.asarray(adj_indptr, dtype=np.int32)
        
        # Keep source logic on CPU
        self.h_nodes_u = nodes_u_cpu
        self.h_nodes_t = nodes_t_cpu
        
        t1 = time.perf_counter()
        logging.info(f"GPU Data Preparation complete in {t1-t0:.4f}s")

    def _reconstruct_paths_cpu(self, predecessors, result_times, source_vertex_int):
        """
        Reconstructs paths from the predecessor array returned by the GPU.
        Uses the mapping packed into the low 32 bits of the result.
        """
        fastest_paths = {}
        
        # Create reverse mapping: GPU Index -> Original Node Object
        idx_to_original_node = {}
        sorted_node_ids = list(self.original_graph.nodes.keys()) # Must match init order
        for i, original_id in enumerate(sorted_node_ids):
            idx_to_original_node[i] = self.original_graph.nodes[original_id]

        # Only reconstruct for reachable nodes
        sorted_dests = sorted(
            [(dest, time) for dest, time in result_times.items()],
            key=lambda x: x[1]
        )[:50] # Limit to top 50 for performance safety in logs
        
        for dest_str, _ in sorted_dests:
            dest_v = int(dest_str)
            if dest_v == source_vertex_int:
                continue
                
            path = []
            curr_v = dest_v
            
            # Max hops safety
            for _ in range(100):
                if curr_v >= len(predecessors): break
                
                pred_node_idx = predecessors[curr_v]
                
                # Check for invalid predecessor (0xFFFFFFFF from init)
                if pred_node_idx == 4294967295: 
                    break
                    
                if pred_node_idx in idx_to_original_node:
                    node_obj = idx_to_original_node[pred_node_idx]
                    path.append(node_obj)
                    curr_v = int(node_obj.u) # Move to the start of this edge
                else:
                    break
            
            if path:
                fastest_paths[dest_str] = path[::-1] # Reverse to get Source -> Dest
                
        return fastest_paths

    def find_fastest_paths(self, source_vertex_s: str, reconstruct_paths: bool = False):
        """
        Executes Algorithm 1: Multiple Breadth-First Search.
        Args:
            source_vertex_s: Source vertex ID as string.
            reconstruct_paths: If True, unpacks predecessors and builds path objects.
        """
        source_vertex_int = int(source_vertex_s)
        t_start = time.perf_counter()

        # --- 1. Initialization ---
        # Status array: 0 = False, 1 = True
        d_status = cp.zeros(self.num_nodes, dtype=cp.int32)
        
        # Packed Results: High 32 bits = Journey Time, Low 32 bits = Predecessor Index
        # Initialize with Max Int (infinity) in high bits
        INF_TIME = 2147483647
        init_val = (INF_TIME << 32) | 0xFFFFFFFF # Max time, invalid pred
        d_packed_results = cp.full(self.max_vertex_id + 1, init_val, dtype=cp.uint64)
        
        # Set source journey to 0
        source_init = (0 << 32) | 0xFFFFFFFF
        d_packed_results[source_vertex_int] = source_init

        # --- 2. Identify and Sort Source Nodes ---
        source_node_indices = np.where(self.h_nodes_u == source_vertex_int)[0]
        if len(source_node_indices) == 0:
            logging.warning(f"No source nodes found for vertex {source_vertex_int}")
            return {}, {}

        # "In non-increasing order of dep(z)"
        source_departures = self.h_nodes_t[source_node_indices]
        sorted_order = np.argsort(-source_departures)
        sorted_indices = source_node_indices[sorted_order]
        sorted_departures = source_departures[sorted_order]
        
        logging.info(f"Processing {len(sorted_indices)} phases for source {source_vertex_int}")

        # --- 3. Parallel Traversal ---
        # Pre-allocate two buffers for frontiers (ping-pong)
        # Max size is num_nodes
        d_frontier_buffers = [
            cp.zeros(self.num_nodes, dtype=cp.int32),
            cp.zeros(self.num_nodes, dtype=cp.int32)
        ]
        
        # Counter for next frontier size (device array of size 1)
        d_next_count = cp.zeros(1, dtype=cp.int32)

        for phase, source_idx in enumerate(sorted_indices):
            # Check status (Requires D->H sync, unavoidable in Algorithm 1 logic)
            if d_status[source_idx].item() != 0:
                continue

            # Init Phase
            source_departure = int(sorted_departures[phase])
            d_status[source_idx] = 1 # Mark visited
            
            # Setup initial frontier
            curr_buf_idx = 0
            d_frontier_buffers[curr_buf_idx][0] = source_idx
            current_frontier_size = 1
            
            # BFS Loop
            while current_frontier_size > 0:
                next_buf_idx = 1 - curr_buf_idx
                
                # Reset counter for next frontier 
                d_next_count.fill(0)
                
                # Calculate grid size
                threads = 256
                blocks = (current_frontier_size + threads - 1) // threads
                
                BFS_KERNEL_OPTIMIZED(
                    (blocks,), (threads,),
                    (
                        current_frontier_size,
                        d_frontier_buffers[curr_buf_idx], # Current Nodes
                        self.d_nodes_v,
                        self.d_nodes_a,
                        source_departure,
                        self.d_adj_indptr,
                        self.d_adj_indices,
                        d_packed_results,
                        d_status,
                        d_frontier_buffers[next_buf_idx], # Next Frontier (Sparse)
                        d_next_count # Atomic Counter
                    )
                )
                
                # Get size of next frontier (Device -> Host sync)
                current_frontier_size = int(d_next_count.item())
                curr_buf_idx = next_buf_idx

        # --- 4. Unpack Results ---
        h_packed = cp.asnumpy(d_packed_results)
        
        # Use bitwise ops to extract time and predecessor
        times = h_packed >> 32
        
        result_times = {}
        for v in range(len(times)):
            t = int(times[v])
            if t < INF_TIME:
                result_times[str(v)] = t

        fastest_paths = {}
        if reconstruct_paths:
            preds = h_packed & 0xFFFFFFFF
            fastest_paths = self._reconstruct_paths_cpu(preds, result_times, source_vertex_int)

        t_end = time.perf_counter()
        logging.info(f"GPU FPD Complete: {t_end - t_start:.4f}s")
        
        return result_times, fastest_paths