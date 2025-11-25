import collections
import time
import logging
from ESD_Graph.structures.esd_graph import ESD_graph

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialESDG_FPD:
    """
    Implements the serial Fastest Path Duration (FPD) algorithm using 
    a pre-computed Edge Scan Dependency (ESD) Graph.
    
    Based on the logic described in Section 2.3 of the paper:
    'Efficient Algorithms for Fastest Path Problem in Temporal Graphs'
    """

    def __init__(self, esd_graph: ESD_graph):
        if not isinstance(esd_graph, ESD_graph):
            raise TypeError("Input must be an ESD_graph object.")
        self.esd_graph = esd_graph
        self._all_temporal_vertices = self._collect_all_vertices()
        logging.info("SerialESDG_FPD solver initialized.")

    def _collect_all_vertices(self):
        """Helper to get a set of all unique vertex IDs."""
        vertices = set()
        for node in self.esd_graph.nodes.values():
            vertices.add(str(node.u))  # Ensure consistency as strings
            vertices.add(str(node.v))
        return vertices

    def find_fastest_paths(self, source_vertex_s: str):
        """
        Executes the serial FPD algorithm from a given source vertex.
        
        Ref: "In each iteration, all the unvisited reachable nodes are identified, 
        journey times are computed... and maintain the minimum journey time." [cite: 194]

        Args:
            source_vertex_s: The starting vertex in the original temporal graph.

        Returns:
            A tuple containing:
            - A dictionary of {destination: fastest_duration}.
            - A dictionary of {destination: path_as_list_of_esd_nodes}.
        """
        start_time = time.perf_counter()
        source_vertex_s = str(source_vertex_s) # Ensure type consistency
        logging.info(f"Starting FPD calculation from source: '{source_vertex_s}'")

        # 1. Initialization
        # "journey[v] is set to inf for all vertices... excluding source... set to 0" [cite: 218]
        journey_times = {vertex: float('inf') for vertex in self._all_temporal_vertices}
        journey_times[source_vertex_s] = 0
        
        # Dictionary to store the reconstructed path: {destination_vertex: [Node, Node, ...]}
        fastest_paths = {}
        
        # --- SEPARATED PREDECESSOR MAPS ---
        # 1. Tracks the BFS tree within the ESD graph (Edge ID -> Parent Edge ID)
        esdg_parent_tree = {} 
        # 2. Tracks which ESDG node provided the best time for a vertex (Vertex ID -> (Final Edge ID, Source Edge ID))
        vertex_best_entry = {}

        # "We ignore already visited nodes since they cannot offer a shorter journey time." 
        visited_esdg_nodes = set()

        # 2. Identify and Sort Source Nodes
        # "Iterates over all the source nodes... in decreasing order based on their departure time." [cite: 193]
        source_nodes_in_esdg = [
            node for node in self.esd_graph.nodes.values() if str(node.u) == source_vertex_s
        ]
        source_nodes_in_esdg.sort(key=lambda node: node.t, reverse=True)
        logging.info(f"Found and sorted {len(source_nodes_in_esdg)} ESDG source nodes.")
        
        if not source_nodes_in_esdg:
            logging.warning(f"No source nodes found for vertex {source_vertex_s}.")
            return journey_times, fastest_paths
        
        # Performance tracking
        total_nodes_processed = 0
        total_phases_executed = 0

        # 3. Iterate and Traverse (Algorithm 1 Logic)
        for i, start_node in enumerate(source_nodes_in_esdg):
            
            # Pruning: "Ignore already visited nodes" 
            if start_node.original_edge_id in visited_esdg_nodes:
                continue
            
            # Initialize frontier
            queue = collections.deque([start_node.original_edge_id])
            visited_esdg_nodes.add(start_node.original_edge_id)
            total_phases_executed += 1
            nodes_in_this_phase = 0

            # Mark the start of this specific BFS tree
            esdg_parent_tree[start_node.original_edge_id] = None 

            while queue:
                current_node_id = queue.popleft()
                current_node = self.esd_graph.nodes[current_node_id]
                nodes_in_this_phase += 1
                total_nodes_processed += 1

                # --- Core FPD Calculation ---
                # Journey = Arrival at current edge - Departure of the source edge of this phase
                current_journey = current_node.a - start_node.t
                destination_vertex = str(current_node.v)
                
                # --- Update Journey Times ---
                # "Maintain the minimum journey time at each vertex" [cite: 194]
                if current_journey < journey_times[destination_vertex]:
                    journey_times[destination_vertex] = current_journey
                    
                    # Record the entry point for path reconstruction
                    # We store: (The edge that reached the vertex, The edge that started this phase)
                    vertex_best_entry[destination_vertex] = (current_node_id, start_node.original_edge_id)
                    
                    # Log only significant updates to avoid clutter
                    # logging.debug(f"New fastest path to '{destination_vertex}': {current_journey}s")

                # --- Continue Traversal ---
                for neighbor_id in self.esd_graph.adj.get(current_node_id, []):
                    # Pruning check
                    if neighbor_id not in visited_esdg_nodes:
                        visited_esdg_nodes.add(neighbor_id)
                        queue.append(neighbor_id)
                        
                        # Update the internal BFS tree
                        esdg_parent_tree[neighbor_id] = current_node_id

            logging.debug(f"Phase {i+1} completed: processed {nodes_in_this_phase} nodes")

        # 4. Reconstruct Paths
        # This occurs strictly AFTER the calculation to ensure we use the global optimums found.
        for dest_vertex in self._all_temporal_vertices:
             if dest_vertex in vertex_best_entry and dest_vertex != source_vertex_s:
                path = []
                try:
                    # Retrieve the best entry point for this vertex
                    final_esdg_id, start_esdg_id = vertex_best_entry[dest_vertex]
                    
                    curr_id = final_esdg_id
                    path_found = True
                    
                    # Backtrack through the ESDG tree
                    while curr_id != start_esdg_id:
                        path.append(self.esd_graph.nodes[curr_id])
                        
                        # Safety check for broken chains
                        if curr_id not in esdg_parent_tree:
                            path_found = False
                            break
                            
                        curr_id = esdg_parent_tree[curr_id]
                        
                        # Stop if we hit None prematurely
                        if curr_id is None and curr_id != start_esdg_id: 
                             path_found = False
                             break

                    if path_found:
                        # Append the starting node of the path
                        path.append(self.esd_graph.nodes[start_esdg_id])
                        fastest_paths[dest_vertex] = path[::-1] # Reverse: Start -> End
                        
                except Exception as e:
                    logging.error(f"Error reconstructing path for {dest_vertex}: {e}")
                    continue

        end_time = time.perf_counter()
        logging.info(f"Serial FPD Summary:")
        logging.info(f"  - Phases executed: {total_phases_executed}")
        logging.info(f"  - Total nodes processed: {total_nodes_processed}")
        logging.info(f"  - Computation time: {end_time - start_time:.4f} seconds")
        
        return journey_times, fastest_paths