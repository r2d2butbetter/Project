# classes for ESD graph and the node struct

from dataclasses import dataclass
from typing import Dict, List
import collections

@dataclass
class ESD_Node:
    """
    Class to represent a single node in the new G' graph

    Each node in this graph is a edge in G(the temporal graph)

    each node is u, v, t, a=t+lambda(see the temporal graph class): 
    """

    original_edge_id: int
    u: int #left vertex
    v: int #right
    t: int #departure
    a: int #arrival time

    def __repr__(self):
        return (f"v_e{self.original_edge_id}[{self.u}->{self.v} | "
                f"dep:{self.t}, arr:{self.a}]")

    def __hash__(self):
        return hash(self.original_edge_id) #hash using unique node id

    def __eq__(self, other):
        # TODO: the other must be an instance of this class itself
        return self.original_edge_id == other.original_edge_id



class ESD_graph:
    """Just a representation of the ESD graph using 
    Adjacency lists
    """

    def __init__(self):
        self.nodes: Dict[int, ESD_Node] = {}
        self.adj: Dict[int, List[int]] = collections.defaultdict(list)
        self.levels: Dict[int, int] = {}

    def add_node(self, node: ESD_Node):
        # add node to the graph
        if node.original_edge_id not in self.nodes:
            self.nodes[node.original_edge_id] = node

    def add_edge(self, from_id: int, to_id: int):
        # add a directed edge bw 2 nodes
        if from_id in self.nodes and to_id in self.nodes:
            self.adj[from_id].append(to_id) # may have multiple edges going outward
        else:
            raise ValueError("One or both nodes were not in the graph")

    def calculate_levels(self):
        """
        This confirms that the graph is a DAG
        level = 1 for source node
        level = 1+ max(level(parents)) for all parents of x
        """

        #reverse adjacency list
        parents = collections.defaultdict(list)
        for u_id, neighbours in self.adj.items():
            for v_id in neighbours:
                parents[v_id].append(u_id)

        memo = {} # Memoization for the recursive calculation

        def get_level(node_id: int) -> int:
            # If level is already computed, return it
            if node_id in memo:
                return memo[node_id]

            # Base case: Node has no parents, level is 1
            if not parents[node_id]:
                memo[node_id] = 1
                return 1

            # Recursive step: 1 + max level of parents
            max_parent_level = 0
            for parent_id in parents[node_id]:
                max_parent_level = max(max_parent_level, get_level(parent_id))
            
            memo[node_id] = 1 + max_parent_level
            return memo[node_id]

        # Calculate level for every node in the graph
        for node_id in self.nodes:
            self.levels[node_id] = get_level(node_id)
            
        print("Node levels calculated.")

    def __repr__(self) -> str:
        output = ["ESD Graph G̃ (Adjacency List):"]
        # Sort nodes by ID for consistent output
        sorted_node_ids = sorted(self.nodes.keys())
        
        for node_id in sorted_node_ids:
            node = self.nodes[node_id]
            level_info = f" (Lvl: {self.levels.get(node_id, 'N/A')})"
            
            successors = ""
            if self.adj[node_id]:
                # Get short representation of successor nodes for printing
                successor_reprs = [f"v_e{s_id}" for s_id in sorted(self.adj[node_id])]
                successors = f" -> [{', '.join(successor_reprs)}]"
            
            output.append(f"  {node}{level_info}{successors}")
        return "\n".join(output)