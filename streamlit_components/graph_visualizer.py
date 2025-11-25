import plotly.graph_objects as go
import networkx as nx
import numpy as np
from collections import defaultdict

def create_graph_topology_view(esd_graph, max_nodes=500):
    """
    Create an interactive 3D visualization of the ESD graph topology
    """
    # Limit nodes for performance
    node_ids = list(esd_graph.nodes.keys())[:max_nodes]
    
    # Build NetworkX graph
    G = nx.DiGraph()
    for node_id in node_ids:
        G.add_node(node_id)
        neighbors = esd_graph.adj.get(node_id, [])
        for neighbor in neighbors:
            if neighbor in node_ids:
                G.add_edge(node_id, neighbor)
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extract coordinates
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]
    
    # Edge coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(125,125,125,0.3)', width=1),
        hoverinfo='none',
        name='Edges'
    )
    
    # Get node degrees for sizing
    node_degrees = [G.degree(node) for node in G.nodes()]
    
    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=[3 + deg * 0.5 for deg in node_degrees],
            color=node_degrees,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Degree'),
            line=dict(width=0.5, color='white')
        ),
        text=[f'Node: {node}<br>Degree: {deg}' for node, deg in zip(G.nodes(), node_degrees)],
        hoverinfo='text',
        name='Nodes'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=f'ESD Graph Topology (showing {len(G.nodes())} nodes)',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    return fig

def create_temporal_heatmap(esd_graph, max_nodes=1000):
    """
    Create a heatmap showing temporal connectivity patterns
    """
    node_ids = list(esd_graph.nodes.keys())[:max_nodes]
    
    # Extract temporal information
    u_vertices = []
    v_vertices = []
    departure_times = []
    arrival_times = []
    
    for node_id in node_ids:
        node = esd_graph.nodes[node_id]
        u_vertices.append(int(node.u))
        v_vertices.append(int(node.v))
        departure_times.append(node.t)
        arrival_times.append(node.a)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=departure_times,
        y=arrival_times,
        mode='markers',
        marker=dict(
            size=4,
            color=np.array(arrival_times) - np.array(departure_times),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Duration'),
            line=dict(width=0.5, color='white')
        ),
        text=[f'From: {u}<br>To: {v}<br>Dep: {d}<br>Arr: {a}<br>Duration: {a-d}' 
              for u, v, d, a in zip(u_vertices, v_vertices, departure_times, arrival_times)],
        hoverinfo='text',
        name='Connections'
    ))
    
    fig.update_layout(
        title='Temporal Connectivity Pattern',
        xaxis_title='Departure Time',
        yaxis_title='Arrival Time',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_degree_distribution(esd_graph):
    """
    Create degree distribution visualization
    """
    # Calculate in-degree and out-degree
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for node_id in esd_graph.nodes.keys():
        out_degree[node_id] = len(esd_graph.adj.get(node_id, []))
        
    for neighbors in esd_graph.adj.values():
        for neighbor in neighbors:
            in_degree[neighbor] += 1
    
    # Get distributions
    in_degrees = list(in_degree.values())
    out_degrees = list(out_degree.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=in_degrees,
        name='In-Degree',
        marker_color='#4ECDC4',
        opacity=0.7,
        nbinsx=50
    ))
    
    fig.add_trace(go.Histogram(
        x=out_degrees,
        name='Out-Degree',
        marker_color='#FF6B6B',
        opacity=0.7,
        nbinsx=50
    ))
    
    fig.update_layout(
        title='Degree Distribution',
        xaxis_title='Degree',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_path_visualization(path_nodes, esd_graph):
    """
    Visualize a specific path through the graph
    """
    if not path_nodes:
        return None
    
    # Build graph with path
    G = nx.DiGraph()
    
    # Add path edges
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i].original_edge_id
        v = path_nodes[i + 1].original_edge_id
        G.add_edge(u, v)
    
    # Add context (neighbors)
    for node in path_nodes:
        node_id = node.original_edge_id
        neighbors = esd_graph.adj.get(node_id, [])
        for neighbor in neighbors[:5]:  # Limit context
            G.add_edge(node_id, neighbor)
    
    pos = nx.spring_layout(G, seed=42)
    
    # Separate path nodes and context nodes
    path_ids = {node.original_edge_id for node in path_nodes}
    
    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(125,125,125,0.5)', width=1),
        hoverinfo='none',
        showlegend=False
    )
    
    # Path nodes
    path_x = [pos[node.original_edge_id][0] for node in path_nodes]
    path_y = [pos[node.original_edge_id][1] for node in path_nodes]
    
    path_trace = go.Scatter(
        x=path_x, y=path_y,
        mode='markers+lines',
        marker=dict(size=15, color='red', line=dict(width=2, color='white')),
        line=dict(color='red', width=3),
        text=[f'Step {i+1}<br>Edge: {node.original_edge_id}<br>From: {node.u}<br>To: {node.v}' 
              for i, node in enumerate(path_nodes)],
        hoverinfo='text',
        name='Path'
    )
    
    # Context nodes
    context_nodes = [node for node in G.nodes() if node not in path_ids]
    if context_nodes:
        context_x = [pos[node][0] for node in context_nodes]
        context_y = [pos[node][1] for node in context_nodes]
        
        context_trace = go.Scatter(
            x=context_x, y=context_y,
            mode='markers',
            marker=dict(size=8, color='lightblue', line=dict(width=1, color='white')),
            text=[f'Node: {node}' for node in context_nodes],
            hoverinfo='text',
            name='Context'
        )
    else:
        context_trace = None
    
    # Create figure
    data = [edge_trace, path_trace]
    if context_trace:
        data.append(context_trace)
    
    fig = go.Figure(data=data)
    
    fig.update_layout(
        title='Path Visualization',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        height=500
    )
    
    return fig

def create_level_distribution(esd_graph):
    """
    Visualize the level distribution in the graph (for Level Order algorithm)
    """
    from collections import deque
    
    # Compute levels using BFS
    in_degree = {node_id: 0 for node_id in esd_graph.nodes}
    for neighbors in esd_graph.adj.values():
        for neighbor in neighbors:
            if neighbor in in_degree:
                in_degree[neighbor] += 1
    
    queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
    levels = {node_id: 1 for node_id in esd_graph.nodes}
    
    while queue:
        u = queue.popleft()
        current_level = levels[u]
        
        for v in esd_graph.adj.get(u, []):
            if v in levels:
                levels[v] = max(levels[v], current_level + 1)
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    # Count nodes per level
    level_counts = defaultdict(int)
    for level in levels.values():
        level_counts[level] += 1
    
    sorted_levels = sorted(level_counts.keys())
    counts = [level_counts[l] for l in sorted_levels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_levels,
        y=counts,
        marker_color='#95E1D3',
        text=counts,
        textposition='auto',
        hovertemplate='<b>Level %{x}</b><br>Nodes: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Graph Level Distribution',
        xaxis_title='Level',
        yaxis_title='Number of Nodes',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_connectivity_matrix(esd_graph, sample_size=100):
    """
    Create a connectivity matrix visualization
    """
    # Sample nodes
    node_ids = list(esd_graph.nodes.keys())[:sample_size]
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    
    # Build adjacency matrix
    matrix = np.zeros((len(node_ids), len(node_ids)))
    
    for i, node_id in enumerate(node_ids):
        neighbors = esd_graph.adj.get(node_id, [])
        for neighbor in neighbors:
            if neighbor in node_map:
                j = node_map[neighbor]
                matrix[i, j] = 1
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Connectivity Matrix (sample of {len(node_ids)} nodes)',
        xaxis_title='Target Node',
        yaxis_title='Source Node',
        height=500
    )
    
    return fig

def create_path_network_map(results, source_vertex, esd_graph, top_n=20):
    """
    Create an interactive network map showing paths to top N destinations
    with travel times and edge durations
    """
    if 'Serial' not in results or not results['Serial'].get('paths'):
        return None
    
    # Get top N destinations by shortest travel time
    journey_times = results['Serial']['data']
    sorted_dests = sorted(
        [(dest, time) for dest, time in journey_times.items() 
         if time != float('inf') and dest != source_vertex],
        key=lambda x: x[1]
    )[:top_n]
    
    if not sorted_dests:
        return None
    
    # Build network graph
    G = nx.DiGraph()
    edge_data = []
    node_info = {}
    
    # Add source
    G.add_node(source_vertex)
    node_info[source_vertex] = {
        'type': 'source',
        'journey_time': 0,
        'label': f'Source: {source_vertex}'
    }
    
    paths_dict = results['Serial']['paths']
    
    # Process each destination path
    for dest, journey_time in sorted_dests:
        if dest not in paths_dict:
            continue
            
        path_nodes = paths_dict[dest]
        if not path_nodes:
            continue
        
        # Add destination node
        if dest not in G:
            G.add_node(dest)
            node_info[dest] = {
                'type': 'destination',
                'journey_time': journey_time,
                'label': f'{dest}\--{journey_time}s'
            }
        
        # Add intermediate vertices and edges from path
        prev_vertex = source_vertex
        cumulative_time = 0
        
        for node in path_nodes:
            curr_vertex = str(node.v)
            edge_duration = node.a - node.t
            cumulative_time = node.a - path_nodes[0].t if path_nodes else edge_duration
            
            # Add intermediate node if not exists
            if curr_vertex not in G:
                G.add_node(curr_vertex)
                node_info[curr_vertex] = {
                    'type': 'intermediate',
                    'journey_time': cumulative_time,
                    'label': f'{curr_vertex}\--{cumulative_time}s'
                }
            
            # Add edge with duration
            if not G.has_edge(prev_vertex, curr_vertex):
                G.add_edge(prev_vertex, curr_vertex)
                edge_data.append({
                    'source': prev_vertex,
                    'target': curr_vertex,
                    'duration': edge_duration,
                    'edge_id': node.original_edge_id
                })
            
            prev_vertex = curr_vertex
    
    # Use hierarchical layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create edge traces
    edge_traces = []
    
    for edge_info in edge_data:
        src = edge_info['source']
        tgt = edge_info['target']
        
        if src not in pos or tgt not in pos:
            continue
        
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        
        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=2,
                color='rgba(100,100,100,0.5)'
            ),
            hoverinfo='text',
            text=f"Edge {edge_info['edge_id']}<br>Duration: {edge_info['duration']}s",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node traces by type
    node_types = {'source': [], 'destination': [], 'intermediate': []}
    node_texts = {'source': [], 'destination': [], 'intermediate': []}
    
    for node in G.nodes():
        if node not in pos or node not in node_info:
            continue
        
        x, y = pos[node]
        node_type = node_info[node]['type']
        
        node_types[node_type].append((x, y))
        
        info = node_info[node]
        hover_text = f"<b>Vertex: {node}</b><br>"
        hover_text += f"Journey Time: {info['journey_time']:.2f}s<br>"
        hover_text += f"Type: {node_type.title()}"
        node_texts[node_type].append(hover_text)
    
    # Create node scatter traces
    node_trace_source = go.Scatter(
        x=[p[0] for p in node_types['source']],
        y=[p[1] for p in node_types['source']],
        mode='markers+text',
        marker=dict(
            size=25,
            color='#FF6B6B',
            line=dict(width=3, color='white'),
            symbol='star'
        ),
        text=[node_info[n]['label'] for n in G.nodes() if n in node_info and node_info[n]['type'] == 'source'],
        textposition='top center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        hoverinfo='text',
        hovertext=node_texts['source'],
        name='Source',
        showlegend=True
    )
    
    node_trace_dest = go.Scatter(
        x=[p[0] for p in node_types['destination']],
        y=[p[1] for p in node_types['destination']],
        mode='markers+text',
        marker=dict(
            size=20,
            color='#4ECDC4',
            line=dict(width=2, color='white'),
            symbol='diamond'
        ),
        text=[node_info[n]['label'] for n in G.nodes() if n in node_info and node_info[n]['type'] == 'destination'],
        textposition='top center',
        textfont=dict(size=9, color='black'),
        hoverinfo='text',
        hovertext=node_texts['destination'],
        name=f'Top {top_n} Destinations',
        showlegend=True
    )
    
    node_trace_inter = go.Scatter(
        x=[p[0] for p in node_types['intermediate']],
        y=[p[1] for p in node_types['intermediate']],
        mode='markers',
        marker=dict(
            size=12,
            color='#95E1D3',
            line=dict(width=1, color='white'),
            symbol='circle'
        ),
        hoverinfo='text',
        hovertext=node_texts['intermediate'],
        name='Intermediate Nodes',
        showlegend=True
    )
    
    # Combine all traces
    fig = go.Figure(data=edge_traces + [node_trace_source, node_trace_dest, node_trace_inter])
    
    fig.update_layout(
        title=dict(
            text=f'Path Network Map: Source {source_vertex} to Top {top_n} Destinations',
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        plot_bgcolor='rgba(240,240,240,0.5)',
        margin=dict(l=20, r=20, t=60, b=20),
        height=700,
        hovermode='closest'
    )
    
    return fig

def create_journey_time_comparison_chart(results, source_vertex, top_n=20):
    """
    Create a bar chart comparing journey times to top N destinations
    """
    import plotly.express as px
    
    if 'Serial' not in results:
        return None
    
    journey_times = results['Serial']['data']
    sorted_dests = sorted(
        [(dest, time) for dest, time in journey_times.items() 
         if time != float('inf') and dest != source_vertex],
        key=lambda x: x[1]
    )[:top_n]
    
    if not sorted_dests:
        return None
    
    destinations = [d[0] for d in sorted_dests]
    times = [d[1] for d in sorted_dests]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=destinations,
        y=times,
        marker=dict(
            color=times,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Journey Time (s)')
        ),
        text=[f'{t:.2f}s' for t in times],
        textposition='outside',
        hovertemplate='<b>Destination: %{x}</b><br>Journey Time: %{y:.2f}s<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Journey Times: Top {top_n} Closest Destinations from {source_vertex}',
        xaxis_title='Destination Vertex',
        yaxis_title='Journey Time (seconds)',
        template='plotly_white',
        height=500,
        xaxis=dict(tickangle=-45)
    )
    
    return fig
