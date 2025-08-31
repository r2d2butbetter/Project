import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

# Load dataset
df = pd.read_csv("network_temporal_day.csv", sep=";")

# Load only the first 1000 lines
df = pd.read_csv("network_temporal_day.csv", sep=";", nrows=10)

# Convert Unix timestamps to datetime (optional for clarity)
df["dep_time"] = pd.to_datetime(df["dep_time_ut"], unit="s")
df["arr_time"] = pd.to_datetime(df["arr_time_ut"], unit="s")

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Define positions for stops (fixed layout, e.g., spring layout)
all_stops = set(df["from_stop_I"]).union(set(df["to_stop_I"]))
G_static = nx.DiGraph()
G_static.add_nodes_from(all_stops)
pos = nx.spring_layout(G_static, seed=42)  # fixed layout

# Animation function
def update(frame_time):
    ax.clear()
    
    # Filter edges active at this frame_time
    active_edges = df[(df["dep_time_ut"] <= frame_time) & (df["arr_time_ut"] >= frame_time)]
    
    G = nx.DiGraph()
    G.add_nodes_from(all_stops)
    for _, row in active_edges.iterrows():
        G.add_edge(row["from_stop_I"], row["to_stop_I"])
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="orange", arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    
    # Add timestamp label
    ts = datetime.datetime.utcfromtimestamp(frame_time)
    ax.set_title(f"Active connections at {ts.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
    ax.axis("off")

# Define time range for animation
time_min, time_max = df["dep_time_ut"].min(), df["arr_time_ut"].max()

ani = FuncAnimation(fig, update, frames=range(time_min, time_max, 60), interval=1000)
plt.show()
