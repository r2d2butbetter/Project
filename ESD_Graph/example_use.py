from esd_transformer import transform_temporal_to_esd
from structures.temporal_graph import Temporal_edge

if __name__ == "__main__":
    # Define the temporal graph G from the image.
    # The list index will be the edge's internal ID (0-13), while the comment
    # refers to the e_id in the image (1-14).
    # Format: (u, v, t, λ)
    temporal_graph_G = [
        # Note: Vertices are stored as strings to handle any label type.
        ('2', '3', 3, 4),  # e1 (code id: 0)
        ('2', '4', 1, 3),  # e2 (code id: 1)
        ('1', '2', 1, 4),  # e3 (code id: 2)
        ('1', '2', 4, 3),  # e4 (code id: 3)
        ('3', '8', 7, 5),  # e5 (code id: 4)
        ('3', '8', 8, 6),  # e6 (code id: 5)
        ('4', '5', 4, 3),  # e7 (code id: 6)
        ('2', '5', 6, 3),  # e8 (code id: 7)
        ('8', '7', 15, 2), # e9 (code id: 8)
        ('5', '6', 8, 4),  # e10 (code id: 9)
        ('5', '7', 9, 4),  # e11 (code id: 10)
        ('7', '10', 15, 2),# e12 (code id: 11)
        ('7', '9', 14, 3), # e13 (code id: 12)
        ('5', '6', 7, 6),  # e14 (code id: 13)
    ]

    print("Input Temporal Graph G (from image):")
    # Print with image e_id for clarity
    for i, edge_data in enumerate(temporal_graph_G):
        edge = Temporal_edge(id=i+1, u=edge_data[0], v=edge_data[1], departure=edge_data[2], duration=edge_data[3])
        print(f"  {edge}")
    print("-" * 30)

    # Perform the transformation
    vanilla_esd_graph = transform_temporal_to_esd(temporal_graph_G)

    # Print the resulting ESD graph G'
    print("\n" + "="*40)
    print("  RESULT: Vanilla ESD Graph G'  ")
    print("="*40)
    print(vanilla_esd_graph)
    print("="*40)