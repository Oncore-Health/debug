import networkx as nx
import matplotlib.pyplot as plt
import math
import pandas as pd


def create_graph():
    # Recreate the graph with all nodes, ensuring to include any missed sub-nodes (A, B, etc.) and their connections
    G = nx.Graph()

    # Define nodes for Floor 1 with specific shapes and sub-nodes
    zones_floor_1 = {
        'Zone 1': {'nodes': ['422', '423', '424A', '424B', '425A', '425B', '426', '427', '429', '430'],
                   'shape': 'square'},
        'Zone 2': {
            'nodes': ['433', '434A', '434B', '434C', '436A', '436B', '436C', '436D', '436E', '436F', '440A', '440B'],
            'shape': 'triangle'},
        'Zone 3': {'nodes': ['441A', '441B', '442A', '442B', '443A', '443B', '445A', '445B', '446', '460A', '460B'],
                   'shape': 'line'}
    }

    # Define nodes for Floor 2 with specific shapes
    zones_floor_2 = {
        'Apheresis Bay': {'nodes': ['4030A', '4030B', '4030C', '4030D'], 'shape': 'square'},
        'Zone 4: a': {'nodes': ['407', '408'], 'shape': 'line'},
        'Zone 4': {'nodes': ['410A', '410B', '411A', '411B', '412', '413', '414', '415', '416'], 'shape': 'line'}
    }

    # Function to add edges in different shapes
    def add_shape_edges(G, nodes, shape):
        if shape == 'square':
            # Make a square by connecting in a cycle
            for i in range(len(nodes)):
                G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
        elif shape == 'triangle':
            # Make a triangle by connecting all nodes to each other
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    G.add_edge(nodes[i], nodes[j])
        elif shape == 'line':
            # Connect nodes in a line
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])

    # Adding nodes and creating shapes
    for zone, info in zones_floor_1.items():
        G.add_nodes_from(info['nodes'])
        for node in info['nodes']:
            G.nodes[node]['zone'] = zone
            G.nodes[node]['floor'] = 'floor_1'
        add_shape_edges(G, info['nodes'], info['shape'])

    for zone, info in zones_floor_2.items():
        G.add_nodes_from(info['nodes'])
        for node in info['nodes']:
            G.nodes[node]['zone'] = zone
            G.nodes[node]['floor'] = 'floor_2'
        add_shape_edges(G, info['nodes'], info['shape'])

    # Connect nodes within the same room if they are labeled with A, B, etc.
    room_connections_floor_1 = [('424A', '424B'), ('425A', '425B'), ('440A', '440B'), ('460A', '460B')]
    room_connections_floor_2 = [('4030A', '4030B'), ('4030B', '4030C'), ('4030C', '4030D')]

    G.add_edges_from(room_connections_floor_1)
    G.add_edges_from(room_connections_floor_2)

    # Add a few edges connecting different zones within each floor
    cross_zone_edges_floor_1 = [('422', '433'), ('424A', '440A')]
    G.add_edges_from(cross_zone_edges_floor_1)

    cross_zone_edges_floor_2 = [('4030D', '407'), ('416', '408')]
    G.add_edges_from(cross_zone_edges_floor_2)

    # Define positions manually to make the graph more organized
    pos = {
        '422': (-5, 2), '423': (-4, 2), '424A': (-3, 2), '424B': (-3, 1.5),
        '425A': (-2, 2), '425B': (-2, 1.5), '426': (-1, 2), '427': (0, 2),
        '429': (1, 2), '430': (2, 2),  # Zone 1
        '433': (2, 1), '434A': (2, 0.5), '434B': (2, 0), '434C': (2, -0.5),
        '436A': (0, -1.5), '436B': (0, -1), '436C': (1, -1), '436D': (1, -1.5), '436E': (2, -1.5), '436F': (2, -1),
        # Zone 2
        '440A': (-1, -1.5), '440B': (-1, -1), '441A': (-2, -1.5), '441B': (-2, -1), '442A': (-3, -1.5),
        '442B': (-3, -1),
        '443A': (-4, -1.5), '443B': (-4, -1), '445A': (-5, -1.5), '445B': (-5, -1), '446': (-6, -1.25), '460A': (-5, 1),
        '460B': (-5, 0.5),  # Zone 3
        '4030A': (-5, -3), '4030B': (-5, -3.5), '4030C': (-4, -3), '4030D': (-4, -3.5),  # Zone 4
        '401': (-4, -3), '403': (-3, -4), '404': (-2, -3),  # Zone 4
        '406': (1, -3), '407': (3, -3.5),  # Zone 5
        '408': (3, -6), '409': (-3, -6), '410A': (2, -7), '410B': (2, -7.5), '411A': (1, -7), '411B': (1, -7.5),
        '412': (0, -7.25), '413': (-1, -7.25), '414': (-2, -7.25), '415': (-3, -7.25), '416': (-4, -7.25)  # Zone 6
    }

    # Calculate edge weights based on positions
    for edge in G.edges():
        node1, node2 = edge
        # Extract numeric parts from node names (ignoring suffixes like A, B)
        numeric1 = ''.join(filter(str.isdigit, node1))
        numeric2 = ''.join(filter(str.isdigit, node2))
        if numeric1 == numeric2:  # If nodes have the same base number (e.g., 424A and 424B)
            G[node1][node2]['weight'] = 0
        else:
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            G[node1][node2]['weight'] = int(round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 0))

    return G, pos


def draw_graph():
    G, pos = create_graph()

    plt.figure(figsize=(16, 16))

    # Draw nodes and edges for Floor 1
    floor_1_nodes = [n for n in G.nodes() if G.nodes[n].get('floor') == 'floor_1']
    nx.draw_networkx_nodes(G, pos, nodelist=floor_1_nodes, node_color='lightgreen', node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if e[0] in floor_1_nodes and e[1] in floor_1_nodes],
                           edge_color='green')

    # Draw nodes and edges for Floor 2
    floor_2_nodes = [n for n in G.nodes() if G.nodes[n].get('floor') == 'floor_2']
    nx.draw_networkx_nodes(G, pos, nodelist=floor_2_nodes, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if e[0] in floor_2_nodes and e[1] in floor_2_nodes],
                           edge_color='blue')

    # Draw labels for all nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw edge labels to show weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Hospital Floor Layout with Detailed Room Nodes and Edges (Edge Weights Included)")
    plt.show()


def floyd_warshall(chairs):
    G, pos = create_graph()

    # Apply the Floyd-Warshall algorithm to compute the shortest paths
    shortest_path_matrix = nx.floyd_warshall_numpy(G)

    # Get the list of nodes in the graph (ordered by their appearance in G.nodes)
    nodes = list(G.nodes)

    # Create a dictionary to map chairName to chairId
    name_to_id = {chair['name']: chair['id'] for chair in chairs}

    # Translate node names (chairName) to chair ids
    translated_nodes = [name_to_id[node] for node in nodes]

    # Create the dataframe with the translated node ids
    df = pd.DataFrame(shortest_path_matrix, index=translated_nodes, columns=translated_nodes)

    return df


# the prints

# Display the shortest path distance matrix
# print("Shortest Path Distance Matrix (Floyd-Warshall):")
# print(shortest_path_matrix)

# # Optionally, print the matrix with row/column headers for better clarity
# print("\nMatrix with Node Labels:")
# print(f"{'':>10}", end="")  # Empty space for the top left corner
# for node in nodes:
#     print(f"{node:>10}", end="")
# print()

# for i, node in enumerate(nodes):
#     print(f"{node:>10}", end="")
#     for j in range(len(nodes)):
#         print(f"{shortest_path_matrix[i, j]:>10.2f}", end="")
#     print()

# print(floyd_warshall())

create_graph()