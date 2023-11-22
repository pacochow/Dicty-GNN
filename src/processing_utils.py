import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def dereference_refs(data, h5_file):
    # If the data is a NumPy array of object references, dereference them
    if data.dtype == 'O':
        dereferenced = np.empty(data.shape, dtype=object)
        for i in np.ndindex(data.shape):
            ref = data[i]
            if isinstance(ref, h5py.Reference):
                dereferenced[i] = h5_file[ref][()]
            else:
                dereferenced[i] = ref
        return dereferenced
    return data

def convert_dataset_to_array(dataset, h5_file):
    data = dataset[()]
    return dereference_refs(data, h5_file)

def matlab_struct_to_python_dict(h5_file, group_name):
    group = h5_file[group_name]
    python_dict = {}
    for field_name, dataset in group.items():
        python_dict[field_name] = convert_dataset_to_array(dataset, h5_file)
    return python_dict



def get_train_test_indices(popState, t, order = 2):
    # Find common cell IDs between consecutive time points
    common_nodes = popState['i'][t][0].flatten()
    for i in range(1, order+1):
        common_nodes = np.intersect1d(common_nodes,popState['i'][t+i][0].flatten())

    indices = np.zeros((order, len(common_nodes)))
    
    for i in range(order):
        # Find indices of those IDs at first time point
        indices[i] = np.where(np.isin(common_nodes,popState['i'][t+i][0].astype(int)))[0]

    # Find indices of those IDs at second time point
    label_indices = np.where(np.isin(common_nodes, popState['i'][t+order][0].astype(int)))[0]

    return indices.astype(int), label_indices.astype(int)

def get_data_at_timepoint(popState, t, indices, label_indices, order):
    # Get training data
    data = np.zeros((indices.shape[-1], order))
    for i in range(order):
        data[:, i:i+1] = popState['tOn'][t+i][0][indices[i]]
    
    # Get training labels
    label_activation = popState['tOn'][t+order][0][label_indices]
    labels = label_activation

    return data, labels

def find_balanced_dataset_timepoint(popState, order):
    timepoints = []
    for t in range(order, len(popState['x'])):
        train_indices, train_label_indices = get_train_test_indices(popState, t-order, order)
        _, train_label_t = get_data_at_timepoint(popState, t-order, train_indices, train_label_indices, order)
        train_label_t[np.isnan(train_label_t)] = 0
        if 0.48<train_label_t.sum()/len(train_label_t) < 0.52:
            timepoints.append(t)
    return timepoints

def get_adjacency_matrix(popState, t, train_indices, cutoff = 50, weight = False, order = 2):
    x = popState['x'][t+order-1][0][train_indices]
    y = popState['y'][t+order-1][0][train_indices]
    dist = np.sqrt((x-x.T)**2+(y-y.T)**2)
    if weight == False:
        adjacency = dist<cutoff
    else:
        adjacency = dist*(dist<cutoff)
    np.fill_diagonal(adjacency, 0)
    return dist, adjacency

def graph_weight_matrix(matrix):
    # Create a graph from the weight matrix
    G = nx.from_numpy_array(matrix)

    # Draw the graph
    pos = nx.kamada_kawai_layout(G, weight='weight')  # positions for all nodes based on edge weight

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=0.5)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.1)

    plt.axis('off')
    plt.show()