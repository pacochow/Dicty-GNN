from src.processing_utils import * 
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import block_diag
import cv2
import os

def load_data():
    # Open the HDF5 file
    with h5py.File('Data/popSig.mat', 'r') as f:
        # Convert the MATLAB structured array to a Python dictionary
        popState = matlab_struct_to_python_dict(f, 'popState')
    return popState

def prepare_train_data(popState, order = 2):

    timepoints = find_balanced_dataset_timepoint(popState, order)

    # timepoints = np.arange(1000, 1500)

    train_data = []
    train_labels = []
    coos = []
    edge_weights = []

    for t in tqdm(timepoints):
        train_indices, train_label_indices = get_train_test_indices(popState, t-order, order)
        train_data_t, train_label_t = get_data_at_timepoint(popState, t-order, train_indices, train_label_indices, order)
        train_data_t[np.isnan(train_data_t)] = 0 
        train_label_t[np.isnan(train_label_t)] = 0
        dist, adjacency = get_adjacency_matrix(popState, t-order, train_indices[-1], order=order)

        train_data.append(train_data_t)
        train_labels.append(train_label_t)

        coo = coo_matrix(adjacency)
        coo = np.vstack((coo.row, coo.col))
        coos.append(coo)

        edge_weights.append(100/dist[coo[0], coo[1]])

    return train_data, train_labels, coos, edge_weights, timepoints

    # train_indices, train_label_indices = get_train_test_indices(popState, balanced_timepoints[0]-order, order = order)
    # train_data, train_labels = get_data_at_timepoint(popState, balanced_timepoints[0]-order, train_indices, train_label_indices, order)
    # train_dist, train_adjacency = get_adjacency_matrix(popState, balanced_timepoints[0]-order, train_indices[-1])
    # for t in tqdm(balanced_timepoints[1:5]):
    #     train_indices, train_label_indices = get_train_test_indices(popState, t-order, order)
    #     train_data_t, train_label_t = get_data_at_timepoint(popState, t-order, train_indices, train_label_indices, order)
    #     dist, adjacency = get_adjacency_matrix(popState, t-order, train_indices[-1], order)
 
    #     train_data = np.vstack((train_data, train_data_t))
    #     train_labels = np.vstack((train_labels, train_label_t))
    #     train_adjacency = block_diag(train_adjacency, adjacency)
    #     train_dist = block_diag(train_dist, dist)
    # coo = coo_matrix(train_adjacency)
    # coo = np.vstack((coo.row, coo.col))

    # edge_weights = 100/train_dist[coo[0], coo[1]]

    # train_data[np.isnan(train_data)] = 0 
    # train_labels[np.isnan(train_labels)] = 0

    # data = Data(x = torch.tensor(train_data).float(), edge_index = torch.tensor(coo), edge_attr = torch.tensor(edge_weights), y = torch.tensor(train_labels))
    
    # return data

def package_data(data, labels, coos, edge_weights, timepoints):
    random_index = np.random.randint(len(timepoints))
    data = Data(x = torch.tensor(data[random_index]).float(), edge_index = torch.tensor(coos[random_index]), edge_attr = torch.tensor(edge_weights[random_index]).unsqueeze(1), y = torch.tensor(labels[random_index]))

    return data

def prepare_test_data(popState, order, timepoints):


    # test_indices, test_label_indices = get_train_test_indices(popState, timepoints[0], order = order)
    # test_data, test_labels = get_data_at_timepoint(popState, timepoints[0], test_indices, test_label_indices, order)
    # test_dist, test_adjacency = get_adjacency_matrix(popState, timepoints[0], test_indices[-1])
    # for t in tqdm(timepoints[1:]):
    #     test_indices, test_label_indices = get_train_test_indices(popState, t, order)
    #     test_data_t, test_label_t = get_data_at_timepoint(popState, t, test_indices, test_label_indices, order)
    #     dist, adjacency = get_adjacency_matrix(popState, t, test_indices[-1], order)
 
    #     test_data = np.vstack((test_data, test_data_t))
    #     test_labels = np.vstack((test_labels, test_label_t))
    #     test_adjacency = block_diag(test_adjacency, adjacency)
    #     test_dist = block_diag(test_dist, dist)
    # coo = coo_matrix(test_adjacency)
    # coo = np.vstack((coo.row, coo.col))

    # edge_weights = 100/test_dist[coo[0], coo[1]]

    # test_data[np.isnan(test_data)] = 0 
    # test_labels[np.isnan(test_labels)] = 0

    # data = Data(x = torch.tensor(test_data).float(), edge_index = torch.tensor(coo), edge_attr = torch.tensor(edge_weights), y = torch.tensor(test_labels))
    
    # return data
    

    test_data = []
    test_labels = []
    coos = []
    edge_weights = []

    for t in tqdm(timepoints):
        test_indices, test_label_indices = get_train_test_indices(popState, t-order, order)
        test_data_t, test_label_t = get_data_at_timepoint(popState, t-order, test_indices, test_label_indices, order)
        test_data_t[np.isnan(test_data_t)] = 0 
        test_label_t[np.isnan(test_label_t)] = 0
        dist, adjacency = get_adjacency_matrix(popState, t-order, test_indices[-1], order=order)

        test_data.append(test_data_t)
        test_labels.append(test_label_t)

        coo = coo_matrix(adjacency)
        coo = np.vstack((coo.row, coo.col))
        coos.append(coo)

        edge_weights.append(100/dist[coo[0], coo[1]])

    return test_data, test_labels, coos, edge_weights


def generate_video(popState):
    # Directory to store images
    image_dir = 'scatter_images'
    os.makedirs(image_dir, exist_ok=True)

    # Generate and save scatter plots
    for t in range(popState['x'].shape[0]):
        plt.scatter(popState['x'][t][0], popState['y'][t][0], s = 0.1, c = popState['tOn'][t][0])  # Replace with your scatter plot code
        plt.title(f"Frame {t}")
        plt.savefig(f"{image_dir}/plot_{t}.png")
        plt.close()

    # Create a video from images
    video_name = 'scatter_video.mp4'
    frame = cv2.imread(os.path.join(image_dir, 'plot_0.png'))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(video_name, fourcc, 60, (width, height))

    for i in range(popState['x'].shape[0]):
        video.write(cv2.imread(os.path.join(image_dir, f'plot_{i}.png')))

    cv2.destroyAllWindows()
    video.release()