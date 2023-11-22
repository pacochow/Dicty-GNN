from src.data_processing import *
from src.model import *
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
model = GCN(100, 5)
model.load_state_dict(torch.load("models/22_30000_4fc.pt"))
order = model.order
popState = load_data()



def evaluate_model_by_time(popState, order, timepoints):
    test_data, test_labels, coos, edge_weights = prepare_test_data(popState, order, timepoints)
    accuracies = []
    for i in tqdm(range(len(timepoints))):
        data = Data(x = torch.tensor(test_data[i]).float(), edge_index = torch.tensor(coos[i]), edge_attr = torch.tensor(edge_weights[i]).unsqueeze(1), y = torch.tensor(test_labels[i]))
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            accuracy = (100*(1-(data.y.int()-torch.sigmoid(out).round()).abs().sum()/len(out))).detach().numpy()
        accuracies.append(accuracy)
    return accuracies


accuracies = evaluate_model_by_time(popState, order, np.arange(order, popState['x'].shape[0]))
plt.plot(accuracies);
