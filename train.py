from src.data_processing import *
from src.model import *
import torch.nn.functional as F
import torch

torch.manual_seed(0)

order = 5
popState = load_data()
train_data, train_labels, coos, edge_weights, timepoints = prepare_train_data(popState, order = order)
model = GCN(hidden_channels = 100, order = order)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# data = data.to(device)



# Initialize optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

criterion = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

losses = []
accuracies = []
epochs = 30000
for epoch in range(epochs):
    data = package_data(train_data, train_labels, coos, edge_weights, timepoints)
    data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y.float())
    losses.append(loss.item())
    with torch.no_grad():
        accuracy = (100*(1-(data.y.int()-torch.sigmoid(out).round()).abs().sum()/len(out))).detach().numpy()
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Loss at epoch {epoch}: {loss.item():.2f}")
    loss.backward()
    # Access gradients of model parameters
    # print(model.out5.weight)
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")

    optimizer.step()


torch.save(model.state_dict(), f"models/{len(timepoints)}_{epochs}_4fc.pt")


