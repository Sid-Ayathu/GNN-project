import os
import gc
import csv
import torch
import itertools
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Yelp
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv, GATConv, BatchNorm ,TransformerConv
from torch.nn import BatchNorm1d





# 1. Load Dataset
local_path = r'C:/Users/Siddharth/Desktop/IIITB/IIITB sem 8/GNN/Project/data/Yelp'
if not os.path.exists(local_path):
    os.makedirs(local_path)

dataset = Yelp(root=local_path)
data = dataset[0]

# 2. Setup Mini-Batch Loaders (Kept very lean for the RTX 3050)
train_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=128, input_nodes=data.train_mask, shuffle=True)
val_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=256, input_nodes=data.val_mask, shuffle=False)
test_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=256, input_nodes=data.test_mask, shuffle=False)

# 3. Define the Architecture
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCEWithLogitsLoss()

# --- EVALUATION HELPER ---
@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.cuda.amp.autocast():
            out = model(batch.x, batch.edge_index)
        pred = (out[:batch.batch_size].sigmoid() > 0.5).float()
        y_pred.append(pred.cpu().numpy())
        y_true.append(batch.y[:batch.batch_size].cpu().numpy())
    
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average='micro')
    return acc, f1

# --- HYPERPARAMETER GRID ---
# Kept conservative to avoid OOM on 4GB VRAM
hidden_channels_opts = [16, 32]
heads_opts = [2, 4]
lr_opts = [0.01, 0.005]
dropout_opts = [0.3, 0.5]

# Generate all combinations
param_combinations = list(itertools.product(hidden_channels_opts, heads_opts, lr_opts, dropout_opts))

# Setup CSV Logging
csv_filename = "gatv2_hyperparam_results.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header if file is new
    if not file_exists:
        writer.writerow(['Hidden_Channels', 'Heads', 'Learning_Rate', 'Dropout', 'Best_Val_F1', 'Test_Acc', 'Test_F1'])

print(f"Starting Hyperparameter Sweep. Total combinations: {len(param_combinations)}")

# --- THE SWEEP LOOP ---
for idx, (h_channels, heads, lr, dropout) in enumerate(param_combinations):
    print(f"\n--- Run {idx+1}/{len(param_combinations)} | Channels: {h_channels}, Heads: {heads}, LR: {lr}, Dropout: {dropout} ---")
    
    # Initialize fresh model and optimizer for this run
    model = GAT(dataset.num_node_features, h_channels, dataset.num_classes, heads, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_f1 = 0.0
    final_test_acc = 0.0
    final_test_f1 = 0.0
    
    # Limit epochs for the sweep (e.g., 30) to save time
    for epoch in range(1, 10):
        # Training Phase
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Evaluation Phase
        val_acc, val_f1 = evaluate_loader(model, val_loader)
        scheduler.step(val_f1)
        
        # Track the best model in this specific run
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Get test metrics at the best validation point
            final_test_acc, final_test_f1 = evaluate_loader(model, test_loader)
            
        # Memory Management to survive the loop
        torch.cuda.empty_cache()
        gc.collect()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Val F1: {val_f1:.4f} | Best Val F1: {best_val_f1:.4f}")

    # Log the best result of this combination to the CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([h_channels, heads, lr, dropout, best_val_f1, final_test_acc, final_test_f1])
        
    print(f"Finished Run {idx+1}. Logged Best Val F1: {best_val_f1:.4f}")
    
    # Ultimate cleanup before the next architecture loads
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

print("\nHyperparameter sweep complete! Check 'gatv2_hyperparam_results.csv'.")