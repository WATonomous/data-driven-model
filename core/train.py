import argparse
import numpy as np
import pandas as pd
import torch
import ml_casadi.torch as mc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def parse_element(elem):
    if isinstance(elem, str):
        
        elem = elem.strip('[]')
        return np.array([float(x.strip()) for x in elem.split(',') if x.strip()])
    return np.array([float(elem)])

class SimpleDroneDataset(Dataset):
    def __init__(self, csv_file, ground_effect=False):
        data = pd.read_csv(csv_file)self.ground_effect = ground_effect
        if self.ground_effect:
            self._map_res = 0.1  
            self._static_ground_map = np.zeros((10, 10))  
            self._org_to_map_org = np.array([-0.5, -0.5])  
        self.x = []
        for _, row in data.iterrows():
            
            
            
            features_raw = row[:-3]      
            parsed_feats = [parse_element(elem) for elem in features_raw]
            base_features = np.concatenate(parsed_feats)      if self.ground_effect:
                
                ge_features = self._get_ground_effect_features(base_features)
                full_features = np.concatenate([base_features, ge_features])
            else:
                full_features = base_features    self.x.append(full_features)
        self.y = []
        for _, row in data.iterrows():
            last3 = row[-3:]
            parsed_y = [parse_element(elem) for elem in last3]
            self.y.append(np.concatenate(parsed_y))self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x = self.x_scaler.fit_transform(self.x)
        self.y = self.y_scaler.fit_transform(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = torch.tensor(self.x[idx], dtype=torch.float32)
        y_i = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_i, y_i

    def _get_ground_effect_features(self, features_17):
        """
        Here features_17 has shape (17,) => [p(3), q(4), v(3), r(3), controls(4)].
        We will add a 9-value patch plus orientation repeated => 13 more => total 30.
        """
        position = features_17[0:3]
        orientation = features_17[3:7]
        map_pos = (position[:2] - self._org_to_map_org) / self._map_res
        x_idx = int(np.clip(map_pos[0], 0, self._static_ground_map.shape[0] - 1))
        y_idx = int(np.clip(map_pos[1], 0, self._static_ground_map.shape[1] - 1))
        ground_effect_patch = np.zeros((3, 3))
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (0 <= x_idx+i < self._static_ground_map.shape[0] and
                    0 <= y_idx+j < self._static_ground_map.shape[1]):
                    ground_effect_patch[i+1, j+1] = self._static_ground_map[x_idx+i, y_idx+j]
        
        patch_and_orientation = np.concatenate([ground_effect_patch.flatten(), orientation])
        return patch_and_orientation

class NormalizedMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        
        self.model = mc.nn.MultiLayerPerceptron(
            input_size, hidden_size, output_size, num_layers, 'Tanh'
        )
        self.register_buffer('x_mean', torch.zeros(input_size))
        self.register_buffer('x_std', torch.ones(input_size))
        self.register_buffer('y_mean', torch.zeros(output_size))
        self.register_buffer('y_std', torch.ones(output_size))

    def forward(self, x):
        if x.shape[-1] != self.x_mean.shape[0]:
            raise ValueError(f"Input dim mismatch: expected {self.x_mean.shape[0]}, got {x.shape[-1]}")
        
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.model(x_norm)
        
        return y_norm * self.y_std + self.y_mean

def train(csv_path, epochs=100, batch_size=64, hidden_size=32, hidden_layers=1):
    dataset = SimpleDroneDataset(csv_path, ground_effect=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = dataset.x.shape[1]   
    output_size = dataset.y.shape[1]  
    model = NormalizedMLP(input_size, hidden_size, output_size, hidden_layers)

    
    model.x_mean.data = torch.from_numpy(dataset.x_scaler.mean_.astype(np.float32))
    model.x_std.data = torch.from_numpy(dataset.x_scaler.scale_.astype(np.float32))
    model.y_mean.data = torch.from_numpy(dataset.y_scaler.mean_.astype(np.float32))
    model.y_std.data = torch.from_numpy(dataset.y_scaler.scale_.astype(np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()print(f"Epoch {ep+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    
    torch.save({
        'config': {
            'ground_effect': True,
            'u_inp': True,
            'input_structure': '13-state+4-ctrl+patch(13)=30'
        },
        'state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'hidden_layers': hidden_layers
    }, 'results/model_fitting/9137159/new/drag__motor_noise__noisy__no_payload.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=1)
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch,
        hidden_size=args.hidden,
        hidden_layers=args.layers
    )
