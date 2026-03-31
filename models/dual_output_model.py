import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualOutputMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(DualOutputMolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        # 修改为输出两个值：负RTI和正RTI
        self.W_property = nn.Linear(dim, 2)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = matrices[0].device if matrices else torch.device('cpu')
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,fingerprints, adjacencies, molecular_sizes = inputs[0:4]
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        
        # 检查维度一致性
        total_molecular_size = sum(molecular_sizes)
        if adjacencies.shape[0] != adjacencies.shape[1]:
            raise ValueError(f"Adjacency matrix is not square: {adjacencies.shape}")
        if adjacencies.shape[0] != total_molecular_size:
            print("Dimension mismatch detected!")
            print(f"Expected total size: {total_molecular_size}, Adjacency matrix size: {adjacencies.shape[0]}")
            cumulative_size = 0
            for i, (smi, size) in enumerate(zip(Smiles, molecular_sizes)):
                print(f"Molecule {i+1} SMILES: {smi}, Expected size: {size}")
                cumulative_size += size
            raise ValueError("Adjacency matrix size doesn't match sum of molecular sizes")
            
        if fingerprints.shape[0] != total_molecular_size:
            raise ValueError(f"Fingerprints count ({fingerprints.shape[0]}) doesn't match total molecular size ({total_molecular_size})")

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(len(self.W_fingerprint)):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return Smiles, molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(len(self.W_output)):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_regressor(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])  # 应该是形状为 (batch_size, 2) 的张量

        if train:
            Smiles, molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                Smiles, molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            # 返回形状为 (batch_size, 2) 的数组
            return Smiles, predicted_values, correct_values