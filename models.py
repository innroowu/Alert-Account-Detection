import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Sampler
from torch_sparse import SparseTensor
from typing import Iterator, Sized



def set_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiViewAttention(nn.Module):
    """Attention module for weighted fusion of multiple graph views."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 16, activation_fn: str = 'softmax'):
        super().__init__()
        self.activation_fn = activation_fn
        
        if hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        else:
            self.projection = nn.Sequential(nn.Linear(input_dim, 1, bias=False))
    
    def forward(self, embeddings: torch.Tensor) -> tuple:
        attention_scores = self.projection(embeddings)
        
        if self.activation_fn == 'softmax':
            attention_weights = torch.softmax(attention_scores, dim=1)
        elif self.activation_fn == 'tanh':
            attention_weights = torch.tanh(attention_scores)
        
        aggregated = (attention_weights * embeddings).sum(dim=1)
        return aggregated, attention_weights


class DualCATAConv(MessagePassing):
    """
    Dual-view Context-Aware Transformation Aggregation convolution.
    Captures differential features via three views: self, incoming, outgoing.
    """
    def __init__(self, in_dim: int, out_dim: int, attention_hidden: int = 16,
                 activation_type: str = 'softmax', aggregation_mode: str = 'add',
                 use_bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', aggregation_mode)
        super().__init__(**kwargs)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.transform_self = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.transform_context = nn.Linear(in_dim * 2, out_dim, bias=use_bias)
        self.view_fusion = MultiViewAttention(out_dim, attention_hidden, activation_type)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.transform_self.reset_parameters()
        self.transform_context.reset_parameters()
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        source_nodes, target_nodes = edge_index
        reversed_edges = torch.stack([target_nodes, source_nodes], dim=0)
        
        # Three views
        self_view = self.transform_self(node_features)
        outgoing_view = self.transform_context(self.propagate(edge_index, x=node_features))
        incoming_view = self.transform_context(self.propagate(reversed_edges, x=node_features))
        
        # Stack and fuse views
        multi_view = torch.stack([self_view, outgoing_view, incoming_view], dim=1)
        output, attention = self.view_fusion(multi_view)
        
        return output, attention
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute differential features for message passing."""
        difference = x_i - x_j
        return torch.cat([difference, x_j], dim=-1)


class DualCATANet(nn.Module):
    """Multi-layer DualCATA network with residual connections and layer fusion."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4,
                 attention_hidden: int = 16, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            in_channels = input_dim if layer_idx == 0 else hidden_dim
            self.conv_layers.append(
                DualCATAConv(in_channels, hidden_dim, attention_hidden=attention_hidden)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        self.layer_fusion_weights = nn.Parameter(torch.ones(num_layers))
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        current_features = node_features
        
        for layer_idx, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            hidden, _ = conv(current_features, edge_index)
            hidden = F.relu(norm(hidden))
            hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
            
            # Residual connection (skip first layer)
            if layer_idx > 0:
                current_features = hidden + current_features
            else:
                current_features = hidden
            
            layer_outputs.append(current_features)
        
        # Weighted layer fusion
        fusion_weights = F.softmax(self.layer_fusion_weights, dim=0)
        fused_features = sum(w * layer_out for w, layer_out in zip(fusion_weights, layer_outputs))
        
        return self.classifier(fused_features)


class MGD(nn.Module):
    """
    Multi-view Graph Discrepancy (MGD) model from DIAM paper.
    Uses DualCATAConv layers with attention-based view fusion and JumpingKnowledge aggregation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2,
                 attention_hidden: int = 16, dropout_rate: float = 0.2, 
                 gnn_norm: str = 'bn', aggr: str = 'add'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.gnn_norm = gnn_norm
        
        # Graph convolution layers 
        self.encoder = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = input_dim if layer_idx == 0 else hidden_dim
            self.encoder.append(
                DualCATAConv(
                    in_channels, 
                    hidden_dim, 
                    attention_hidden=attention_hidden,
                    activation_type='tanh',  # tanh activation for attention
                    aggregation_mode=aggr,
                    use_bias=True
                )
            )
        
        # Normalization layers 
        self.bns = nn.ModuleList()
        if gnn_norm == 'ln':
            for _ in range(num_layers):
                self.bns.append(nn.LayerNorm(hidden_dim))
        elif gnn_norm == 'bn':
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        
        x = node_features
        layer_outputs = []
        first_att = None
        
        # Graph convolution layers
        for i, conv in enumerate(self.encoder):
            x, att = conv(x, edge_index)
            
            # Store first layer attention for analysis
            if i == 0:
                first_att = att.clone().detach()
            
            # Normalization
            if self.gnn_norm != 'none':
                x = self.bns[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            layer_outputs.append(x)
        
        # Use last layer output
        gnn_emb = layer_outputs[-1]
        
        # Decoder (MLP)
        x = gnn_emb
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            if i != len(self.decoder) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.log_softmax(x, dim=1)
        
        return x, first_att


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4,
                 aggregation_fn: str = 'mean', dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.aggregation_fn = aggregation_fn
        
        self.sage_convs = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            in_channels = input_dim if layer_idx == 0 else hidden_dim
            self.sage_convs.append(
                SAGEConv(in_channels, hidden_dim, normalize=True, project=True, aggr=self.aggregation_fn)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        self.layer_fusion_weights = nn.Parameter(torch.ones(num_layers))
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        current_features = node_features
        
        for conv, norm in zip(self.sage_convs, self.norm_layers):
            current_features = conv(current_features, edge_index)
            current_features = norm(current_features)
            current_features = F.relu(current_features)
            current_features = F.dropout(current_features, p=self.dropout_rate, training=self.training)
            layer_outputs.append(current_features)
        
        # Weighted fusion of all layers
        fusion_weights = F.softmax(self.layer_fusion_weights, dim=0)
        fused_features = sum(w * layer_out for w, layer_out in zip(fusion_weights, layer_outputs))
        
        return self.classifier(fused_features)



class PU_Loss(nn.Module):
    def __init__(self, prior_probability: float):
        super().__init__()
        self.alpha = prior_probability
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get probability of positive class
        prob_positive = F.softmax(logits, dim=1)[:, 1]
        
        epsilon = 1e-6
        loss_positive = -torch.log(prob_positive + epsilon)
        loss_negative = -torch.log(1 - prob_positive + epsilon)
        
        is_labeled_positive = (targets == 1)
        is_unlabeled = (targets == 0)
        
        total_loss = torch.tensor(0.0, device=logits.device)
        
        # Labeled positive loss
        if is_labeled_positive.sum() > 0:
            total_loss = total_loss + loss_positive[is_labeled_positive].mean()
        
        # Unlabeled loss
        if is_unlabeled.sum() > 0:
            unlabeled_loss = (self.alpha * loss_positive[is_unlabeled] + 
                             (1 - self.alpha) * loss_negative[is_unlabeled]).mean()
            total_loss = total_loss + unlabeled_loss
        
        return total_loss
    
    
class nnPU_Loss(nn.Module):
    def __init__(self, prior_probability: float, beta=0.0, gamma=1.0):
        super().__init__()
        self.pi = prior_probability 
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 使用 sigmoid 取得機率
        probs = torch.sigmoid(logits[:, 1] - logits[:, 0]) # 針對 output_dim=2
        
        pos_mask = (targets == 1).float()
        unlabeled_mask = (targets == 0).float()
        
        # Positive loss
        loss_pos = -torch.log(probs + 1e-7)
        # Negative loss
        loss_neg = -torch.log(1 - probs + 1e-7)
        
        n_pos = pos_mask.sum()
        n_unlabeled = unlabeled_mask.sum()
        
        if n_pos == 0 or n_unlabeled == 0:
            return F.cross_entropy(logits, targets)

        # PU Risk Estimation
        risk_pos = (pos_mask * loss_pos).sum() / n_pos
        risk_unlabeled_neg = (unlabeled_mask * loss_neg).sum() / n_unlabeled
        risk_pos_neg = (pos_mask * loss_neg).sum() / n_pos
        
        # R_pu = pi * R_p+ + max(0, R_u- - pi * R_p-)
        risk_neg = risk_unlabeled_neg - self.pi * risk_pos_neg
        
        if risk_neg < self.beta:
            return self.gamma * risk_pos
        
        return self.pi * risk_pos + risk_neg


class MGD_SeqOnly(nn.Module):

    def __init__(self, seq_feat_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, attention_hidden: int = 16,
                 dropout_rate: float = 0.2, gnn_norm: str = 'bn', aggr: str = 'add',
                 rnn_type: str = 'gru'):
        super().__init__()

        self.seq_feat_dim = seq_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.gnn_norm = gnn_norm
        self.rnn_type = rnn_type

        rnn_out_dim = hidden_dim // 2  # Each RNN outputs hidden_dim // 2

        # ===== RNN Encoders =====
        if rnn_type == 'gru':
            self.rnn_in = nn.GRU(seq_feat_dim, rnn_out_dim, batch_first=True)
            self.rnn_out = nn.GRU(seq_feat_dim, rnn_out_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn_in = nn.LSTM(seq_feat_dim, rnn_out_dim, batch_first=True)
            self.rnn_out = nn.LSTM(seq_feat_dim, rnn_out_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # RNN output normalization
        self.rnn_norm_in = nn.LayerNorm(rnn_out_dim)
        self.rnn_norm_out = nn.LayerNorm(rnn_out_dim)

        # ===== GNN Encoder (DualCATAConv) =====
        self.encoder = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = hidden_dim if layer_idx == 0 else hidden_dim
            self.encoder.append(
                DualCATAConv(
                    in_channels,
                    hidden_dim,
                    attention_hidden=attention_hidden,
                    activation_type='tanh',
                    aggregation_mode=aggr,
                    use_bias=True
                )
            )

        # Normalization layers
        self.bns = nn.ModuleList()
        if gnn_norm == 'ln':
            for _ in range(num_layers):
                self.bns.append(nn.LayerNorm(hidden_dim))
        elif gnn_norm == 'bn':
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ===== Decoder (MLP) =====
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder.append(nn.Linear(hidden_dim, output_dim))

    def encode_sequences(self, sequences: torch.Tensor, lengths: torch.Tensor,
                        rnn: nn.Module, norm: nn.Module) -> torch.Tensor:
        """Encode variable-length sequences using RNN."""

        # Pack sequences
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            sequences, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # RNN encoding
        if self.rnn_type == 'gru':
            _, h_n = rnn(packed)  # h_n: (1, batch, hidden)
            h = h_n.squeeze(0)
        else:  # LSTM
            _, (h_n, _) = rnn(packed)
            h = h_n.squeeze(0)

        # Normalize
        h = norm(h)

        return h

    def forward(self, in_sequences: torch.Tensor, out_sequences: torch.Tensor,
                lengths_in: torch.Tensor, lengths_out: torch.Tensor,
                edge_index: torch.Tensor) -> tuple:
        """
        Forward pass using ONLY sequence features.
        """
        # Step 1: Encode sequences with RNN
        h_in = self.encode_sequences(in_sequences, lengths_in, self.rnn_in, self.rnn_norm_in)
        h_out = self.encode_sequences(out_sequences, lengths_out, self.rnn_out, self.rnn_norm_out)
        x = torch.cat([h_in, h_out], dim=1)  # (N, hidden_dim)

        # Step 2: GNN encoding
        first_att = None
        for layer_idx, (conv, norm) in enumerate(zip(self.encoder, self.bns)):
            hidden, att = conv(x, edge_index)

            if layer_idx == 0:
                first_att = att.clone().detach()

            if self.gnn_norm != 'none':
                hidden = norm(hidden)

            hidden = F.relu(hidden)
            x = F.dropout(hidden, p=self.dropout_rate, training=self.training)

        # Step 3: Decode
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(self.decoder) - 1:  # Not last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Log softmax
        logits = F.log_softmax(x, dim=1)

        return logits, first_att


class MGD_Hybrid(nn.Module):
    # Hybrid MGD model combining sequence features (RNN) and statistical features.
    def __init__(self, stat_feat_dim: int, seq_feat_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int = 2, attention_hidden: int = 16,
                 dropout_rate: float = 0.2, gnn_norm: str = 'bn', aggr: str = 'add',
                 rnn_type: str = 'gru'):
        super().__init__()
        
        self.stat_feat_dim = stat_feat_dim
        self.seq_feat_dim = seq_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.gnn_norm = gnn_norm
        self.rnn_type = rnn_type
        
        rnn_out_dim = hidden_dim // 4  # 32-dim for each RNN
        
        # ===== RNN Encoders =====
        if rnn_type == 'gru':
            self.rnn_in = nn.GRU(seq_feat_dim, rnn_out_dim, batch_first=True)
            self.rnn_out = nn.GRU(seq_feat_dim, rnn_out_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn_in = nn.LSTM(seq_feat_dim, rnn_out_dim, batch_first=True)
            self.rnn_out = nn.LSTM(seq_feat_dim, rnn_out_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # RNN output normalization
        self.rnn_norm_in = nn.LayerNorm(rnn_out_dim)
        self.rnn_norm_out = nn.LayerNorm(rnn_out_dim)
        
        # ===== Statistical Feature Projection =====
        # Project to hidden_dim // 2 to balance with RNN features (2 * rnn_out_dim)
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ===== GNN Encoder (DualCATAConv) =====
        self.encoder = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = hidden_dim if layer_idx == 0 else hidden_dim
            self.encoder.append(
                DualCATAConv(
                    in_channels,
                    hidden_dim,
                    attention_hidden=attention_hidden,
                    activation_type='tanh',
                    aggregation_mode=aggr,
                    use_bias=True
                )
            )
        
        # Normalization layers
        self.bns = nn.ModuleList()
        if gnn_norm == 'ln':
            for _ in range(num_layers):
                self.bns.append(nn.LayerNorm(hidden_dim))
        elif gnn_norm == 'bn':
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # ===== Decoder (MLP) =====
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder.append(nn.Linear(hidden_dim, output_dim))
    
    def encode_sequences(self, sequences: torch.Tensor, lengths: torch.Tensor, 
                        rnn: nn.Module, norm: nn.Module) -> torch.Tensor:
        # Encode variable-length sequences using RNN.
        
        # Pack sequences
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            sequences, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # RNN encoding
        if self.rnn_type == 'gru':
            _, h_n = rnn(packed)  # h_n: (1, batch, hidden)
            h = h_n.squeeze(0)
        else:  # LSTM
            _, (h_n, _) = rnn(packed)
            h = h_n.squeeze(0)
        
        # Normalize
        h = norm(h)
        
        return h
    
    def forward(self, stat_features: torch.Tensor, in_sequences: torch.Tensor,
                out_sequences: torch.Tensor, lengths_in: torch.Tensor,
                lengths_out: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        Forward pass combining statistical and sequence features.
        """
        # Step 1: Encode sequences with RNN
        h_in = self.encode_sequences(in_sequences, lengths_in, self.rnn_in, self.rnn_norm_in)
        h_out = self.encode_sequences(out_sequences, lengths_out, self.rnn_out, self.rnn_norm_out)
        h_rnn = torch.cat([h_in, h_out], dim=1)  # (N, hidden_dim // 2)
        
        # Step 2: Project statistical features
        h_stat = self.stat_proj(stat_features)  # (N, hidden_dim // 2)
        
        # Step 3: Fuse features
        x = torch.cat([h_rnn, h_stat], dim=1)  # (N, hidden_dim)
        
        # Step 4: GNN encoding
        first_att = None
        for layer_idx, (conv, norm) in enumerate(zip(self.encoder, self.bns)):
            hidden, att = conv(x, edge_index)
            
            if layer_idx == 0:
                first_att = att.clone().detach()
            
            if self.gnn_norm != 'none':
                hidden = norm(hidden)
            
            hidden = F.relu(hidden)
            x = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        
        # Step 5: Decode
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(self.decoder) - 1:  # Not last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Log softmax
        logits = F.log_softmax(x, dim=1)
        
        return logits, first_att


# ==================== Mini-batch Sampling Utilities ====================

class BalancedSampler(Sampler[int]):
    """
    Balanced sampler for imbalanced datasets.
    Samples equal numbers of positive and negative samples in each epoch.
    Only for BCELoss, will not use when using PULoss.
    """
    
    def __init__(self, labels: torch.Tensor, generator=None):
        self.pos_index = (labels == 1).nonzero().reshape(-1)
        self.neg_index = (labels == 0).nonzero().reshape(-1)
        self.n_pos = len(self.pos_index)
        self._num_samples = 2 * self.n_pos
        self.generator = generator
    
    @property
    def num_samples(self) -> int:
        return self._num_samples
    
    def __iter__(self) -> Iterator[int]:
        n = 2 * self.n_pos
        
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            neg_seed = int(torch.empty((), dtype=torch.int64).random_().item())
            neg_generator = torch.Generator()
            neg_generator.manual_seed(neg_seed)
            
            # Sample equal number of negatives as positives
            chosen_neg = self.neg_index[torch.randperm(len(self.neg_index), generator=neg_generator)[:self.n_pos]]
            data_source = torch.cat((self.pos_index, chosen_neg), -1)
        else:
            generator = self.generator
            neg_generator = self.generator
            chosen_neg = self.neg_index[torch.randperm(len(self.neg_index), generator=neg_generator)[:self.n_pos]]
            data_source = torch.cat((self.pos_index, chosen_neg), -1)
        
        # Shuffle combined positive and negative samples
        for _ in range(self.num_samples // n):
            yield from data_source[torch.randperm(n, generator=generator)].tolist()
        yield from data_source[torch.randperm(n, generator=generator)].tolist()[:self.num_samples % n]
    
    def __len__(self) -> int:
        return self.num_samples


class DualNeighborSampler(torch.utils.data.DataLoader):
    """
    (DIAM) Sample edges from both directions.
    """
    def __init__(self, edge_index: torch.Tensor,
                 sizes: list, node_idx: torch.Tensor = None,
                 num_nodes: int = None, return_e_id: bool = True,
                 transform=None, sampler=None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        src, dst = edge_index
        edge_index_inverse = torch.vstack((dst, src))
        edge_index = torch.vstack((src, dst))
        edge_index_non_inverse = torch.cat((edge_index, edge_index), -1)
        edge_index_inverse = torch.cat((edge_index, edge_index_inverse), -1)
        self.edge_index = edge_index_non_inverse
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1
            self.num_nodes = num_nodes
            value = torch.arange(edge_index_inverse.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index_inverse[0], col=edge_index_inverse[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index_inverse
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        # Handle sampler
        if sampler is not None:
            kwargs['sampler'] = sampler
            kwargs['shuffle'] = False

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append((adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append((edge_index, e_id, size))

        adjs = adjs if len(adjs) == 1 else adjs[::-1]
        eids = adjs[0][1]
        edge_index = self.edge_index[:, eids]
        node_idx = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx[n_id] = torch.arange(n_id.size(0))
        edge_index = node_idx[edge_index]
        return edge_index, n_id, batch_size

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'