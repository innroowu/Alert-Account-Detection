import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler

from models import *
from preprocessing import *


def create_model(model_type: str, input_dim: int, hidden_dim: int, output_dim: int,
                num_layers: int = 4, dropout: float = 0.3, feature_mode: str = 'statistical',
                seq_feat_dim: int = 3, **kwargs) -> nn.Module:

    model_type = model_type.lower()

    if model_type == 'mgd':
        attention_hidden = kwargs.get('atten_hidden', 16)
        gnn_norm = kwargs.get('gnn_norm', 'bn')
        aggr = kwargs.get('aggr', 'add')

        if feature_mode == 'seq_and_stat':
            # Hybrid model: Statistical + Sequence features
            return MGD_Hybrid(
                stat_feat_dim=input_dim,
                seq_feat_dim=seq_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                attention_hidden=attention_hidden,
                dropout_rate=dropout,
                gnn_norm=gnn_norm,
                aggr=aggr,
                rnn_type='gru'
            )
        elif feature_mode == 'sequence_only':
            # Sequence-only model: RNN features ONLY (no statistical features)
            return MGD_SeqOnly(
                seq_feat_dim=seq_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                attention_hidden=attention_hidden,
                dropout_rate=dropout,
                gnn_norm=gnn_norm,
                aggr=aggr,
                rnn_type='gru'
            )
        else:
            # Original MGD: Statistical features only
            return MGD(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                attention_hidden=attention_hidden,
                dropout_rate=dropout,
                gnn_norm=gnn_norm,
                aggr=aggr
            )
    
    elif model_type == 'graphsage':
        if feature_mode in ['seq_and_stat', 'sequence_only']:
            raise ValueError("GraphSAGE does not support sequence features. Use --feature statistical")
        
        aggregation = kwargs.get('aggr', 'mean')
        return GraphSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            aggregation_fn=aggregation,
            dropout_rate=dropout
        )
    
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Choose from ['mgd', 'graphsage']"
        )



class MGDSeqOnlyTrainer:
    """
    Trainer for MGD model with sequence-only features (no statistical features).
    Uses only RNN-encoded transaction sequences.
    """

    def __init__(self, model, data, in_sequences, out_sequences, seq_lengths,
                 criterion, optimizer, device, batch_size=128, neighbor_sizes=[25, 10],
                 use_balanced_sampling=False):
        self.model = model
        self.data = data
        self.in_sequences = in_sequences
        self.out_sequences = out_sequences
        self.seq_lengths = seq_lengths
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.neighbor_sizes = neighbor_sizes
        self.use_balanced_sampling = use_balanced_sampling
        self.best_val_auprc = 0.0
        self.checkpoint_path = 'best_mgd_seqonly_model.pt'

        # Early stopping
        self.patience = 30
        self.patience_counter = 0

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )

        # Create data loaders
        self._setup_loaders()

    def _setup_loaders(self):
        """Setup train/val/test data loaders with neighborhood sampling."""
        from models import BalancedSampler, DualNeighborSampler

        # Get node indices for each split
        train_nid = torch.nonzero(self.data.train_mask, as_tuple=True)[0]
        val_nid = torch.nonzero(self.data.val_mask, as_tuple=True)[0]
        test_nid = torch.nonzero(self.data.test_mask, as_tuple=True)[0]

        # Training loader with optional balanced sampling
        if self.use_balanced_sampling:
            train_labels = self.data.y[train_nid]
            sampler = BalancedSampler(train_labels)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.train_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=train_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle
        )

        self.val_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=val_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=test_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )

    def train_epoch(self):
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for edge_index, node_ids, batch_size in self.train_loader:
            edge_index = edge_index.to(self.device)

            # Get sequence features ONLY (no statistical features)
            in_seq = self.in_sequences[node_ids].to(self.device)
            out_seq = self.out_sequences[node_ids].to(self.device)
            lengths_in = self.seq_lengths['in'][node_ids]
            lengths_out = self.seq_lengths['out'][node_ids]

            # Get labels
            labels = self.data.y[node_ids].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(in_seq, out_seq, lengths_in, lengths_out, edge_index)

            # Compute loss only on batch nodes
            loss = self.criterion(logits[:batch_size], labels[:batch_size])

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, loader):
        """Evaluate on given data loader."""
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for edge_index, node_ids, batch_size in loader:
                edge_index = edge_index.to(self.device)

                # Get sequence features ONLY
                in_seq = self.in_sequences[node_ids].to(self.device)
                out_seq = self.out_sequences[node_ids].to(self.device)
                lengths_in = self.seq_lengths['in'][node_ids]
                lengths_out = self.seq_lengths['out'][node_ids]

                # Forward
                logits, _ = self.model(in_seq, out_seq, lengths_in, lengths_out, edge_index)

                # Get probabilities
                probs = torch.exp(logits[:batch_size, 1]).cpu().numpy()
                labels = self.data.y[node_ids[:batch_size]].cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels)

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        return {
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'probs': all_probs,
            'labels': all_labels
        }

    def train(self, num_epochs=200):
        """Execute complete training loop."""
        print(f'\nTraining MGD Sequence-Only model for {num_epochs} epochs...\n')

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # Update scheduler
            self.scheduler.step(val_metrics['auprc'])

            # Print progress
            print(f'Epoch {epoch:4d} | Loss: {train_loss:.4f} | '
                  f'Val AUPRC: {val_metrics["auprc"]:.4f} | '
                  f'Val F1: {val_metrics["f1"]:.4f}')

            # Save best model
            if val_metrics['auprc'] > self.best_val_auprc:
                self.best_val_auprc = val_metrics['auprc']
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'  *** New best: {self.best_val_auprc:.4f} (saved) ***')
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

            # Memory cleanup
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f'\nTraining complete! Best Val AUPRC: {self.best_val_auprc:.4f}')


class MGDHybridTrainer:
    """
    Trainer for MGD model with hybrid features (statistical + sequence).
    Similar to MGDTrainer but handles sequence data in addition to statistical features.
    """

    def __init__(self, model, data, in_sequences, out_sequences, seq_lengths,
                 criterion, optimizer, device, batch_size=128, neighbor_sizes=[25, 10],
                 use_balanced_sampling=False):
        self.model = model
        self.data = data
        self.in_sequences = in_sequences
        self.out_sequences = out_sequences
        self.seq_lengths = seq_lengths
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.neighbor_sizes = neighbor_sizes
        self.use_balanced_sampling = use_balanced_sampling
        self.best_val_auprc = 0.0
        self.checkpoint_path = 'best_mgd_hybrid_model.pt'
        
        # Early stopping
        self.patience = 30
        self.patience_counter = 0
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        # Create data loaders
        self._setup_loaders()
    
    def _setup_loaders(self):
        """Setup train/val/test data loaders with neighborhood sampling."""
        from models import BalancedSampler, DualNeighborSampler
        
        # Get node indices for each split
        train_nid = torch.nonzero(self.data.train_mask, as_tuple=True)[0]
        val_nid = torch.nonzero(self.data.val_mask, as_tuple=True)[0]
        test_nid = torch.nonzero(self.data.test_mask, as_tuple=True)[0]
        
        # Training loader with optional balanced sampling
        if self.use_balanced_sampling:
            train_labels = self.data.y[train_nid]
            sampler = BalancedSampler(train_labels)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        self.train_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=train_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle
        )
        
        self.val_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=val_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.test_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=test_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def train_epoch(self):
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for edge_index, node_ids, batch_size in self.train_loader:
            edge_index = edge_index.to(self.device)
            
            # Get statistical features
            stat_feat = self.data.x[node_ids].to(self.device)
            
            # Get sequence features
            in_seq = self.in_sequences[node_ids].to(self.device)
            out_seq = self.out_sequences[node_ids].to(self.device)
            lengths_in = self.seq_lengths['in'][node_ids]
            lengths_out = self.seq_lengths['out'][node_ids]
            
            # Get labels
            labels = self.data.y[node_ids].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(stat_feat, in_seq, out_seq, lengths_in, lengths_out, edge_index)
            
            # Compute loss only on batch nodes
            loss = self.criterion(logits[:batch_size], labels[:batch_size])
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, loader):
        """Evaluate on given data loader."""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for edge_index, node_ids, batch_size in loader:
                edge_index = edge_index.to(self.device)
                
                # Get features
                stat_feat = self.data.x[node_ids].to(self.device)
                in_seq = self.in_sequences[node_ids].to(self.device)
                out_seq = self.out_sequences[node_ids].to(self.device)
                lengths_in = self.seq_lengths['in'][node_ids]
                lengths_out = self.seq_lengths['out'][node_ids]
                
                # Forward
                logits, _ = self.model(stat_feat, in_seq, out_seq, lengths_in, lengths_out, edge_index)
                
                # Get probabilities
                probs = torch.exp(logits[:batch_size, 1]).cpu().numpy()
                labels = self.data.y[node_ids[:batch_size]].cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels)
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        return {
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'probs': all_probs,
            'labels': all_labels
        }
    
    def train(self, num_epochs=200):
        """Execute complete training loop."""
        print(f'\nTraining MGD Hybrid model for {num_epochs} epochs...\n')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['auprc'])
            
            # Print progress
            print(f'Epoch {epoch:4d} | Loss: {train_loss:.4f} | '
                  f'Val AUPRC: {val_metrics["auprc"]:.4f} | '
                  f'Val F1: {val_metrics["f1"]:.4f}')
            
            # Save best model
            if val_metrics['auprc'] > self.best_val_auprc:
                self.best_val_auprc = val_metrics['auprc']
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'  *** New best: {self.best_val_auprc:.4f} (saved) ***')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break
            
            # Memory cleanup
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f'\nTraining complete! Best Val AUPRC: {self.best_val_auprc:.4f}')


# ==================== Training Loop ====================

class Trainer:
    """Handles model training with mixed precision and validation."""
    
    def __init__(self, model, compiled_model, data, criterion, optimizer, scaler, 
                 device, eval_threshold=0.95):
        self.model = model
        self.compiled_model = compiled_model
        self.data = data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.eval_threshold = eval_threshold
        self.best_auprc = 0.0
        self.checkpoint_path = 'best_model.pt'
    
    def train_epoch(self):
        """Execute one training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            logits = self.compiled_model(self.data.x, self.data.edge_index)
            loss = self.criterion(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def validate(self):
        """Perform validation and return metrics."""
        self.model.eval()
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                logits = self.model(self.data.x, self.data.edge_index)
                val_loss = self.criterion(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            
            probs = F.softmax(logits[self.data.val_mask].float(), dim=1)[:, 1].cpu().numpy()
            labels = self.data.y[self.data.val_mask].cpu().numpy()
            
            auprc = average_precision_score(labels, probs)
            preds = (probs >= self.eval_threshold).astype(int)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
        
        return {'auprc': auprc, 'val_loss': val_loss.item(),
            'precision': precision, 'recall': recall, 'f1': f1
        }
    
    def train(self, num_epochs=1000):
        """Execute complete training loop."""
        print(f'\nTraining for {num_epochs} epochs...\n')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            
            metrics = self.validate()
                
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {metrics["val_loss"]:.4f} | Val AUPRC: {metrics["auprc"]:.4f}')
            
            if metrics['auprc'] > self.best_auprc:
                self.best_auprc = metrics['auprc']
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'  *** New best: {self.best_auprc:.4f} (saved) ***')
            
            if epoch % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f'\nTraining complete! Best AUPRC: {self.best_auprc:.4f}')


class MGDTrainer:
    """
    Mini-batch trainer for MGD model (following DIAM's exact training procedure).
    Uses neighborhood sampling and optional balanced sampling for handling class imbalance.
    """
    
    def __init__(self, model, data, criterion, optimizer, device,
                 batch_size=128, neighbor_sizes=[25, 10], use_balanced_sampling=False):
        self.model = model
        self.data = data
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.neighbor_sizes = neighbor_sizes
        self.use_balanced_sampling = use_balanced_sampling
        self.best_val_auprc = 0.0
        self.checkpoint_path = 'best_mgd_model.pt'
        
        # Add early stopping
        self.patience = 30
        self.patience_counter = 0
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=True
        )
        
        # Create data loaders
        self._setup_loaders()
    
    def _setup_loaders(self):
        """Setup train/val/test data loaders with neighborhood sampling."""
        from models import BalancedSampler, DualNeighborSampler
        
        # Get node indices for each split
        train_nid = torch.nonzero(self.data.train_mask, as_tuple=True)[0]
        val_nid = torch.nonzero(self.data.val_mask, as_tuple=True)[0]
        test_nid = torch.nonzero(self.data.test_mask, as_tuple=True)[0]
        
        # Training loader with optional balanced sampling
        if self.use_balanced_sampling:
            train_labels = self.data.y[train_nid]
            sampler = BalancedSampler(train_labels)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        self.train_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=train_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle
        )
        
        # Validation loader
        self.val_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=val_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Test loader
        self.test_loader = DualNeighborSampler(
            edge_index=self.data.edge_index,
            sizes=self.neighbor_sizes,
            node_idx=test_nid,
            num_nodes=self.data.x.shape[0],
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def train_epoch(self):
        """Execute one training epoch with mini-batches."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Compute class weights (only once)
        if not hasattr(self, 'class_weights'):
            pos_count = (self.data.y[self.data.train_mask] == 1).sum().item()
            neg_count = (self.data.y[self.data.train_mask] == 0).sum().item()
            total = pos_count + neg_count
            self.class_weights = torch.FloatTensor([
                total / (2 * neg_count),  # weight for negative class
                total / (2 * pos_count)   # weight for positive class
            ]).to(self.device)
            print(f'\nClass weights: neg={self.class_weights[0]:.4f}, pos={self.class_weights[1]:.4f}')
        
        for batch_idx, (edge_index, n_id, batch_size) in enumerate(self.train_loader):
            batch_x = self.data.x[n_id].to(self.device)
            batch_y = self.data.y[n_id[:batch_size]].to(self.device)
            edge_index = edge_index.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_x, edge_index)
            
            # Use weighted NLLLoss
            loss = F.nll_loss(logits[:batch_size], batch_y, weight=self.class_weights)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, loader):
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for edge_index, n_id, batch_size in loader:
                batch_x = self.data.x[n_id].to(self.device)
                batch_y_device = self.data.y[n_id[:batch_size]].to(self.device)
                batch_y_cpu = batch_y_device.cpu()
                edge_index = edge_index.to(self.device)
                
                logits, _ = self.model(batch_x, edge_index)
                
                # Compute loss for this batch
                if hasattr(self, 'class_weights'):
                    loss = F.nll_loss(logits[:batch_size], batch_y_device, weight=self.class_weights)
                else:
                    loss = F.nll_loss(logits[:batch_size], batch_y_device)
                total_loss += loss.item()
                num_batches += 1
                
                preds = logits[:batch_size].cpu()
                all_preds.append(preds)
                all_labels.append(batch_y_cpu)
        
        # Average validation loss
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        probs = F.softmax(all_preds, dim=1)[:, 1].numpy()
        labels = all_labels.numpy()
        
        auprc = average_precision_score(labels, probs)
        
        # Calculate metrics at 0.5 threshold (for consistency)
        preds_05 = (probs >= 0.5).astype(int)
        precision = precision_score(labels, preds_05, zero_division=0)
        recall = recall_score(labels, preds_05, zero_division=0)
        f1 = f1_score(labels, preds_05, zero_division=0)
        
        return {
            'auprc': auprc,
            'val_loss': avg_val_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'probs': probs,
            'labels': labels
        }
    
    def train(self, num_epochs=100):
        """Execute complete training loop."""
        print(f'\n{"="*80}')
        print(f'MGD Mini-batch Training')
        print(f'{"="*80}')
        print(f'Batch size: {self.batch_size}')
        print(f'Neighbor sizes: {self.neighbor_sizes}')
        print(f'Balanced sampling: {self.use_balanced_sampling}')
        print(f'{"="*80}\n')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            
            # Validate every epoch
            val_metrics = self.evaluate(self.val_loader)
            
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_metrics["val_loss"]:.4f} | '
                  f'Val AUPRC: {val_metrics["auprc"]:.4f}')
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auprc'])
            
            # Early stopping based on AUPRC
            if val_metrics['auprc'] > self.best_val_auprc:
                self.best_val_auprc = val_metrics['auprc']
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'  *** New best: {self.best_val_auprc:.4f} (saved) ***')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break
            
            if (epoch + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f'\nTraining complete! Best AUPRC: {self.best_val_auprc:.4f}')


def search_optimal_threshold(probs, labels, start=0.1, end=1.0, step=0.05):
    """Grid search for threshold that maximizes F1 score."""
    print('\nSearching optimal threshold:')
    print('-' * 60)
    
    thresholds = np.arange(start, end, step)
    best_f1, best_threshold = 0.0, 0.5
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        print(f'Threshold: {threshold:.2f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}')
        
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    
    print('-' * 60)
    print(f'Optimal: {best_threshold:.2f} with F1={best_f1:.4f}\n')
    
    return best_threshold, best_f1


def evaluate_and_predict_mgd(model, data, accounts, test_df, trainer, checkpoint_path='best_mgd_model.pt'):
    """
    Load best MGD model and generate predictions (matching GraphSAGE output format).
    """
    print(f'\n{"="*80}')
    print('Evaluation and Prediction (MGD)')
    print(f'{"="*80}')
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Get validation predictions
    val_metrics = trainer.evaluate(trainer.val_loader)
    val_probs = val_metrics['probs']
    val_labels = val_metrics['labels']
    
    # Search optimal threshold on validation set
    optimal_threshold, _ = search_optimal_threshold(val_probs, val_labels)
    
    # Apply optimal threshold to validation set
    val_preds = (val_probs >= optimal_threshold).astype(int)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    
    print(f'Performance with threshold {optimal_threshold:.2f}:')
    print(f'  Val F1: {val_f1:.4f}')
    
    # Get test predictions
    test_metrics = trainer.evaluate(trainer.test_loader)
    test_probs = test_metrics['probs']
    
    # Apply optimal threshold
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Map predictions back to accounts
    test_mask_np = data.test_mask.cpu().numpy()
    test_accounts = np.array(accounts)[test_mask_np]
    
    predictions_df = pd.DataFrame({'acct': test_accounts, 'label': test_preds})
    output_df = test_df[['acct']].merge(predictions_df, on='acct', how='left')
    output_df['label'] = output_df['label'].fillna(0).astype(int)
    
    print(f'\nTest predictions: {len(output_df)} total, {output_df["label"].sum()} positive')
    
    return output_df


def evaluate_and_predict(model, data, accounts, test_df, checkpoint_path='best_model.pt'):
    """Load best model and generate predictions (for GraphSAGE)."""
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        all_probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # Find optimal threshold on validation set
    val_mask_np = data.val_mask.cpu().numpy()
    val_probs = all_probs[val_mask_np]
    val_labels = data.y[val_mask_np].cpu().numpy()
    
    optimal_threshold, _ = search_optimal_threshold(val_probs, val_labels)
    
    # Apply threshold
    all_preds = (all_probs >= optimal_threshold).astype(int)
    
    # Report performance
    train_mask_np = data.train_mask.cpu().numpy()
    train_f1 = f1_score(data.y[train_mask_np].cpu().numpy(), all_preds[train_mask_np], zero_division=0)
    val_f1 = f1_score(val_labels, all_preds[val_mask_np], zero_division=0)
    
    print(f'Performance with threshold {optimal_threshold:.2f}:')
    print(f'  Train F1: {train_f1:.4f}')
    print(f'  Val F1:   {val_f1:.4f}')
    
    # Generate test predictions
    test_mask_np = data.test_mask.cpu().numpy()
    test_preds = all_preds[test_mask_np]
    test_accounts = np.array(accounts)[test_mask_np]
    
    predictions_df = pd.DataFrame({'acct': test_accounts, 'label': test_preds})
    output_df = test_df[['acct']].merge(predictions_df, on='acct', how='left')
    output_df['label'] = output_df['label'].fillna(0).astype(int)
    
    print(f'\nTest predictions: {len(output_df)} total, {output_df["label"].sum()} positive')
    
    return output_df


# ==================== Argument Parsing ====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GNN training for AML detection')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='/share/nas169/innroowu/top11/data')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--force_rebuild', action='store_true')
    
    # Model
    parser.add_argument('--model', type=str, default='mgd', choices=['mgd', 'graphsage'])
    parser.add_argument('--feature', type=str, default='statistical',
                       choices=['statistical', 'seq_and_stat', 'sequence_only'],
                       help='Feature type: statistical (stat only), seq_and_stat (stat+seq hybrid), or sequence_only (seq only) - MGD only')
    parser.add_argument('--max_seq_len', type=int, default=32,
                       help='Maximum sequence length (only for --feature seq_and_stat or sequence_only)')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--atten_hidden', type=int, default=16)
    parser.add_argument('--aggr', type=str, default='add', choices=['mean', 'max', 'add'])
    parser.add_argument('--gnn_norm', type=str, default='bn', choices=['bn', 'ln', 'none'])
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--threshold', type=float, default=0.5)
    
    # Mini-batch settings (for MGD)
    parser.add_argument('--use_minibatch', action='store_true', 
                       help='Use mini-batch training (required for MGD)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--neighbor_sizes', type=str, default='20,10',
                       help='Neighbor sampling sizes for each layer (comma-separated)')
    parser.add_argument('--balanced_sampling', action='store_true',
                       help='Use balanced sampling for imbalanced data')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='/share/nas169/innroowu/top11/results')
    parser.add_argument('--output_file', type=str, default='predictions.csv')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


# ==================== Main Pipeline ====================

def main():
    """Main training pipeline."""
    args = parse_args()
    
    print('=' * 80)
    print('AML Detection Training')
    print('=' * 80)
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Validate arguments
    if args.feature in ['seq_and_stat', 'sequence_only'] and args.model != 'mgd':
        raise ValueError("Sequence features are only supported with MGD model. Use --model mgd")

    # Force mini-batch training for MGD
    if args.model == 'mgd':
        args.use_minibatch = True
        print(f'\nMGD model requires mini-batch training - enabled automatically')

    # Determine if we need to build sequences
    build_sequences = (args.model == 'mgd' and args.feature in ['seq_and_stat', 'sequence_only'])
    
    # Prepare data
    cache_dir = args.cache_dir or os.path.join(args.data_dir, 'processed')
    
    print('\n' + '=' * 80)
    print('Data Preparation')
    print('=' * 80)
    print(f'Model: {args.model}')
    print(f'Feature mode: {args.feature}')
    
    result = prepare_graph_dataset(
        data_dir=args.data_dir,
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild,
        build_sequences=build_sequences,
        max_seq_len=args.max_seq_len
    )
    
    # Unpack results based on feature mode
    if build_sequences:
        (accounts, esun_accts, test_accts, node_features, labels, edges,
         train_mask, val_mask, test_mask, scaler, data,
         in_sequences, out_sequences, seq_lengths) = result
        seq_feat_dim = in_sequences.shape[-1]
        print(f'\nSequence features: {in_sequences.shape}')
        print(f'Sequence feature dim: {seq_feat_dim}')
    else:
        (accounts, esun_accts, test_accts, node_features, labels, edges,
         train_mask, val_mask, test_mask, scaler, data) = result
        in_sequences = None
        out_sequences = None
        seq_lengths = None
        seq_feat_dim = 0
    
    data = data.to(device)
    test_df = pd.read_csv(os.path.join(args.data_dir, 'acct_predict.csv'))
    
    print(f'\nGraph: {data.x.shape[0]:,} nodes, {edges.shape[1]:,} edges, {data.x.shape[1]} features')
    print(f'Split: {train_mask.sum():,} train, {val_mask.sum():,} val, {test_mask.sum():,} test')
    
    # Class distribution
    train_pos = (data.y[data.train_mask] == 1).sum().item()
    train_neg = (data.y[data.train_mask] == 0).sum().item()
    print(f'Training set: {train_pos} positive, {train_neg} negative (ratio: 1:{train_neg/train_pos:.1f})')
    
    # Initialize model
    print('\n' + '=' * 80)
    print(f'Model: {args.model.upper()}')
    if args.feature == 'seq_and_stat':
        print(f'Feature: Hybrid (Statistical + Sequence)')
    elif args.feature == 'sequence_only':
        print(f'Feature: Sequence only (RNN)')
    else:
        print(f'Feature: Statistical only')
    print('=' * 80)
    
    model = create_model(
        model_type=args.model,
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_channels,
        output_dim=2,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_mode=args.feature,
        seq_feat_dim=seq_feat_dim,
        atten_hidden=args.atten_hidden,
        aggr=args.aggr,
        gnn_norm=args.gnn_norm
    ).to(device)
    
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Setup training
    print('\n' + '=' * 80)
    print('Training Setup')
    print('=' * 80)
    
    # Estimate PU prior
    num_positive = (data.y[data.train_mask] == 1).sum().item()
    num_unlabeled = (data.y[data.train_mask] == 0).sum().item()
    prior_alpha = num_positive / num_unlabeled
    print(f'PU Learning: α={prior_alpha:.4f}')
    
    criterion = PU_Loss(prior_probability=prior_alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train
    print('\n' + '=' * 80)
    print('Training')
    print('=' * 80)
    
    if args.use_minibatch:
        # Mini-batch training (for MGD)
        neighbor_sizes = [int(x) for x in args.neighbor_sizes.split(',')]

        if args.feature == 'seq_and_stat':
            # Use hybrid trainer (statistical + sequence)
            trainer = MGDHybridTrainer(
                model=model,
                data=data,
                in_sequences=in_sequences,
                out_sequences=out_sequences,
                seq_lengths=seq_lengths,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                batch_size=args.batch_size,
                neighbor_sizes=neighbor_sizes,
                use_balanced_sampling=args.balanced_sampling
            )
        elif args.feature == 'sequence_only':
            # Use sequence-only trainer (no statistical features)
            trainer = MGDSeqOnlyTrainer(
                model=model,
                data=data,
                in_sequences=in_sequences,
                out_sequences=out_sequences,
                seq_lengths=seq_lengths,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                batch_size=args.batch_size,
                neighbor_sizes=neighbor_sizes,
                use_balanced_sampling=args.balanced_sampling
            )
        else:
            # Use standard MGD trainer (statistical only)
            trainer = MGDTrainer(
                model=model,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                batch_size=args.batch_size,
                neighbor_sizes=neighbor_sizes,
                use_balanced_sampling=args.balanced_sampling
            )
        
        trainer.train(num_epochs=args.num_epochs)
        
        # Evaluate and generate predictions with threshold search
        predictions = evaluate_and_predict_mgd(
            model, data, accounts, test_df, trainer, 
            checkpoint_path=trainer.checkpoint_path
        )
        
    else:
        # Full-graph training (for GraphSAGE)
        try:
            compiled_model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f'Warning: torch.compile failed ({e}), using original model')
            compiled_model = model
        
        scaler = GradScaler()
        
        trainer = Trainer(
            model=model,
            compiled_model=compiled_model,
            data=data,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            eval_threshold=args.threshold
        )
        trainer.train(num_epochs=args.num_epochs)
        
        # Evaluate and generate predictions
        print('\n' + '=' * 80)
        print('Evaluation')
        print('=' * 80)
        
        predictions = evaluate_and_predict(model, data, accounts, test_df)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    predictions.to_csv(output_path, index=False)
    print(f'\nSaved to: {output_path}')
    
    print('\n' + '=' * 80)
    print('Complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()