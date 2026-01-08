import os
import pickle
from typing import Tuple, Dict, Set, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils import *

CUR_RATES = {
    'TWD': 1.00, 'USD': 30.46, 'JPY': 0.21, 'AUD': 20.16,
    'CNY': 4.28, 'EUR': 35.81, 'SEK': 3.24, 'GBP': 41.01,
    'HKD': 3.91, 'THB': 0.94, 'CAD': 21.88, 'NZD': 17.67,
    'CHF': 38.34, 'SGD': 23.62, 'ZAR': 1.77, 'MXN': 1.66,
}

def convert_to_twd(transactions: pd.DataFrame) -> pd.DataFrame:
    """Convert transaction amounts to TWD based on CUR_RATES."""
    df = transactions.copy()
    df['twd_rate'] = df['currency_type'].map(CUR_RATES)
    df['final_amt'] = df['txn_amt'] * df['twd_rate']
    return df


def out_stats_feat(transactions: pd.DataFrame, alert_dates: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
        transactions: Transaction dataframe
        alert_dates: Dictionary mapping alert account IDs to their alert dates (event_date)
                     For alert accounts, exclude transactions on or after the alert date
                     For normal accounts, use all transactions
    """
    agg_ops = {
        'final_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'from_acct_type': lambda x: 1 if len(x) > 0 and x.mode()[0] == 1 else 0,
        'is_self_txn': lambda x: (x == 'Y').sum(),
        'txn_time': 'nunique',
        'channel_type': 'nunique',
        'to_acct': 'nunique'
    }

    # If no alert dates, compute normally
    if alert_dates is None or len(alert_dates) == 0:
        result = transactions.groupby('from_acct').agg(agg_ops).fillna(0)
        print("There's no alert dates.")
    else:
        # Process alert and normal accounts separately
        alert_accounts = set(alert_dates.keys())

        # Normal accounts: use all transactions
        normal_txns = transactions[~transactions['from_acct'].isin(alert_accounts)]
        normal_stats = normal_txns.groupby('from_acct').agg(agg_ops).fillna(0) if len(normal_txns) > 0 else pd.DataFrame()

        # Alert accounts: filter by date for each account
        alert_stats_list = []
        for acct, alert_date in alert_dates.items():
            acct_txns = transactions[(transactions['from_acct'] == acct) & (transactions['txn_date'] < alert_date)]
            if len(acct_txns) > 0:
                acct_stat = acct_txns.groupby('from_acct').agg(agg_ops).fillna(0)
                alert_stats_list.append(acct_stat)

        # Combine results
        if len(alert_stats_list) > 0:
            alert_stats = pd.concat(alert_stats_list)
            if len(normal_stats) > 0:
                result = pd.concat([normal_stats, alert_stats])
            else:
                result = alert_stats
        else:
            result = normal_stats if len(normal_stats) > 0 else pd.DataFrame()

    result.columns = ['from_count', 'from_sum', 'from_mean', 'from_std', 'from_min',
                     'from_max', 'from_acct_type', 'self_txn', 'time_diversity',
                     'channel_diversity', 'counterparty_diversity']
    return result


def in_stats_feat(transactions: pd.DataFrame, alert_dates: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Aggregate incoming transaction statistics."""
    agg_ops = {
        'final_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'to_acct_type': lambda x: 1 if len(x) > 0 and x.mode()[0] == 1 else 0,
        'is_self_txn': lambda x: (x == 'Y').sum(),
        'txn_time': 'nunique',
        'channel_type': 'nunique',
        'from_acct': 'nunique'
    }

    # If no alert dates, compute normally
    if alert_dates is None or len(alert_dates) == 0:
        result = transactions.groupby('to_acct').agg(agg_ops).fillna(0)
    else:
        # Process alert and normal accounts separately
        alert_accounts = set(alert_dates.keys())

        # Normal accounts: use all transactions
        normal_txns = transactions[~transactions['to_acct'].isin(alert_accounts)]
        normal_stats = normal_txns.groupby('to_acct').agg(agg_ops).fillna(0) if len(normal_txns) > 0 else pd.DataFrame()

        # Alert accounts: filter by date for each account
        alert_stats_list = []
        for acct, alert_date in alert_dates.items():
            acct_txns = transactions[(transactions['to_acct'] == acct) & (transactions['txn_date'] < alert_date)]
            if len(acct_txns) > 0:
                acct_stat = acct_txns.groupby('to_acct').agg(agg_ops).fillna(0)
                alert_stats_list.append(acct_stat)

        if len(alert_stats_list) > 0:
            alert_stats = pd.concat(alert_stats_list)
            if len(normal_stats) > 0:
                result = pd.concat([normal_stats, alert_stats])
            else:
                result = alert_stats
        else:
            result = normal_stats if len(normal_stats) > 0 else pd.DataFrame()

    result.columns = ['to_count', 'to_sum', 'to_mean', 'to_std', 'to_min', 'to_max',
                     'to_acct_type', 'to_self_txn', 'to_time_diversity',
                     'to_channel_diversity', 'payer_diversity']
    return result


def compute_ratio_features(stats: pd.DataFrame, direction: str) -> list:
    """Compute ratio-based derived features."""
    eps = 1e-6
    
    if direction == 'outgoing':
        cv = stats['from_std'] / (stats['from_mean'] + eps)
        self_ratio = stats['self_txn'] / (stats['from_count'] + eps)
        concentration = stats['from_count'] / (stats['counterparty_diversity'] + eps)
    else:
        cv = stats['to_std'] / (stats['to_mean'] + eps)
        self_ratio = stats['to_self_txn'] / (stats['to_count'] + eps)
        concentration = stats['to_count'] / (stats['payer_diversity'] + eps)
    
    return [cv, self_ratio, concentration]


def build_node_features(transactions: pd.DataFrame, accounts: list,
                       alert_accounts: Set[str], alert_dates: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, torch.Tensor]:

    outgoing = out_stats_feat(transactions, alert_dates)
    incoming = in_stats_feat(transactions, alert_dates)

    feature_matrix = []
    labels = []

    for account in accounts:
        node_feat = []

        # Outgoing feat (14 dim)
        if account in outgoing.index:
            base_feat = outgoing.loc[account].values[:11]
            ratio_feat = compute_ratio_features(outgoing.loc[account], 'outgoing')
            node_feat.extend(base_feat)
            node_feat.extend(ratio_feat)
        else:
            node_feat.extend([0] * 14)

        # Incoming feat (14 dim)
        if account in incoming.index:
            base_feat = incoming.loc[account].values[:11]
            ratio_feat = compute_ratio_features(incoming.loc[account], 'incoming')
            node_feat.extend(base_feat)
            node_feat.extend(ratio_feat)
        else:
            node_feat.extend([0] * 14)

        feature_matrix.append(node_feat)
        labels.append(1 if account in alert_accounts else 0)

    features = np.array(feature_matrix, dtype=np.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels


def extract_accounts(transactions: pd.DataFrame) -> Tuple[list, Dict[str, int]]:
    """Extract unique accounts and create index mapping."""
    all_accounts = (set(transactions['from_acct'].unique()) | 
                   set(transactions['to_acct'].unique()))
    accounts_sorted = sorted(list(all_accounts))
    account_index = {acc: idx for idx, acc in enumerate(accounts_sorted)}
    return accounts_sorted, account_index


def identify_esun_accounts(transactions: pd.DataFrame) -> Set[str]:

    esun_from = set(transactions[transactions['from_acct_type'] == 1]['from_acct'].unique())
    esun_to = set(transactions[transactions['to_acct_type'] == 1]['to_acct'].unique())
    return esun_from | esun_to


def build_edge_index(transactions: pd.DataFrame, account_index: Dict[str, int]) -> torch.Tensor:

    edges = {(account_index[src], account_index[dst]) 
            for src, dst in zip(transactions['from_acct'], transactions['to_acct'])}
    edge_tensor = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    #edge_index = to_undirected(edge_tensor) #[graphsage] undirected better
    
    return edge_tensor

def construct_graph(transactions: pd.DataFrame, alerts: pd.DataFrame,
                   test_accounts: pd.DataFrame) -> Dict:

    accounts, account_idx = extract_accounts(transactions)
    esun_accounts = identify_esun_accounts(transactions)

    alert_set = set(alerts['acct'].values)
    test_set = set(test_accounts['acct'].values)

    # Create alert_dates dictionary: {acct: event_date}
    alert_dates = dict(zip(alerts['acct'], alerts['event_date']))

    features, labels = build_node_features(transactions, accounts, alert_set, alert_dates)
    edges = build_edge_index(transactions, account_idx)

    return {
        'accounts': accounts,
        'esun_accounts': esun_accounts,
        'test_accounts': test_set,
        'node_features': features,
        'labels': labels,
        'edge_index': edges,
        'num_nodes': len(accounts),
        'alert_dates': alert_dates  # Store for later use in sequence building
    }


# ==================== Data Splitting ====================

def create_split_masks(accounts: list, esun_accounts: Set[str], test_accounts: Set[str],
                      labels: torch.Tensor, num_nodes: int, val_ratio: float = 0.1,
                      random_state: int = 42) -> Tuple[torch.Tensor, ...]:
    """
    Create boolean masks for train/val/test split.
    Only esun accounts are included in train/val/test sets.
    """
    # Separate train+val vs test
    train_val_idx = [i for i, acc in enumerate(accounts) 
                    if acc in esun_accounts and acc not in test_accounts]
    test_idx = [i for i, acc in enumerate(accounts)
               if acc in esun_accounts and acc in test_accounts]
    
    # Split train and validation
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=random_state,
        stratify=labels[train_val_idx]
    )
    
    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask



def scale_features(features: np.ndarray, train_mask: torch.Tensor) -> Tuple[np.ndarray, StandardScaler]:
    """Fit scaler on training data and transform all features."""
    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    scaled_features = scaler.transform(features).astype(np.float32)
    return scaled_features, scaler


def save_graph_cache(directory: str, graph_dict: Dict, scaled_data: Data,
                    scaler: StandardScaler, metadata: Optional[Dict] = None):
    os.makedirs(directory, exist_ok=True)
    print(f'\nSaving graph data to {directory}...')
    
    # Prepare save dictionary
    save_payload = {
        'accounts': graph_dict['accounts'],
        'esun_accounts': graph_dict['esun_accounts'],
        'test_accounts': graph_dict['test_accounts'],
        'node_features_raw': graph_dict['node_features'],
        'labels': graph_dict['labels'],
        'edge_index': graph_dict['edge_index'],
        'train_mask': graph_dict['train_mask'],
        'val_mask': graph_dict['val_mask'],
        'test_mask': graph_dict['test_mask'],
        'scaler': scaler,
        'num_nodes': graph_dict['num_nodes'],
        'num_edges': graph_dict['edge_index'].shape[1],
        'num_features': graph_dict['node_features'].shape[1],
    }
    
    # Add metadata
    if metadata is None:
        metadata = {
            'creation_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_size': graph_dict['train_mask'].sum().item(),
            'val_size': graph_dict['val_mask'].sum().item(),
            'test_size': graph_dict['test_mask'].sum().item(),
            'positive_samples': (graph_dict['labels'] == 1).sum().item(),
            'negative_samples': (graph_dict['labels'] == 0).sum().item(),
        }
    save_payload['metadata'] = metadata
    
    # Save to files
    with open(os.path.join(directory, 'graph_data.pkl'), 'wb') as f:
        pickle.dump(save_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    torch.save(scaled_data, os.path.join(directory, 'data_scaled.pt'))
    
    print(f'✓ Graph data saved')
    print(f'  Nodes: {save_payload["num_nodes"]:,}')
    print(f'  Edges: {save_payload["num_edges"]:,}')
    print(f'  Features: {save_payload["num_features"]}')
    print(f'  Train: {metadata["train_size"]:,} | Val: {metadata["val_size"]:,} | Test: {metadata["test_size"]:,}')


def load_graph_cache(directory: str, verbose: bool = True) -> Tuple:
    """Load preprocessed graph data from disk."""
    if verbose:
        print(f'\nLoading graph data from {directory}...')
    
    # Load pickle
    with open(os.path.join(directory, 'graph_data.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    
    # Load tensor
    scaled_data = torch.load(os.path.join(directory, 'data_scaled.pt'))
    
    if verbose:
        print(f'✓ Graph data loaded')
        print(f'  Nodes: {data_dict["num_nodes"]:,}')
        print(f'  Edges: {data_dict["num_edges"]:,}')
        print(f'  Features: {data_dict["num_features"]}')
    
    return (
        data_dict['accounts'],
        data_dict['esun_accounts'],
        data_dict['test_accounts'],
        data_dict['node_features_raw'],
        data_dict['labels'],
        data_dict['edge_index'],
        data_dict['train_mask'],
        data_dict['val_mask'],
        data_dict['test_mask'],
        data_dict['scaler'],
        scaled_data,
        data_dict.get('metadata', {})
    )


def cache_exists(directory: str) -> bool:
    """Check if cached data exists."""
    pkl_path = os.path.join(directory, 'graph_data.pkl')
    pt_path = os.path.join(directory, 'data_scaled.pt')
    return os.path.exists(pkl_path) and os.path.exists(pt_path)


# ==================== Main Pipeline ====================

def build_pyg_data(features: np.ndarray, edge_index: torch.Tensor, labels: torch.Tensor,
                  train_mask: torch.Tensor, val_mask: torch.Tensor, 
                  test_mask: torch.Tensor) -> Data:
    """Package components into PyG Data object."""
    return Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


def prepare_graph_dataset(data_dir: str, cache_dir: Optional[str] = None,
                         force_rebuild: bool = False, 
                         build_sequences: bool = False,
                         max_seq_len: int = 32) -> Tuple:
    
    # Try loading from cache
    if not force_rebuild and cache_exists(cache_dir):
        print('=' * 80)
        print('Found cached graph data, loading...')
        print('=' * 80)
        result = load_graph_cache(cache_dir, verbose=True)
        
        # Load sequences if requested
        if build_sequences:
            if sequence_cache_exists(cache_dir):
                seq_data = load_sequence_cache(cache_dir, verbose=True)
                if seq_data is not None:
                    # Append sequence data to result
                    return result[:-1] + seq_data[:3]  # Exclude metadata from both
            
            # Sequences not in cache, build them WITHOUT rebuilding graph
            print('\n⚠ Sequences not found in cache, building sequences only...')

            (accounts, esun_accts, test_accts, node_features, labels, edges,
            train_mask, val_mask, test_mask, scaler, scaled_data, metadata) = result
            
            # 重新載入 transactions（只為序列構建）
            transactions = pd.read_csv(os.path.join(data_dir.replace('/processed', ''), 'acct_transaction.csv'))
            # convert_to_twd is already defined in this file, no need to import
            transactions = convert_to_twd(transactions)
            
            # 重新載入 alerts 以獲取 alert_dates
            alerts = pd.read_csv(os.path.join(data_dir.replace('/processed', ''), 'acct_alert.csv'))
            alert_dates = dict(zip(alerts['acct'], alerts['event_date']))

            in_sequences, out_sequences, seq_lengths = build_transaction_sequences(
                transactions,
                accounts,
                max_len=max_seq_len,
                alert_dates=alert_dates
            )
            save_sequence_cache(cache_dir, in_sequences, out_sequences, seq_lengths)
            
            return (accounts, esun_accts, test_accts, node_features, labels, edges,
                    train_mask, val_mask, test_mask, scaler, scaled_data,
                    in_sequences, out_sequences, seq_lengths)
        else:
            return result[:-1]  # Exclude metadata
    
    # Process from scratch
    print('Processing raw data from scratch...')
    
    # Load CSV files
    print('\nLoading CSV files...')
    transactions = pd.read_csv(os.path.join(data_dir, 'acct_transaction.csv'))
    transactions = convert_to_twd(transactions)
    alerts = pd.read_csv(os.path.join(data_dir, 'acct_alert.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'acct_predict.csv'))
    print('✓ CSV files loaded')
    
    # Build graph structure
    print('\nBuilding graph structure...')
    graph_dict = construct_graph(transactions, alerts, test_df)
    print('✓ Graph structure built')
    
    # Create data splits
    print('\nCreating data splits...')
    train_mask, val_mask, test_mask = create_split_masks(
        graph_dict['accounts'],
        graph_dict['esun_accounts'],
        graph_dict['test_accounts'],
        graph_dict['labels'],
        graph_dict['num_nodes']
    )
    graph_dict.update({
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    })
    print('✓ Data splits created')
    
    # Scale features
    print('\nScaling features...')
    scaled_features, scaler = scale_features(graph_dict['node_features'], train_mask)
    scaled_data = build_pyg_data(
        scaled_features, graph_dict['edge_index'], graph_dict['labels'],
        train_mask, val_mask, test_mask
    )
    print('✓ Features scaled')
    
    # Build sequences if requested
    if build_sequences:
        in_sequences, out_sequences, seq_lengths = build_transaction_sequences(
            transactions,
            graph_dict['accounts'],
            max_len=max_seq_len,
            alert_dates=graph_dict.get('alert_dates', None)
        )
        
        # Save sequence cache
        save_sequence_cache(cache_dir, in_sequences, out_sequences, seq_lengths)
    
    # Save to cache
    save_graph_cache(cache_dir, graph_dict, scaled_data, scaler)
    
    print('\n' + '=' * 80)
    print('Data processing complete!')
    print('=' * 80)
    
    base_result = (
        graph_dict['accounts'], graph_dict['esun_accounts'], graph_dict['test_accounts'],
        graph_dict['node_features'], graph_dict['labels'], graph_dict['edge_index'],
        train_mask, val_mask, test_mask, scaler, scaled_data
    )
    
    if build_sequences:
        return base_result + (in_sequences, out_sequences, seq_lengths)
    else:
        return base_result