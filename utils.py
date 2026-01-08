import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict, List
from tqdm import tqdm


def build_transaction_sequences(transactions: pd.DataFrame, accounts: list,
    max_len: int = 32, feature_cols: List[str] = None,
    alert_dates: Dict[str, int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    
    if feature_cols is None:
        # Default sequence features (7 dimensions) - 全是統計(數量)特徵
        feature_cols = [
            'final_amt',          # Transaction amount (will be normalized)
            'currency_type',      # Currency type (will be encoded)
            'channel_type',       # Transaction channel (will be encoded)
            'is_self_txn',        # Self-transfer indicator (will be encoded)
            'txn_timestamp',      # Combined date+time (normalized)
            'txn_hour',           # Hour of day (normalized)
            'txn_day_of_week',    # Day of week (normalized)
        ]
    
    print(f'\nBuilding transaction sequences (OPTIMIZED)...')
    print(f'  Max sequence length: {max_len}')
    print(f'  Sequence features: {feature_cols}')
    
    # ===== Feature Engineering =====
    
    print(f'  Processing features...')
    
    is_self_mapping = {
        'N': 0.0,    
        'UNK': 0.5,  
        'Y': 1.0    
    }
    
    transactions['is_self_txn_encoded'] = transactions['is_self_txn'].map(is_self_mapping).astype(float)    
    
    # 2. Process txn_date (days since first day, starting from 1)
    date_min = transactions['txn_date'].min()
    date_max = transactions['txn_date'].max()
    transactions['txn_date_norm'] = (transactions['txn_date'] - date_min) / (date_max - date_min + 1e-8)
    
    print(f'  Date range: day {date_min} to day {date_max} ({date_max - date_min + 1} days)')
    
    
    # 3. Process txn_time
    if transactions['txn_time'].dtype == 'object':
        transactions['txn_time_parsed'] = pd.to_datetime(transactions['txn_time'], format='%H:%M:%S', errors='coerce')
        transactions['txn_seconds'] = (
            transactions['txn_time_parsed'].dt.hour * 3600 + 
            transactions['txn_time_parsed'].dt.minute * 60 + 
            transactions['txn_time_parsed'].dt.second
        )
    else:
        transactions['txn_seconds'] = transactions['txn_time']
    
    transactions['txn_time_norm'] = transactions['txn_seconds'] / 86400.0
    
    
    # 4. Combine date and time into timestamp
    transactions['txn_timestamp_seconds'] = (transactions['txn_date'] - 1) * 86400 + transactions['txn_seconds']
    
    ts_min = transactions['txn_timestamp_seconds'].min()
    ts_max = transactions['txn_timestamp_seconds'].max()
    transactions['txn_timestamp'] = (transactions['txn_timestamp_seconds'] - ts_min) / (ts_max - ts_min + 1e-8)
    
    # Sort by timestamp (critical for chronological order)
    transactions = transactions.sort_values('txn_timestamp_seconds').reset_index(drop=True)
    
    
    # 5. Extract temporal features
    transactions['txn_hour'] = (transactions['txn_seconds'] // 3600) / 23.0
    transactions['txn_day_of_week'] = ((transactions['txn_date'] - 1) % 7) / 6.0
    
    
    # 6. Encode channel_type (兩位數字字串: '01', '02', ... '99', 'UNK')
    channel_mapping = {
        '01': 0,   # ATM
        '02': 1,   # 臨櫃
        '03': 2,   # 銀行
        '04': 3,   # 網銀
        '05': 4,   # 語音
        '06': 5,   # eatm
        '07': 6,   # 電子支付
        '99': 7,   # 系統排程
        'UNK': 8   # 空值
    }
    
    # 直接映射（已經是字串格式）
    transactions['channel_type_encoded'] = transactions['channel_type'].map(channel_mapping).fillna(8).astype(float)
        
    # 7. Encode currency_type
    if 'currency_type' in transactions.columns:
        if transactions['currency_type'].dtype == 'object':
            unique_currencies = sorted(transactions['currency_type'].unique())
            currency_mapping = {curr: idx for idx, curr in enumerate(unique_currencies)}
            transactions['currency_type_encoded'] = transactions['currency_type'].map(currency_mapping).fillna(0).astype(float)
            print(f'  Currency types encoded: {len(currency_mapping)} unique currencies')
            curr_counts = transactions['currency_type'].value_counts().head(3)
            print(f'  Top 3 currencies: {list(curr_counts.index)} ({curr_counts.values[0]:,}, {curr_counts.values[1]:,}, {curr_counts.values[2]:,} txns)')
        else:
            transactions['currency_type_encoded'] = transactions['currency_type'].astype(float)
    else:
        transactions['currency_type_encoded'] = 0.0
        print(f'  Warning: No currency_type column found, using default value 0')
    
    
    # 8. Normalize final_amt using log transformation
    print(f'  Normalizing final_amt...')
        
    # Log1p transformation (handles values close to 0)
    transactions['final_amt_log'] = np.log1p(transactions['final_amt'])
    
    # Standardization (mean=0, std=1)
    log_mean = transactions['final_amt_log'].mean()
    log_std = transactions['final_amt_log'].std()
    transactions['final_amt_normalized'] = (transactions['final_amt_log'] - log_mean) / (log_std + 1e-8)
    
    
    # Update feature_cols to use encoded/normalized versions
    feature_cols_final = []
    for col in feature_cols:
        if col == 'channel_type':
            feature_cols_final.append('channel_type_encoded')
        elif col == 'currency_type':
            feature_cols_final.append('currency_type_encoded')
        elif col == 'final_amt':
            feature_cols_final.append('final_amt_normalized')
        elif col == 'is_self_txn':
            feature_cols_final.append('is_self_txn_encoded')
        elif col == 'txn_time':
            feature_cols_final.append('txn_time_norm')
        elif col == 'txn_date':
            feature_cols_final.append('txn_date_norm')
        else:
            feature_cols_final.append(col)
    
    feature_cols = feature_cols_final
    print(f'  Final feature columns: {feature_cols}')


    # ======================= Build Sequences Using Dictionary ==========================

    print(f'  Building sequences (optimized method)...')
    if alert_dates is not None and len(alert_dates) > 0:
        print(f'  Alert filtering enabled for {len(alert_dates)} alert accounts')

    # Create account index mapping
    account_to_idx = {acc: idx for idx, acc in enumerate(accounts)}
    n_accounts = len(accounts)
    
    # Initialize dictionaries to collect sequences
    in_sequences_dict = {idx: [] for idx in range(n_accounts)}
    out_sequences_dict = {idx: [] for idx in range(n_accounts)}

    # Extract feature matrix once
    feature_matrix = transactions[feature_cols].values
    from_accts = transactions['from_acct'].values
    to_accts = transactions['to_acct'].values
    txn_dates = transactions['txn_date'].values

    # Single pass through transactions (O(M))
    print(f'  Collecting transactions...')
    filtered_count = 0
    for i in tqdm(range(len(transactions)), desc='Processing transactions'):
        feat = feature_matrix[i]
        txn_date = txn_dates[i]
        from_acc = from_accts[i]
        to_acc = to_accts[i]

        # Outgoing transaction - only add if account is not alert OR txn is before alert date
        if from_acc in account_to_idx:
            if alert_dates is None or from_acc not in alert_dates or txn_date < alert_dates[from_acc]:
                out_sequences_dict[account_to_idx[from_acc]].append(feat)
            else:
                filtered_count += 1

        # Incoming transaction - only add if account is not alert OR txn is before alert date
        if to_acc in account_to_idx:
            if alert_dates is None or to_acc not in alert_dates or txn_date < alert_dates[to_acc]:
                in_sequences_dict[account_to_idx[to_acc]].append(feat)
            else:
                filtered_count += 1

    if alert_dates is not None and len(alert_dates) > 0:
        print(f'  Excluded {filtered_count:,} transaction entries from alert account sequences')
    
    # Convert to tensors and truncate/pad
    print(f'  Converting to tensors and padding...')
    in_sequences = []
    out_sequences = []
    lengths_in = []
    lengths_out = []
    
    for idx in tqdm(range(n_accounts), desc='Padding sequences'):
        # Incoming sequences
        in_seq_list = in_sequences_dict[idx]
        if len(in_seq_list) > 0:
            # Take most recent max_len transactions
            in_seq = np.array(in_seq_list[-max_len:], dtype=np.float32)
            in_sequences.append(torch.FloatTensor(in_seq))
            lengths_in.append(len(in_seq))
        else:
            in_sequences.append(torch.zeros(1, len(feature_cols), dtype=torch.float32))
            lengths_in.append(1)
        
        # Outgoing sequences
        out_seq_list = out_sequences_dict[idx]
        if len(out_seq_list) > 0:
            out_seq = np.array(out_seq_list[-max_len:], dtype=np.float32)
            out_sequences.append(torch.FloatTensor(out_seq))
            lengths_out.append(len(out_seq))
        else:
            out_sequences.append(torch.zeros(1, len(feature_cols), dtype=torch.float32))
            lengths_out.append(1)
    
    # Pad sequences to same length
    in_sequences_padded = pad_sequence(in_sequences, batch_first=True, padding_value=0.0)
    out_sequences_padded = pad_sequence(out_sequences, batch_first=True, padding_value=0.0)
    
    seq_lengths = {
        'in': torch.LongTensor(lengths_in),
        'out': torch.LongTensor(lengths_out)
    }
    
    print(f'✓ Sequence features built')
    print(f'  In sequences shape: {in_sequences_padded.shape}')
    print(f'  Out sequences shape: {out_sequences_padded.shape}')
    print(f'  Feature dim: {in_sequences_padded.shape[-1]}')
    
    # Print feature statistics
    print(f'\n  Feature statistics:')
    for i, feat in enumerate(feature_cols):
        in_values = in_sequences_padded[:, :, i][in_sequences_padded[:, :, i] != 0]
        if len(in_values) > 0:
            print(f'    {feat}: min={in_values.min():.3f}, max={in_values.max():.3f}, mean={in_values.mean():.3f}')
    
    return in_sequences_padded, out_sequences_padded, seq_lengths


def save_sequence_cache(directory: str, in_sequences: torch.Tensor, 
                       out_sequences: torch.Tensor, seq_lengths: Dict):
    
    seq_cache_path = os.path.join(directory, 'sequences.pt')
    
    torch.save({
        'in_sequences': in_sequences,
        'out_sequences': out_sequences,
        'seq_lengths_in': seq_lengths['in'],
        'seq_lengths_out': seq_lengths['out'],
        'metadata': {
            'max_len': in_sequences.shape[1],
            'feat_dim': in_sequences.shape[2],
            'num_accounts': in_sequences.shape[0]
        }
    }, seq_cache_path)
    
    print(f'✓ Sequences saved to {seq_cache_path}')


def load_sequence_cache(directory: str, verbose: bool = True) -> Tuple:
    
    seq_cache_path = os.path.join(directory, 'sequences.pt')
    
    if not os.path.exists(seq_cache_path):
        return None
    
    data = torch.load(seq_cache_path)
    
    if verbose:
        print(f'\n✓ Sequences loaded from cache')
        print(f'  Shape: {data["in_sequences"].shape}')
        print(f'  Feature dim: {data["metadata"]["feat_dim"]}')
    
    return (
        data['in_sequences'],
        data['out_sequences'],
        {
            'in': data['seq_lengths_in'],
            'out': data['seq_lengths_out']
        },
        data['metadata']
    )


def sequence_cache_exists(directory: str) -> bool:
    """Check if sequence cache exists."""
    return os.path.exists(os.path.join(directory, 'sequences.pt'))