
# Alert-Account-Detection

data files put into ./data folder.

    
! preprocessing.py中的build_edge_index()，可以設定有向圖或無向圖，graphsage模型在無向圖的結構下數據較好~~

## GraphSage: 
python train.py --model graphsage --hidden_channels 56 --lr 0.005 --weight_decay 1e-5 --num_epochs 1000 --dropout 0.3 --num_layers 4 --aggr mean
