# training DeepDDI
python -m baselines.DDI_deepwalk --seed 1 --data_dir ./data/DeepDDI --data_prefix ddi --data_ratio 5
python -m DDItest.DDI_nn_conv --seed 1 --data_dir ./data/DeepDDI --data_prefix ddi --data_ratio 5 --no_cuda
python -m DDItest.DDI_Energy --seed 1 --lr 0.001 --data_dir ./data/DeepDDI --data_prefix ddi --resume_path saved_models/CONV_dataset_ddi_seed_1_ratio_5.0 --data_ratio 5 --no_cuda
python -m DDItest.DDI_Local_Energy --seed 1 --lr 0.001 --data_dir ./data/DeepDDI --data_prefix ddi --resume_path saved_models/CONV_dataset_ddi_seed_1_ratio_5.0 --data_ratio 5 --no_cuda
python -m DDItest.DDI_EnergySup --seed 1 --lr 0.001 --data_dir ./data/DeepDDI --data_prefix ddi --resume_path saved_models/CONV_dataset_ddi_seed_1_ratio_5.0 --data_ratio 5 --no_cuda

# training Decagon
python -m baselines.DDI_deepwalk --seed 1 --data_dir ./data/Decagon --data_prefix decagon --data_ratio 5
python -m DDItest.DDI_nn_conv --seed 1 --data_dir ./data/Decagon --data_prefix decagon --data_ratio 5 --no_cuda
python -m DDItest.DDI_Energy --seed 1 --lr 0.001 --data_dir ./data/Decagon --data_prefix decagon --resume_path saved_models/CONV_dataset_decagon_seed_1_ratio_5.0 --data_ratio 5 --no_cuda
python -m DDItest.DDI_Local_Energy --seed 1 --lr 0.001 --data_dir ./data/Decagon --data_prefix decagon --resume_path saved_models/CONV_dataset_decagon_seed_1_ratio_5.0 --data_ratio 5 --no_cuda
python -m DDItest.DDI_EnergySup --seed 1 --lr 0.001 --data_dir ./data/Decagon --data_prefix decagon --resume_path saved_models/CONV_dataset_decagon_seed_1_ratio_5.0 --data_ratio 5 --no_cuda
python -m baselines.DDI_LP --seed 1 --data_dir ./data/Decagon --data_prefix decagon --data_ratio 5

