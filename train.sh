### training WTRN
CUDA_VISIBLE_DEVICES=0 \
python main.py --save_dir ./train/CUFED/WTRN-allloss \
               --reset True \
               --which_model WTRN \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 9 \
               --dataset CUFED \
               --dataset_dir /home/lab426/Codes/WTRN_TIP/dataset/CUFED \
               --n_feats 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-4 \
               --batch_size 9 \
               --num_init_epochs 2 \
               --num_epochs 50 \
               --print_every 50 \
               --save_every 10 \
               --val_every 1


# ### training WTRN-rec
#CUDA_VISIBLE_DEVICES=0 \
# python main.py --save_dir ./train/CUFED/WTRN-rec \
#                --reset True \
#                --which_model WTRN \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
#                --dataset CUFED \
#                --dataset_dir /home/lab426/Codes/Reference-SR/dataset/CUFED \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 9 \
#                --num_init_epochs 0 \
#                --num_epochs 200 \
#                --print_every 50 \
#                --save_every 5 \
#                --val_every 1