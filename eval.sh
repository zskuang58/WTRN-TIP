### evaluation
CUDA_VISIBLE_DEVICES=0 \
python main.py --save_dir ./eval/CUFED/WTRN \
               --reset True \
               --which_model WTRN \
               --log_file_name eval.log \
               --eval True \
               --save_model False \
               --eval_save_results False \
               --num_workers 4 \
               --dataset CUFED \
               --dataset_dir /home/lab426/Codes/Codes/WTRN/ \
               --model_path /home/lab426/Codes/WTRN/WTRN.pth
