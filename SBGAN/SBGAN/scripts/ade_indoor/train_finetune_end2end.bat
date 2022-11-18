@echo off

SET gpu_ids=0
SET dataset=ade_indoor
SET name_joint=%dataset%
SET name=ft_%name_joint%_end2end
SET dataroot=datasets\ade_indoor
SET ckpt=weights
SET spade_epoch=300
SET pro_iter=519766
SET D2_iter=0
SET lambda=10
SET bs=1
SET load_size=256
SET crp_size=256
SET num_semantics=95
SET label_seg=95
SET lr=0.0001
SET niter=50
SET niterd=50
SET nums_fid=433
SET save_freq=25
SET eval_freq=5
IF NOT EXIST "%ckpt%/%name%" (
  mkdir "%ckpt%/%name%"
)
SET pro_gan_pretrained="%ckpt%\%name_joint%_segment"
IF NOT EXIST "%ckpt%\%name%\%pro_iter%.pth" (
  COPY  "%pro_gan_pretrained%\%pro_iter%.pth" "%ckpt%\%name%/"
)
SET spade_pretrained="..\SPADE\%ckpt%\%name_joint%"
IF NOT EXIST "%ckpt%\%name%\%spade_epoch%_net_G.pth" (
  COPY  "%spade_pretrained%\%spade_epoch%_net_G.pth" "%ckpt%\%name%\"
  COPY  "%spade_pretrained%\%spade_epoch%_net_D.pth" "%ckpt%\%name%\"
)

python "SBGAN\trainers\progressive_seg_end2end_trainer.py" "--name" "%name%" "--dataset" "%dataset%" "--dataset_mode" "%dataset%" "--dataroot" "%dataroot%" "--num_semantics" "%num_semantics%" "--label_seg" "%label_seg%" "--lr_pgan" "%lr%" "--load_size" "%load_size%" "--crop_size" "%crp_size%" "--checkpoints_dir" "%ckpt%" "--nums_fid" "%nums_fid%" "--eval_freq" "%eval_freq%" "--save_epoch_freq" "%save_freq%" "--batchSize" "%bs%" "--lambda_pgan" "%lambda%" "--which_epoch" "%spade_epoch%" "--which_iter" "%pro_iter%" "--which_iter_D2" "%D2_iter%" "--niter" "%niter%" "--niter_decay" "%niterd%" "--tf_log" "--no_instance" "--end2end" "--pretrain_D2" "--continue_train" "--cont_train"