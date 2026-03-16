# train SID
python train_denoising.py  --use_tb_logger --loss_l1 --vis_freq 50  --save_epoch_freq 50 --name train_denoising_sid --trainset SIDSyntheticDataset --batch_size 4 --max_iter 500 --crop_size 256 --darkshading_folder data/SID_DarkShadings_BlurSigma50  --darkframe_folder data/SID_SampledDarkFrames --darkframe_num 400


# train LRID
python train_denoising.py  --use_tb_logger --loss_l1 --vis_freq 50  --save_epoch_freq 50 --name train_denoising_lrid --trainset LRIDSyntheticDataset --batch_size 4 --max_iter 500 --crop_size 256 --darkshading_folder data/LRID_DarkShadings_BlurSigma50 --hot_darkshading_folder data/LRID_Hot_DarkShadings_BlurSigma50  --darkframe_folder data/LRID_SampledDarkFrames --hot_darkframe_folder data/LRID_Hot_SampledDarkFrames --darkframe_num 400  