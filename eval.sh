#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=eval                                # sets the job name if not set from environment
#SBATCH --array=38                        # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/eval_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/eval_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=16:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --gres=gpu:8
#SBATCH --account=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=lowpri
#SBATCH --nice=0                                              #positive means lower priority


source ~/miniconda/etc/profile.d/conda.sh
conda activate ffcv


command_list[0]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100/8c14b25a-1e2a-4ea6-8627-d18958c71d2f/'"
command_list[1]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_advinput0.001' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100_advinput0.001/c1cd6d82-0d84-41dc-8145-d508bd44f63e'"
command_list[2]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_sam0.01' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100_sam0.01/5ee301fb-5765-409a-a0c8-e0b908a538ac'"
command_list[3]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_advfeature0.001_l4' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100_advfeature0.001_l4/80320285-22c9-43a0-8b5b-38d956482d3c'"
command_list[4]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_advfeature0.001_l10' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100_advfeature0.001_l10/90342640-fdb4-47f7-9705-a63f3826bba3'"

command_list[5]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100/ab0b15c5-fd26-4fb9-a9c9-6fe01b447b21'"
command_list[6]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001/72f3d57a-f8c1-4e7a-bbfa-f61813dbe456'"
command_list[7]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_sam0.01' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_sam0.01/203899f5-3445-4009-a3bd-b0bd42180ea2'"
command_list[8]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_sam0.2' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_sam0.2/68d97634-dc56-48a4-818f-0946dd61d69a'"

command_list[9]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.001_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.001_l3/ac709ae1-a472-4709-99c5-f121ecdbb9fa'"
command_list[10]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.001_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.001_l7/d3fdeb8b-1ed3-4458-950a-d52ec2a81357'"
command_list[11]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.001_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.001_l11/390f5866-0f65-446d-a010-d81e0325091a'"

command_list[12]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.002_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.002_l3/d543c969-a30a-4c47-86c6-e61ae368ec53'"
command_list[13]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.002_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.002_l7/8aa05183-b9f1-4b33-acce-9aa62144f7f8'"
command_list[14]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.002_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.002_l11/05b627f1-3548-462e-a01d-12972f8478f7'"

command_list[15]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.0005_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.0005_l3/97ccd2b8-de7c-464b-8e2a-9b43e2dd1b71'"

command_list[16]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.0005_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.0005_l7/9862f907-41fd-428b-a573-8c2505bece22'"
command_list[17]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.0005_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.0005_l11/127a7577-9a17-4efe-b872-d39854d2263d'"


command_list[20]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.005_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.005_l3/b719c4fc-bad1-4780-8654-f7773d532c8c'"
command_list[21]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.005_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.005_l7/d5508d8c-3dcd-468c-b6a8-1b06b626ea9f'"
command_list[22]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.005_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.005_l7/d5508d8c-3dcd-468c-b6a8-1b06b626ea9f'"

command_list[23]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.01_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.01_l3/0a4424e7-be9d-4ae8-ad90-06a6e8ac9ca3'"
command_list[24]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.01_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.01_l7/96d8abfc-5dc1-428a-9342-a7e29dd5b8e1'"
command_list[25]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.01_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.01_l11/1f3d0d45-66d6-4ce1-b326-ebddb3f91261'"

command_list[26]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.02_l3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.02_l3/367cfc0d-338b-41f8-9ce3-5622aebae1d1'"
command_list[27]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.02_l7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.02_l7/198b5fe2-a910-4b2f-a703-7269ddaefee8'"
command_list[28]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advfeature0.02_l11' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advfeature0.02_l11/337b9e21-1925-48ad-a3d0-0739804e8b56'"

command_list[29]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze4_fixed' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze4/e566822b-c417-48cc-976e-07ff5333c4a9'"

command_list[30]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze4_twohead' --eval.checkpoint_path='./output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead/d3dce8a9-f4a2-4801-a58d-38bbfe669b8c'"
command_list[31]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze4_twohead_flip' --eval.checkpoint_path='./output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead_flip/9ff2e570-e73b-441c-b24c-e050e6c2b342'"

command_list[32]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze3' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze3/8d241ca4-6905-4200-b2e0-b057c59f2ea9'"
command_list[33]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze4' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze4/e566822b-c417-48cc-976e-07ff5333c4a9'"
command_list[34]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze5' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze5/08d65163-ebe4-45ae-be4a-0b40a18220f9'"
command_list[35]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze6' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze6/0ffed975-5471-4a50-a5e4-d2976a7cc493'"
command_list[36]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze7' --eval.checkpoint_path='./outputs/nomixup_v3/vit_imagenet100_advinput0.001_freeze7/92451ce7-70d4-4d68-80c5-52c23527d31c'"

command_list[37]="python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --logging.folder './outputs/eval/vit_imagenet100_nomix_advinput0.001_freeze4_twohead_flip_v2' --eval.checkpoint_path='./output/nomixup_v3/vit_imagenet100_advinput0.001_freeze4_twohead_flip_v2/d5ec9fde-aa7b-4440-b55d-a524d4f56704'"

cur_command=${command_list[SLURM_ARRAY_TASK_ID]}
echo $cur_command
eval "$cur_command"

# python eval_checkpoints.py --config-file configs/vit_100cls_eval.yaml --eval.gsnr=1 --logging.folder './outputs/eval/vit_imagenet100_gsnr' --eval.checkpoint_path='./outputs/nomixup_v2/vit_imagenet100/8c14b25a-1e2a-4ea6-8627-d18958c71d2f/' --dist.world_size=2 --eval.linear_probe=0