source /data/home/pingchiang/miniconda/etc/profile.d/conda.sh
conda activate ffcv_v4

module load /data/home/vkhalidov/modulefiles/cuda/11.3
module load /data/home/vkhalidov/modulefiles/nccl/2.12.7-cuda.11.3
module load /data/home/vkhalidov/modulefiles/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCC_INFO=INFO
export NCCL_SOCKET_IFNAME=ens32

python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_success --logging.project_name test --dist.world_size=4 --data.num_workers=4" &> test1_success.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_incorrect_num_workers --logging.project_name test --dist.world_size=4 --data.num_workers=20" &> test2_num_worker.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_incorrect_space --logging.project_name test --dist.world_size=4       --data.num_workers=4" &> test3_incorrect_space.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_fake_bug --logging.project_name test --dist.world_size=4  --training.fake_bug=1 --data.num_workers=4" &> test4_fake_bug.log &
python -u launch.py --command="python train_imagenet.py --config-file configs/vit_100cls.yaml --training.epochs=3 --logging.folder outputs/test_node_failure --logging.project_name test --dist.world_size=4 --data.num_workers=4" --test_node_failure &> test5_test_node_failure.log &
