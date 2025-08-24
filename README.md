#

```bash
pkill -f mihomo

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
conda activate robodiff

CONFIG_NAME="qpos40"
HDF5_NAME="${CONFIG_NAME}_new"
python train.py --config-dir=config --config-name="${CONFIG_NAME}.yaml" \
training.device=cuda:4 training.seed=42 \
task.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
task.dataset.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
logging.name="NEW_${HDF5_NAME}_action_8" \
hydra.run.dir="data/outputs/0803/${HDF5_NAME}_action_8"

# gripper_qpos = 16 
# action= 25
python train.py --config-dir=config --config-name=40_updateframe.yaml training.device=cuda:1 training.seed=42 \
task.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
task.dataset.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
logging.name="${HDF5_NAME}_action_2" \
n_action_steps=2 policy.n_action_steps=2 \
hydra.run.dir="data/outputs/0727/${HDF5_NAME}_action_2"




# 一定要在主机上scp，走集群io非常慢
ssh -p 1200 yaxun@10.19.126.196

REMOTE=/home/zhangzhh12024/HomeWork/diffusion_policy/data/outputs/0803
scp -P 22112 -r zhangzhh12024@10.15.49.36:"${REMOTE/#\/home/\/grpczm}" /media/yaxun/B197/teleop_data/checkpoints_NEW/

# 36是文件专用节点，如果不行试试6，7
# 10.15.49.6
# 10.15.49.7
```
