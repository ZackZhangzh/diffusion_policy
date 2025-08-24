#!/bin/bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890


tasks=("50+30" "40+40" "60+20")

# 设置最大并行任务数
MAX_PARALLEL_TASKS=3
# 轮询间隔时间（秒）
POLL_INTERVAL=30

# --- 脚本主体 ---

# 获取可用的 GPU 数量
num_gpus=$(nvidia-smi -L | wc -l)
if [ "$num_gpus" -eq 0 ]; then
    echo "错误：未检测到任何 NVIDIA GPU。"
    exit 1
fi
echo "检测到 $num_gpus 个 GPU。"

# 创建一个数组来跟踪每个 GPU 上由本脚本启动的进程 ID (PID)
# 0 表示该 GPU 未被本脚本分配任务
declare -a script_gpu_pids
for (( i=0; i<num_gpus; i++ )); do
    script_gpu_pids[$i]=0
done

# 持续循环直到所有任务都完成
while [ ${#tasks[@]} -gt 0 ] || [[ " ${script_gpu_pids[@]} " =~ " [1-9][0-9]* " ]]; do
    # 遍历所有 GPU，检查已完成的任务
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        pid=${script_gpu_pids[$gpu_id]}
        if [ "$pid" -ne 0 ] && ! ps -p "$pid" > /dev/null; then
            echo "信息: GPU $gpu_id 上由本脚本启动的任务 (PID: $pid) 已完成。"
            script_gpu_pids[$gpu_id]=0
        fi
    done

    # 在一次轮询中，尝试填满所有可用的并行槽位
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        # 每次迭代都重新计算当前正在运行的任务数
        running_tasks=0
        for pid in "${script_gpu_pids[@]}"; do
            if [ "$pid" -ne 0 ]; then
                running_tasks=$((running_tasks + 1))
            fi
        done

        # 如果正在运行的任务数已达到上限，或没有待处理任务，则停止寻找
        if [ "$running_tasks" -ge "$MAX_PARALLEL_TASKS" ] || [ ${#tasks[@]} -eq 0 ]; then
            break # 退出寻找空闲 GPU 的循环
        fi

        # 检查此 GPU 是否已被本脚本使用
        if [ "${script_gpu_pids[$gpu_id]}" -eq 0 ]; then
            # 使用 nvidia-smi 检查该 GPU 是否真的没有任何计算任务在运行
            if [ -z "$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader,nounits)" ]; then
                # 找到一个完全空闲的 GPU，分配任务
                HDF5_NAME=${tasks[0]}
                tasks=("${tasks[@]:1}")

                LOG_NAME="${HDF5_NAME}_action_8"
                RUN_DIR="data/outputs/0727/${LOG_NAME}"

                echo "----------------------------------------------------"
                echo "状态: 发现完全空闲的 GPU: $gpu_id"
                echo "操作: 在其上启动新任务 (HDF5_NAME=$HDF5_NAME)"
                echo "当前运行任务数: $running_tasks, 最大并行数: $MAX_PARALLEL_TASKS"
                echo "----------------------------------------------------"

                # 在后台运行训练命令
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
                    --config-dir=config \
                    --config-name=40_updateframe.yaml \
                    training.device=cuda:0 \
                    training.seed=42 \
                    task.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
                    task.dataset.dataset_path="data/hdf5/${HDF5_NAME}.hdf5" \
                    logging.name="${LOG_NAME}" \
                    hydra.run.dir="${RUN_DIR}" &

                # 保存新任务的 PID
                script_gpu_pids[$gpu_id]=$!
                echo "任务已启动，PID 为 ${script_gpu_pids[$gpu_id]}"
            fi
        fi
    done

    # 检查是否需要继续循环
    if [ ${#tasks[@]} -gt 0 ] || [[ " ${script_gpu_pids[@]} " =~ " [1-9][0-9]* " ]]; then
         # 如果还有任务在等待分配，或者还有任务正在运行，则打印等待信息
        if [ ${#tasks[@]} -gt 0 ]; then
            echo "等待空闲 GPU... 剩余任务数: ${#tasks[@]}。将在 $POLL_INTERVAL 秒后再次检查。"
        else
            echo "所有任务都已分配。等待正在运行的任务完成... 将在 $POLL_INTERVAL 秒后检查状态。"
        fi
        sleep $POLL_INTERVAL
    fi
done

echo "所有任务均已成功完成。脚本执行结束。"