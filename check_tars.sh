#!/bin/bash

# 设置日志文件
LOG_FILE="tar_check_$(date +%Y%m%d_%H%M%S).log"
TEMP_DIR="/tmp/tar_check_$$"  # 使用进程ID创建临时目录

# 创建临时目录
mkdir -p "$TEMP_DIR"

# 记录开始时间
echo "开始检查时间: $(date)" | tee -a "$LOG_FILE"

# 计数器
total_files=0
corrupted_files=0

# 检查每个tar文件
for tar_file in /data/training/laion-300M-sampling/*.tar; do
    if [ ! -f "$tar_file" ]; then
        continue
    fi
    
    total_files=$((total_files + 1))
    echo "正在检查: $tar_file" | tee -a "$LOG_FILE"
    
    # 尝试解压到临时目录
    if tar -tf "$tar_file" >/dev/null 2>&1; then
        echo "文件完整: $tar_file" | tee -a "$LOG_FILE"
    else
        echo "文件损坏: $tar_file" | tee -a "$LOG_FILE"
        rm -f "$tar_file"
        corrupted_files=$((corrupted_files + 1))
    fi
done

# 清理临时目录
rm -rf "$TEMP_DIR"

# 记录统计信息
echo "检查完成时间: $(date)" | tee -a "$LOG_FILE"
echo "总文件数: $total_files" | tee -a "$LOG_FILE"
echo "损坏文件数: $corrupted_files" | tee -a "$LOG_FILE"

echo "检查完成，详细日志请查看: $LOG_FILE"