#!/bin/bash

# 检查参数数量
if [ $# -ne 3 ]; then
    echo "用法: $0 <源文件夹> <目标文件夹> <目标大小(GB)>"
    exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"
TARGET_SIZE_GB="$3"

# 检查源文件夹是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源文件夹 '$SOURCE_DIR' 不存在"
    exit 1
fi

# 检查目标文件夹是否存在，如果不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# 将目标大小转换为字节
TARGET_SIZE_BYTES=$((TARGET_SIZE_GB * 1024 * 1024 * 1024))
CURRENT_SIZE=0

# 获取所有 tar 文件并随机排序
echo "正在收集 tar 文件列表..."
tar_files=$(find "$SOURCE_DIR" -type f -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" | shuf)

# 复制文件直到达到目标大小
for file in $tar_files; do
    if [ $CURRENT_SIZE -ge $TARGET_SIZE_BYTES ]; then
        break
    fi
    
    file_size=$(stat -c %s "$file")
    if [ $((CURRENT_SIZE + file_size)) -le $TARGET_SIZE_BYTES ]; then
        echo "正在复制: $file"
        cp "$file" "$TARGET_DIR/"
        CURRENT_SIZE=$((CURRENT_SIZE + file_size))
    fi
done

# 显示最终结果
echo "复制完成！"
echo "已复制文件总大小: $((CURRENT_SIZE / 1024 / 1024 / 1024)) GB" 