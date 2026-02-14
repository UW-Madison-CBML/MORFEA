# 检查整个目录

## 运行完整的检查

```bash
cd /staging/groups/bhaskar_group/rho9

# 1. 检查所有目录大小（按大小排序）
du -sh * 2>/dev/null | sort -hr

# 2. 检查所有大于 10M 的文件
find . -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr

# 3. 检查 checkpoints 目录
ls -lh checkpoints/

# 4. 检查是否有压缩文件
find . -type f \( -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" \) -exec du -sh {} \; 2>/dev/null

# 5. 检查所有子目录
find . -type d -maxdepth 2 -exec du -sh {} \; 2>/dev/null | sort -hr

# 6. 检查配额
quota -s
```

## 通常可以删除的内容

1. **旧的测试目录**
   - aadhitya_v1_test (如果还没删除)
   - 其他测试目录

2. **旧的备份或压缩文件**
   - .tar.gz 文件
   - .zip 文件

3. **重复的 checkpoint**
   - 如果有多个版本的 checkpoint，只保留最新的

4. **临时文件**
   - tmp 目录
   - 临时输出文件

运行检查命令，告诉我结果，我会帮你找出可以删除的内容。






