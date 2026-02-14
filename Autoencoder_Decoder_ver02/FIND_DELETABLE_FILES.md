# 查找可以删除的文件

## 运行检查命令

```bash
cd /staging/groups/bhaskar_group/rho9

# 1. 检查各个目录大小
du -sh * 2>/dev/null | sort -h

# 2. 检查日志文件
find . -name "*.log" -type f -exec du -sh {} \; 2>/dev/null | sort -h

# 3. 检查 ~/logs 目录
du -sh ~/logs
ls -lh ~/logs/*.log 2>/dev/null | head -20

# 4. 检查临时文件
du -sh tmp
ls -lh tmp/ 2>/dev/null
```

## 通常可以安全删除的文件

1. **日志文件** (.log)
   - 旧的运行日志
   - 已经完成的作业日志

2. **临时文件** (tmp/)
   - 临时脚本
   - 临时输出

3. **旧的测试结果目录**
   - aadhitya_v1_test (11M)
   - ivf_analysis (17M)
   - curvature_analysis (31M)
   - 如果这些是旧的结果，可以删除或压缩

4. **旧的 checkpoint**
   - 如果 checkpoints 目录有多个版本，可以删除旧的

运行检查命令后，告诉我结果，我会帮你决定哪些可以安全删除。






