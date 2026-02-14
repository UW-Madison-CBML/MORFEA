# 清理 Home 目录释放配额

## 检查命令

```bash
# 1. 检查 home 目录总大小
du -sh ~

# 2. 检查各个目录大小
du -sh ~/* ~/.[^.]* 2>/dev/null | sort -hr | head -20

# 3. 检查常见的大目录
du -sh ~/.cache ~/.local ~/logs ~/tmp ~/.conda ~/.ipython 2>/dev/null

# 4. 检查所有大于 10M 的文件
find ~ -type f -size +10M -exec du -sh {} \; 2>/dev/null | sort -hr | head -20

# 5. 检查日志文件
find ~ -type f -name "*.log" -size +1M -exec du -sh {} \; 2>/dev/null | sort -hr
```

## 通常可以安全删除的内容

1. **.cache 目录** - 缓存文件，可以删除
2. **旧的日志文件** - 可以删除
3. **.local/lib/python3.x/site-packages 的缓存** - 可以清理
4. **临时文件** - tmp 目录

运行检查命令，找出可以删除的内容。






