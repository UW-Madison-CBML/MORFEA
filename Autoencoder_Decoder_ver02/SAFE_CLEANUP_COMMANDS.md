# 安全清理 Home 目录

## 可以安全删除的内容

### 1. .cache 目录 (124M)
```bash
rm -rf ~/.cache
```
**安全**：这是缓存文件，删除后会在需要时自动重建。

### 2. ivf-embryo-analysis-Raffael.tgz (555M)
```bash
rm ~/ivf-embryo-analysis-Raffael.tgz
```
**注意**：如果这个压缩文件不再需要，可以删除。如果以后可能需要，建议先备份或确认。

### 3. 重复的虚拟环境（谨慎）
- `.venv` (7.0G)
- `Desktop/.venv` (如果有的话)

如果这些虚拟环境是重复的，可以删除其中一个。但要确认它们不是必需的。

## 总计可以释放

- .cache: 124M
- ivf-embryo-analysis-Raffael.tgz: 555M
- **总计：约 679M**

## 运行清理

```bash
# 1. 删除 .cache（最安全）
rm -rf ~/.cache

# 2. 删除压缩文件（如果不需要）
rm ~/ivf-embryo-analysis-Raffael.tgz

# 3. 检查配额
quota -s

# 4. 检查释放了多少空间
du -sh ~
```

删除后可以释放约 679M，配额应该会从 39921M 降到约 39242M，这样就可以继续运行了。






