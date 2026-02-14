# 监控 HTCondor 作业

## ✅ 作业已提交并运行

如果 `condor_q` 显示作业在 running，说明作业已经成功提交！

## 🔍 查看日志的方法

### 方法 1: 直接查看日志文件（推荐）

```bash
# 查看输出日志（实时）
tail -f ~/logs/generate_tphate_*.out

# 查看错误日志
tail -f ~/logs/generate_tphate_*.err

# 查看完整日志
tail -f ~/logs/generate_tphate_*.log
```

### 方法 2: 查看最新的日志

```bash
# 找到最新的日志文件
ls -t ~/logs/generate_tphate_*.out | head -1

# 查看最新输出
tail -f $(ls -t ~/logs/generate_tphate_*.out | head -1)
```

### 方法 3: 使用 condor_tail（如果知道 Cluster ID）

```bash
# 先获取 Cluster ID
condor_q -submitter rho9

# 然后使用（替换 XXXXX 为实际的 Cluster ID）
condor_tail -f XXXXX
```

### 方法 4: 查看作业详细信息

```bash
# 查看作业状态
condor_q -submitter rho9 -better-analyze

# 查看作业历史
condor_history -submitter rho9 | head -5
```

## 📊 检查进度

```bash
# 查看已生成的 plot 数量
ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | wc -l
ls -1 /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/curvature_plots/*.png 2>/dev/null | wc -l

# 查看最后处理的胚胎
ls -t /staging/groups/bhaskar_group/rho9/aadhitya_v1_val/tphate_plots/*.png 2>/dev/null | head -1
```

## 💡 为什么 condor_tail 可能不工作？

可能的原因：
1. 日志文件路径问题（日志在 home 目录，不是 staging）
2. 作业刚启动，还没开始写日志
3. condor_tail 需要正确的 Cluster ID

**解决方案**：直接使用 `tail -f` 查看日志文件更可靠！






