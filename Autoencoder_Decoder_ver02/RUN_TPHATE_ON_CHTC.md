# 在 CHTC 上运行 T-PHATE 生成脚本

## 🔍 当前方式：直接在登录节点运行

你现在的运行方式：

```bash
# 直接在登录节点运行
python3 generate_tphate_for_aadhitya.py ...
```

**问题**：
- 登录节点资源有限
- 处理 704 个胚胎可能需要很长时间
- 可能会影响其他用户
- 断开连接后可能中断

## ✅ 推荐方式：使用 HTCondor 提交作业

### 步骤 1: 准备文件

```bash
# SSH 到 CHTC
ssh rho9@ap2001.chtc.wisc.edu

# 确保脚本和 submit 文件都在工作目录
cd /staging/groups/bhaskar_group/rho9

# 创建日志目录
mkdir -p logs
```

### 步骤 2: 编辑 submit 文件

```bash
# 检查 submit 文件
cat generate_tphate.sub

# 如果需要修改参数（例如只处理 validation set），编辑文件
nano generate_tphate.sub
```

### 步骤 3: 提交作业

```bash
# 提交作业
condor_submit generate_tphate.sub

# 你会看到类似输出：
# Submitting job(s).
# 1 job(s) submitted to cluster XXXXX.
```

### 步骤 4: 监控作业

```bash
# 查看作业状态
condor_q -submitter rho9

# 查看实时输出
condor_tail -f <ClusterID>

# 查看详细状态
condor_q <ClusterID> -better-analyze
```

## 📊 两种方式对比

| 特性 | 直接在登录节点运行 | HTCondor 提交作业 |
|------|-------------------|------------------|
| 资源 | 有限（共享） | 专用计算节点 |
| GPU | 无 | 可申请 |
| 长时间运行 | 可能被限制 | 支持 |
| 断开连接 | 可能中断 | 继续运行 |
| 监控 | 实时输出 | 查看日志 |
| 适合 | 小规模测试 | 大规模处理 |

## 🎯 建议

对于 **704 个胚胎**，建议：

1. **小规模测试**（5-10 个）：直接在登录节点运行 ✅
2. **大规模处理**（全部或 validation set）：使用 HTCondor ✅

## 💡 快速切换

如果当前脚本正在登录节点运行，你可以：

```bash
# 1. 停止当前运行（Ctrl+C）

# 2. 使用 HTCondor 重新提交
condor_submit generate_tphate.sub

# 3. 监控作业
condor_q -submitter rho9
condor_tail -f <ClusterID>
```






