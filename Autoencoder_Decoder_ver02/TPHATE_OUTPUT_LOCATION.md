# T-PHATE Plot 输出位置

## 📍 文件位置

根据你的运行，文件保存在：

**完整路径**: `/staging/groups/bhaskar_group/rho9/aadhitya_v1_test/`

**目录结构**:
```
/staging/groups/bhaskar_group/rho9/aadhitya_v1_test/
├── tphate_plots/          # 3D T-PHATE plots（按时间着色）
│   ├── AA83-7_tphate.png
│   ├── AAL839-6_tphate.png
│   ├── AB028-6_tphate.png
│   ├── AB91-1_tphate.png
│   └── AC264-1_tphate.png
└── curvature_plots/       # 3D T-PHATE plots（按曲率着色）
    ├── AA83-7_curvature.png
    ├── AAL839-6_curvature.png
    ├── AB028-6_curvature.png
    ├── AB91-1_curvature.png
    └── AC264-1_curvature.png
```

## 🔍 确认路径

在 CHTC 上运行：

```bash
# 查看完整路径
pwd
ls -la aadhitya_v1_test/

# 或者使用绝对路径
ls -lh /staging/groups/bhaskar_group/rho9/aadhitya_v1_test/
```

## 📥 下载到本地

```bash
# 在本地 Mac 上运行
scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/aadhitya_v1_test ~/Downloads/
```

## 📝 注意事项

- 文件保存在 **staging** 目录（`/staging/groups/bhaskar_group/rho9/`）
- 这是共享存储，可以被其他用户访问（如果有权限）
- 如果需要移动到 home 目录，可以使用：
  ```bash
  cp -r aadhitya_v1_test ~/
  ```






