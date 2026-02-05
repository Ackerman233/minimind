import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    # 检查是否因为后缀带了 +cpu
    if "cpu" in torch.__version__:
        print("确认：你安装的是纯 CPU 版本的 Torch。")
    else:
        print("警告：安装了支持 CUDA 的版本，但驱动或工具链有问题。")