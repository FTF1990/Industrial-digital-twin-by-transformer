#!/usr/bin/env python3
"""
Diagnostic script to test if Gradio interface can load
"""
import os
import sys

print("=" * 80)
print("🔍 测试Gradio界面加载")
print("=" * 80)

# Step 1: Check environment
print("\n1️⃣ 检查环境...")
print(f"   Python版本: {sys.version}")
print(f"   工作目录: {os.getcwd()}")
print(f"   data/ 文件夹存在: {os.path.exists('data')}")

if os.path.exists('data'):
    import glob
    csv_files = glob.glob('data/*.csv')
    print(f"   data/ 下的CSV文件: {len(csv_files)} 个")
    for f in csv_files[:5]:
        print(f"      - {f}")
    if len(csv_files) > 5:
        print(f"      ... 还有 {len(csv_files) - 5} 个文件")

# Step 2: Test imports
print("\n2️⃣ 测试导入...")
try:
    print("   导入 gradio...", end=" ")
    import gradio as gr
    print(f"✅ (版本: {gr.__version__})")
except Exception as e:
    print(f"❌\n   错误: {e}")
    sys.exit(1)

try:
    print("   导入 pandas...", end=" ")
    import pandas as pd
    print(f"✅ (版本: {pd.__version__})")
except Exception as e:
    print(f"❌\n   错误: {e}")
    sys.exit(1)

# Step 3: Test simple interface
print("\n3️⃣ 测试简单界面...")
try:
    def simple_func(x):
        return f"输入: {x}"

    demo = gr.Interface(fn=simple_func, inputs="text", outputs="text")
    print("   ✅ 简单界面创建成功")
except Exception as e:
    print(f"   ❌ 简单界面创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test main app import
print("\n4️⃣ 测试主应用导入...")
try:
    print("   导入 gradio_residual_tft_app...", end=" ")
    import gradio_residual_tft_app as app
    print("✅")
except Exception as e:
    print(f"❌\n   错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test main interface creation
print("\n5️⃣ 测试主界面创建...")
try:
    print("   调用 create_unified_interface()...")
    demo_main = app.create_unified_interface()
    print("   ✅ 主界面创建成功")
except Exception as e:
    print(f"   ❌ 主界面创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Summary
print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
print("\n📝 建议: 请尝试以下命令启动应用:")
print("   python gradio_residual_tft_app.py")
print("=" * 80)
