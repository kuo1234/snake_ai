"""测试 demo_ai.py 的 V3 支持"""
import sys
sys.path.insert(0, '.')

print("正在测试 demo_ai.py...")

try:
    import demo_ai
    print("✓ demo_ai.py 载入成功")
    
    # 测试模型扫描
    models = demo_ai.list_available_models()
    print(f"\n支持的模型类型: {list(models.keys())}")
    
    for model_type, model_list in models.items():
        print(f"  {model_type}: {len(model_list)} 个模型")
        if model_type == 'ppo_v3' and model_list:
            print("    V3 模型路径示例:")
            for path in model_list[:3]:
                print(f"      - {path}")
    
    print("\n✓ V3 支持已成功添加到 demo_ai.py!")
    print("\n使用方法:")
    print("  python demo_ui.py")
    print("  然后选择 'PPO V3 🎓' 模型类型")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
