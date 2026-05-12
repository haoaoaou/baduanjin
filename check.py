try:
    print("正在检查 DeepFace...")
    import deepface

    print("✅ DeepFace 加载成功！")

    print("正在检查 Ultralytics...")
    import ultralytics

    print("✅ Ultralytics 加载成功！")

except Exception as e:
    print("\n❌ 依然报错，具体原因如下：")
    print(e)