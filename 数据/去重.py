import pandas as pd
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# 桌面的“原数据.xlsx”路径
input_file_path = os.path.join(desktop_path, "原数据.xlsx")
# 去重后保存到桌面，文件名“去重后的数据.xlsx”
output_file_path = os.path.join(desktop_path, "去重后的数据.xlsx")

try:
    # 读取桌面的Excel文件
    df_original = pd.read_excel(input_file_path)
    print("✅ 成功读取桌面的原数据.xlsx")
    print(f"原始数据行数：{df_original.shape[0]} 行")

    # 检测并去重
    duplicate_count = df_original.duplicated(keep='first').sum()
    df_deduplicated = df_original.drop_duplicates(keep='first')
    print(f"检测到重复行数：{duplicate_count} 行")
    print(f"去重后数据行数：{df_deduplicated.shape[0]} 行")

    # 保存到桌面
    df_deduplicated.to_excel(output_file_path, index=False)
    print(f"✅ 去重完成！文件已保存到桌面：{output_file_path}")

except FileNotFoundError:
    print("❌ 错误：桌面上没找到“原数据.xlsx”，请确认文件在桌面且名字正确！")
except Exception as e:
    print(f"❌ 运行出错：{str(e)}")
    