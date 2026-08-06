import cv2
import os
import sys

# 1. 检查命令行参数是否足够
# sys.argv[0] 是脚本名，[1] 是第一个参数，[2] 是第二个参数
if len(sys.argv) < 3:
    print("使用方法: python script.py <输入路径> <输出路径>")
    print("示例: python script.py ./denoised ./darker")
    sys.exit(1)

input_root = sys.argv[1]
output_root = sys.argv[2]

# 确保输出目录存在
os.makedirs(output_root, exist_ok=True)

# 调整参数
alpha = 0.5
beta = 1

print(f"正在从 {input_root} 读取并处理...")

for entry in os.scandir(input_root):
    if not entry.is_file():
        continue

    filename = entry.name
    if filename.startswith('.') or not filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        continue

    image_path = entry.path
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: 无法打开图像 {filename}.")
        continue

    # 应用调整 (注意：convertScaleAbs 会强制转为 8-bit)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    output_path = os.path.join(output_root, filename)
    cv2.imwrite(output_path, adjusted)
    print(f"Processed: {filename}")

print("\n--- 处理完成！ ---")