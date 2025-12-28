#hello 
import cv2
import numpy as np
import os


def read_file_with_encoding(txt_file):
    """尝试多种编码读取文件"""
    encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16', 'gb2312']

    for encoding in encodings:
        try:
            with open(txt_file, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"✓ 使用 {encoding} 编码成功读取")
            return content, encoding
        except UnicodeDecodeError:
            continue

    # 如果所有编码都失败，使用二进制模式
    with open(txt_file, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    print("⚠ 使用二进制模式读取（可能丢失字符）")
    return content, 'binary'


def main():
    txt_file = "../data/pic3.txt"

    # 检查文件是否存在
    if not os.path.exists(txt_file):
        print(f"错误：文件 {txt_file} 不存在")
        print("请将像素值文件放在 ../data/ 目录下")
        return

    # 1. 读取文件（解决编码问题）
    content, encoding = read_file_with_encoding(txt_file)
    lines = content.splitlines()

    if not lines:
        print("错误：文件为空")
        return

    # 2. 解析第一行获取尺寸信息
    first_line = lines[0].strip()
    print(f"第一行内容: '{first_line}'")

    # 提取数字
    import re
    numbers = re.findall(r'\d+', first_line)

    if len(numbers) >= 2:
        height = int(numbers[0])
        width = int(numbers[1])
        channels = int(numbers[2]) if len(numbers) >= 3 else 1
        print(f"尺寸信息: {height}x{width}x{channels}")
    else:
        print("错误：第一行没有有效的尺寸信息")
        print("文件第一行应该是: 高度 宽度 [通道数]")
        print("例如: 256 256 1  # 256x256灰度图")
        print("或: 512 512 3   # 512x512彩色图")
        return

    # 3. 使用 np.zeros 创建矩阵
    if channels == 1:
        img_matrix = np.zeros((height, width), dtype=np.uint8)
    else:
        img_matrix = np.zeros((height, width, channels), dtype=np.uint8)

    print(f"创建矩阵: {img_matrix.shape}")

    # 4. 提取所有像素值
    all_pixels = []
    for line_num, line in enumerate(lines[1:], start=2):  # 从第二行开始
        line = line.strip()
        if line and not line.startswith('#'):  # 跳过空行和注释
            # 提取数字（支持小数和负数）
            numbers = re.findall(r'[-+]?\d*\.?\d+', line)
            for num_str in numbers:
                try:
                    pixel_value = int(float(num_str))
                    # 限制在0-255范围内
                    pixel_value = max(0, min(255, pixel_value))
                    all_pixels.append(pixel_value)
                except ValueError:
                    print(f"警告：第{line_num}行有无效数字: {num_str}")

    print(f"提取到 {len(all_pixels)} 个像素值")

    # 5. 检查像素数量
    expected_pixels = height * width * channels
    if len(all_pixels) != expected_pixels:
        print(f"⚠ 警告: 像素数量 {len(all_pixels)} ≠ 期望 {expected_pixels}")

        if len(all_pixels) > expected_pixels:
            print(f"  截取前 {expected_pixels} 个像素值")
            all_pixels = all_pixels[:expected_pixels]
        else:
            print(f"  用0填充到 {expected_pixels} 个像素值")
            all_pixels.extend([0] * (expected_pixels - len(all_pixels)))

    # 6. 将像素值填充到矩阵（简单方法）
    try:
        pixels_array = np.array(all_pixels, dtype=np.uint8)

        if channels == 1:
            img_matrix = pixels_array.reshape(height, width)
        else:
            img_matrix = pixels_array.reshape(height, width, channels)

        print("✓ 像素值填充完成")

    except Exception as e:
        print(f"❌ 填充失败: {e}")
        return

    # 7. 显示图像信息
    print(f"\n图像信息:")
    print(f"  形状: {img_matrix.shape}")
    print(f"  数据类型: {img_matrix.dtype}")
    print(f"  像素范围: [{img_matrix.min()}, {img_matrix.max()}]")

    # 8. 展示图像
    cv2.imshow(f"恢复的图像 {img_matrix.shape}", img_matrix)

    # 9. 等待按键
    print("\n按任意键保存图像，按ESC退出...")
    key = cv2.waitKey(0)

    # 10. 保存图像
    if key != 27:  # 不是ESC键
        cv2.imwrite("../data/restored.png", img_matrix)
        cv2.imwrite("../data/restored.jpg", img_matrix)
        print("✓ 图像已保存为PNG和JPG格式")

    cv2.destroyAllWindows()
    print("\n程序完成")


if __name__ == "__main__":
    main()