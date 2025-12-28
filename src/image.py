import cv2
import numpy as np
import os
import re


def restore_color_image(txt_file):
    """
    从文本文件恢复彩色图像
    第一行格式: 图像信息: 宽度=1440, 高度=1920, 通道数=3, 类型=CV_8UC3
    后续行: 像素值（BGR顺序）
    """
    print(f"正在恢复彩色图像: {txt_file}")

    # 检查文件是否存在
    if not os.path.exists(txt_file):
        print(f"错误：文件不存在 {txt_file}")
        return None

    # 读取文件（处理编码问题）
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(txt_file, 'r', encoding='gbk') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(txt_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')

    lines = content.splitlines()

    if not lines:
        print("错误：文件为空")
        return None

    # 解析第一行获取图像信息
    first_line = lines[0].strip()
    print(f"第一行信息: '{first_line}'")

    # 提取关键参数
    width_match = re.search(r'宽度=(\d+)', first_line)
    height_match = re.search(r'高度=(\d+)', first_line)
    channels_match = re.search(r'通道数=(\d+)', first_line)

    if width_match and height_match and channels_match:
        width = int(width_match.group(1))  # 1440
        height = int(height_match.group(1))  # 1920
        channels = int(channels_match.group(1))  # 3

        print(f"✅ 解析成功:")
        print(f"   宽度: {width}")
        print(f"   高度: {height}")
        print(f"   通道数: {channels}")
    else:
        print("错误：第一行格式不正确")
        print("期望格式: 图像信息: 宽度=1440, 高度=1920, 通道数=3, 类型=CV_8UC3")
        return None

    # 检查尺寸是否正确（根据您的信息应该是1920x1440）
    if width == 1440 and height == 1920 and channels == 3:
        print("✅ 尺寸匹配: 1920x1440x3 (彩色图像)")
    else:
        print(f"⚠ 注意: 尺寸为 {height}x{width}x{channels} (可能不是标准RGB)")

    # 提取所有像素值（从第二行开始）
    print("\n正在提取像素值...")
    all_pixels = []
    pixel_count = 0

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if line and not line.startswith('#'):
            # 提取所有数字
            numbers = re.findall(r'[-+]?\d*\.?\d+', line)
            for num_str in numbers:
                try:
                    pixel_value = int(float(num_str))
                    # 限制在0-255范围内（确保是有效像素值）
                    pixel_value = max(0, min(255, pixel_value))
                    all_pixels.append(pixel_value)
                    pixel_count += 1
                except ValueError:
                    print(f"警告：第{line_num}行有无效数字: {num_str}")

    print(f"提取到 {pixel_count} 个像素值")

    # 计算期望的像素数量
    expected_pixels = height * width * channels
    print(f"期望像素数量: {expected_pixels} (1920 * 1440 * 3 = {1920 * 1440 * 3})")

    # 检查像素数量
    if pixel_count != expected_pixels:
        print(f"⚠ 警告: 像素数量 {pixel_count} ≠ 期望 {expected_pixels}")

        if pixel_count > expected_pixels:
            print(f"  截取前 {expected_pixels} 个像素值")
            all_pixels = all_pixels[:expected_pixels]
        else:
            print(f"  用0填充到 {expected_pixels} 个像素值")
            all_pixels.extend([0] * (expected_pixels - pixel_count))

    # 创建图像矩阵
    print("\n正在创建图像矩阵...")
    try:
        # 将像素列表转换为numpy数组
        pixels_array = np.array(all_pixels, dtype=np.uint8)

        # 重塑为正确的形状: (高度, 宽度, 通道数)
        img_matrix = pixels_array.reshape((height, width, channels))

        print(f"✅ 图像矩阵创建成功!")
        print(f"   形状: {img_matrix.shape}")
        print(f"   数据类型: {img_matrix.dtype}")
        print(f"   像素范围: [{img_matrix.min()}, {img_matrix.max()}]")

        return img_matrix

    except Exception as e:
        print(f"❌ 创建图像失败: {e}")
        return None


def main():
    # 文件路径
    txt_file = "../data/pic1.txt"  # 请根据实际情况修改文件名

    # 恢复图像
    color_image = restore_color_image(txt_file)

    if color_image is not None:
        print(f"\n{'=' * 60}")
        print("图像恢复成功!")
        print(f"{'=' * 60}")

        # 显示详细信息
        print(f"图像尺寸: {color_image.shape}")
        print(f"宽度: {color_image.shape[1]}")
        print(f"高度: {color_image.shape[0]}")
        print(f"通道数: {color_image.shape[2]}")

        # 显示每个通道的统计信息
        if color_image.shape[2] == 3:
            print("\n各通道统计:")
            channels_names = ['Blue', 'Green', 'Red']
            for i in range(3):
                channel = color_image[:, :, i]
                print(f"  {channels_names[i]}通道: [{channel.min()}, {channel.max()}] 均值: {channel.mean():.1f}")

        # 显示图像
        cv2.imshow(f"恢复的彩色图像 {color_image.shape[1]}x{color_image.shape[0]}", color_image)
        print("\n图像显示中...")
        print("按键说明:")
        print("  's' - 保存图像")
        print("  'q' - 退出")
        print("  其他任意键 - 继续")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):  # 保存图像
                # 确保输出目录存在
                output_dir = "E:/my_project/AirHUST_Vision_Tutorial/data"
                os.makedirs(output_dir, exist_ok=True)

                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                output_path_png = f"{output_dir}/{base_name}_restored.png"
                output_path_jpg = f"{output_dir}/{base_name}_restored.jpg"

                # 保存图像
                success_png = cv2.imwrite(output_path_png, color_image)
                success_jpg = cv2.imwrite(output_path_jpg, color_image)

                if success_png:
                    print(f"✅ PNG格式保存成功: {output_path_png}")
                else:
                    print(f"❌ PNG格式保存失败")

                if success_jpg:
                    print(f"✅ JPG格式保存成功: {output_path_jpg}")
                else:
                    print(f"❌ JPG格式保存失败")

            elif key == ord('q') or key == 27:  # 退出
                print("退出程序")
                break

        # 关闭窗口
        cv2.destroyAllWindows()
        print("\n🎉 程序完成!")
    else:
        print("图像恢复失败")


if __name__ == "__main__":
    main()