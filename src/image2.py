import cv2
import numpy as np
import os
import re


def restore_binary_image(txt_file):
    """
    从文本文件恢复二值图像（0和1）
    第一行格式: 图像信息: 宽度=161, 高度=108, 通道数=1, 类型=CV_8UC1
    后续行: 像素值（全是0或1）
    """
    print(f"正在恢复二值图像: {txt_file}")

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
    print(f"图像信息: '{first_line}'")

    # 提取关键参数
    width_match = re.search(r'宽度=(\d+)', first_line)
    height_match = re.search(r'高度=(\d+)', first_line)
    channels_match = re.search(r'通道数=(\d+)', first_line)

    if width_match and height_match and channels_match:
        width = int(width_match.group(1))  # 161
        height = int(height_match.group(1))  # 108
        channels = int(channels_match.group(1))  # 1

        print(f"✅ 解析成功:")
        print(f"   宽度: {width}")
        print(f"   高度: {height}")
        print(f"   通道数: {channels}")
        print(f"   类型: 二值图像 (0和1)")
    else:
        print("错误：第一行格式不正确")
        print("期望格式: 图像信息: 宽度=161, 高度=108, 通道数=1, 类型=CV_8UC1")
        return None

    # 检查是否是二值图像
    if channels != 1:
        print(f"⚠ 警告: 通道数={channels}，但期望为1（二值图像）")

    # 提取所有像素值（从第二行开始）
    print("\n正在提取二进制像素值...")
    binary_pixels = []
    pixel_count = 0

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if line and not line.startswith('#'):
            # 只提取0和1
            numbers = re.findall(r'[01]', line)
            for num_str in numbers:
                pixel_value = int(num_str)
                binary_pixels.append(pixel_value)
                pixel_count += 1

    print(f"提取到 {pixel_count} 个二进制像素值")

    # 检查像素值是否全是0和1
    unique_values = set(binary_pixels)
    if not unique_values.issubset({0, 1}):
        print(f"⚠ 警告: 发现非二进制值: {unique_values}")
        print("将强制转换为二进制 (非0值转为1)")
        binary_pixels = [1 if p != 0 else 0 for p in binary_pixels]

    # 计算期望的像素数量
    expected_pixels = height * width
    print(f"期望像素数量: {expected_pixels} (108 * 161 = {108 * 161})")

    # 检查像素数量
    if pixel_count != expected_pixels:
        print(f"⚠ 警告: 像素数量 {pixel_count} ≠ 期望 {expected_pixels}")

        if pixel_count > expected_pixels:
            print(f"  截取前 {expected_pixels} 个像素值")
            binary_pixels = binary_pixels[:expected_pixels]
        else:
            print(f"  用0填充到 {expected_pixels} 个像素值")
            binary_pixels.extend([0] * (expected_pixels - pixel_count))

    # 创建图像矩阵（将0和1转换为0和255以便显示）
    print("\n正在创建二值图像矩阵...")
    try:
        # 将0和1转换为0和255
        pixels_255 = [255 if p == 1 else 0 for p in binary_pixels]

        # 转换为numpy数组并重塑
        img_matrix = np.array(pixels_255, dtype=np.uint8)
        img_matrix = img_matrix.reshape((height, width))

        print(f"✅ 二值图像矩阵创建成功!")
        print(f"   形状: {img_matrix.shape}")
        print(f"   数据类型: {img_matrix.dtype}")
        print(f"   像素值: 0->黑色, 255->白色")

        return img_matrix

    except Exception as e:
        print(f"❌ 创建图像失败: {e}")
        return None


def main():
    # 文件路径
    txt_file = "../data/pic2.txt"  # 请根据实际情况修改文件名

    # 恢复图像
    binary_image = restore_binary_image(txt_file)

    if binary_image is not None:
        print(f"\n{'=' * 60}")
        print("二值图像恢复成功!")
        print(f"{'=' * 60}")

        # 显示详细信息
        print(f"图像尺寸: {binary_image.shape[1]}x{binary_image.shape[0]}")
        print(f"宽度: {binary_image.shape[1]}")
        print(f"高度: {binary_image.shape[0]}")

        # 统计0和255的数量
        count_0 = np.sum(binary_image == 0)
        count_255 = np.sum(binary_image == 255)
        total = binary_image.size

        print(f"\n二值统计:")
        print(f"  黑色像素(0): {count_0} ({count_0 / total * 100:.1f}%)")
        print(f"  白色像素(255): {count_255} ({count_255 / total * 100:.1f}%)")

        # 显示图像
        cv2.imshow(f"二值图像 {binary_image.shape[1]}x{binary_image.shape[0]}", binary_image)
        print("\n图像显示中...")
        print("按键说明:")
        print("  's' - 保存图像")
        print("  'q' - 退出")
        print("  其他任意键 - 继续显示")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):  # 保存图像
                # 确保输出目录存在
                output_dir = "E:/my_project/AirHUST_Vision_Tutorial/data"
                os.makedirs(output_dir, exist_ok=True)

                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                output_path_png = f"{output_dir}/{base_name}_restored.png"
                output_path_bmp = f"{output_dir}/{base_name}_restored.bmp"

                # 保存图像（推荐PNG或BMP格式，二值图像无损）
                success_png = cv2.imwrite(output_path_png, binary_image)
                success_bmp = cv2.imwrite(output_path_bmp, binary_image)

                if success_png:
                    print(f"✅ PNG格式保存成功: {output_path_png}")
                else:
                    print(f"❌ PNG格式保存失败")

                if success_bmp:
                    print(f"✅ BMP格式保存成功: {output_path_bmp}")
                else:
                    print(f"❌ BMP格式保存失败")

                # 同时保存原始二进制数据
                original_binary = (binary_image == 255).astype(np.uint8)
                binary_data_path = f"{output_dir}/{base_name}_binary.txt"
                np.savetxt(binary_data_path, original_binary, fmt='%d')
                print(f"✅ 原始二进制数据保存为: {binary_data_path}")

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