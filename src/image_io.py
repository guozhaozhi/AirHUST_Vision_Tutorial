import cv2
import numpy as np
import os


def main():
    print(f"OpenCV Version: {cv2.__version__}")

    # 1. 将图像读取为cv2格式（相当于C++的cv::Mat）
    input_path = "../data/hesiqi.png"  # 可以是.jpg/.png/.jpeg格式
    # 或者使用其他格式测试
    # input_path = "../data/hesiqi.png"
    # input_path = "../data/hesiqi.jpeg"

    # 读取图像 - cv2.imread返回的就是类似cv::Mat的numpy数组
    image = cv2.imread(input_path)

    # 检查图像是否成功读取
    if image is None:
        print(f"错误：无法读取图像 {input_path}")
        print("请检查：")
        print(f"1. 文件路径: {os.path.abspath(input_path)}")
        print(f"2. 文件是否存在: {os.path.exists(input_path)}")
        return

    # 显示图像信息
    print("✅ 图像读取成功！")
    print(f"   文件路径: {input_path}")
    print(f"   尺寸: {image.shape}")  # (高度, 宽度, 通道数)
    print(f"   数据类型: {image.dtype}")
    print(f"   总像素数: {image.size}")

    # 2. 基于OpenCV展示这张图像
    window_name = "OpenCV图像显示 - 按任意键继续"
    cv2.imshow(window_name, image)
    print("\n📺 图像已显示，按任意键继续...")
    print("   可以尝试：")
    print("   - 按 's' 保存图像")
    print("   - 按 'q' 或 ESC 退出")
    print("   - 按其他任意键进行图像处理")

    # 等待按键
    key = cv2.waitKey(0)

    # 3. 随便做点什么操作
    print("\n🔄 进行一些图像处理操作...")

    # 操作1: 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("   1. 已转换为灰度图像")

    # 操作2: 调整图像大小
    height, width = image.shape[:2]
    new_width, new_height = 400, 300
    resized_image = cv2.resize(image, (new_width, new_height))
    print(f"   2. 已调整大小: {width}x{height} -> {new_width}x{new_height}")

    # 操作3: 旋转图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转45度
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    print("   3. 已旋转45度")

    # 操作4: 高斯模糊
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    print("   4. 已应用高斯模糊")

    # 操作5: 边缘检测
    edges = cv2.Canny(image, 100, 200)
    print("   5. 已进行边缘检测")

    # 显示所有处理结果
    cv2.imshow("1. 灰度图像", gray_image)
    cv2.imshow("2. 调整大小", resized_image)
    cv2.imshow("3. 旋转45度", rotated_image)
    cv2.imshow("4. 高斯模糊", blurred_image)
    cv2.imshow("5. 边缘检测", edges)

    print("\n🖼️  所有处理结果已显示")
    print("   按 's' 保存图像，按 'q' 退出")

    # 交互式保存选项
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):  # 按 's' 保存图像
            # 保存原始图像为不同格式
            cv2.imwrite("../data/output_original.jpg", image)
            cv2.imwrite("../data/output_original.png", image)
            print("💾 已保存原始图像为JPG和PNG格式")

            # 保存处理后的图像
            cv2.imwrite("../data/output_gray.jpg", gray_image)
            cv2.imwrite("../data/output_resized.png", resized_image)
            cv2.imwrite("../data/output_rotated.jpg", rotated_image)
            cv2.imwrite("../data/output_blurred.jpg", blurred_image)
            cv2.imwrite("../data/output_edges.png", edges)

            print("💾 已保存所有处理后的图像")
            print("   保存位置: ../data/ 目录")
            print("   格式: .jpg 和 .png")

        elif key == ord('q') or key == 27:  # 按 'q' 或 ESC 退出
            print("👋 退出程序")
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()