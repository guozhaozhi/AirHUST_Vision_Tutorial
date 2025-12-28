#hello 
import cv2
import numpy as np


def main():
    # 1. 读取原始图像并分析其尺寸与色彩模式
    img = cv2.imread("../data/pic3_restored.png")

    if img is None:
        print("错误：无法读取图像")
        return

    # 分析图像尺寸与色彩模式
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    print("原始图像分析:")
    print(f"  尺寸: {width}x{height}")
    print(f"  色彩模式: {channels}通道")
    print(f"  数据类型: {img.dtype}")

    # 计算中心点
    center = (width / 2.0, height / 2.0)

    # 2. 绕图像中心逆时针旋转30度，将结果保存为pic1
    print("\n步骤1: 逆时针旋转30度")
    # 逆时针旋转30度（角度为正）
    rotation_matrix1 = cv2.getRotationMatrix2D(center, 30, 1.0)
    pic1 = cv2.warpAffine(img, rotation_matrix1, (width, height), flags=cv2.INTER_LINEAR)

    # 保存pic1
    cv2.imwrite("../data/pic1.jpg", pic1)
    print("  已保存为: ../data/pic1.jpg")

    # 3. 基于pic1，再绕图像中心顺时针旋转30度，将结果保存为pic2
    print("\n步骤2: 顺时针旋转30度")
    # 顺时针旋转30度（角度为负）
    rotation_matrix2 = cv2.getRotationMatrix2D(center, -30, 1.0)
    pic2 = cv2.warpAffine(pic1, rotation_matrix2, (width, height), flags=cv2.INTER_LINEAR)

    # 保存pic2
    cv2.imwrite("../data/pic2.jpg", pic2)
    print("  已保存为: ../data/pic2.jpg")

    # 4. 比较并验证pic2与原始图像的差异
    print("\n步骤3: 比较差异")

    # 计算绝对差异
    diff = cv2.absdiff(img, pic2)

    # 转换为灰度图计算总差异
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    total_diff = np.sum(diff_gray)
    max_diff = np.max(diff_gray)
    mean_diff = np.mean(diff_gray)

    print(f"  总差异值: {total_diff}")
    print(f"  最大像素差异: {max_diff}")
    print(f"  平均像素差异: {mean_diff:.2f}")

    # 显示图像
    cv2.imshow("1. 原始图像", img)
    cv2.imshow("2. pic1 (逆时针30度)", pic1)
    cv2.imshow("3. pic2 (再顺时针30度)", pic2)
    cv2.imshow("4. 差异图", diff)

    print("\n显示图像中...")
    print("按任意键继续")
    cv2.waitKey(0)

    # 5. 如果想要完全消除这个差异，该怎么做？
    print("\n步骤4: 分析如何完全消除差异")
    print("差异原因: 旋转操作会丢失边缘像素信息，因为图像是矩形而非圆形")
    print("每次旋转都会进行插值计算，导致像素信息有微小损失")
    print("\n完全消除差异的方法:")
    print("1. 使用无损旋转方法（如保持原图所有像素）")
    print("2. 增加旋转后图像的尺寸，避免裁剪")
    print("3. 使用更高精度的插值方法")
    print("4. 先放大图像，旋转后再裁剪回原尺寸")

    cv2.destroyAllWindows()
    print("\n程序完成")


if __name__ == "__main__":
    main()