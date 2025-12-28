import cv2
import numpy as np


def fix_specific_problem():
    """
    根据您的分析结果进行修复
    问题：帽子区域绿色最强（应该是红色最强）
    修复：交换G和R通道（BRG -> BGR）
    """

    # 读取图像
    img = cv2.imread("../data/processed_lena.jpg")
    if img is None:
        print("错误：无法读取图像")
        return

    print("分析结果:")
    print("- 帽子区域：绿色通道最强（170.0）")
    print("- 但帽子应该是红色的")
    print("- 推测：G和R通道交换了（BRG顺序）")
    print("\n修复方案：交换G和R通道")

    # 分离通道
    b, g, r = cv2.split(img)

    # 修复：交换G和R通道
    # 原来的顺序：B, G, R
    # 正确的顺序：B, R, G （交换G和R）
    restored = cv2.merge([b, r, g])  # BRG -> BGR

    # 验证修复效果
    print("\n验证修复效果:")
    height, width = img.shape[:2]
    hat_region = restored[0:height // 4, width // 4:3 * width // 4]
    b_hat, g_hat, r_hat = cv2.split(hat_region)

    print(f"修复后帽子区域:")
    print(f"  蓝色均值: {b_hat.mean():.1f}")
    print(f"  绿色均值: {g_hat.mean():.1f}")
    print(f"  红色均值: {r_hat.mean():.1f}")

    # 显示对比
    cv2.imshow("1. 原始图像 (G和R交换)", img)
    cv2.imshow("2. 修复后图像 (BGR顺序)", restored)

    # 显示差异
    diff = cv2.absdiff(img, restored)
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("3. 修复差异", diff_normalized)

    print("\n按任意键保存修复后的图像...")
    cv2.waitKey(0)

    # 保存
    output_path = "../data/lena_correct_fixed.jpg"
    cv2.imwrite(output_path, restored)
    print(f"✅ 修复后的图像已保存到: {output_path}")

    cv2.destroyAllWindows()


# 运行修复
fix_specific_problem()