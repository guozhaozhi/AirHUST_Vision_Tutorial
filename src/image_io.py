# 检查OpenCV是否正确安装
import cv2
print(f"OpenCV Version: {cv2.__version__}")

# 基本的图像读取和显示
img = cv2.imread("../data/hesiqi.png")
cv2.imshow("Image", img)
cv2.waitKey(0)

