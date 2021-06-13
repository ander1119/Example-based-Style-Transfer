import os, cv2

# l = os.listdir("./inputImage/styleImage")
# for name in l:
#     if name.endswith(".png"):
#         img = cv2.imread("./inputImage/styleImage/"+name)
#         img = cv2.resize(img, (640, 800))
#         cv2.imwrite("./inputImage/styleImage/"+name, img)

# l = os.listdir("./inputImage/targetImage")

# for name in l:
#     if name.endswith(".png"):
#         img = cv2.imread("./inputImage/targetImage/"+name)
#         img = cv2.resize(img, (640, 800))
#         cv2.imwrite("./inputImage/targetImage/"+name, img)

img = cv2.imread("/Users/qiuyuyang/Desktop/class-3-2/DIP/proposal/code/Example-Based-Synthesis-of-Stylized-Facial-Animations/inputImage/targetImage/target_chai.jpg")
img = cv2.resize(img, (640, 800))
cv2.imwrite("/Users/qiuyuyang/Desktop/class-3-2/DIP/proposal/code/Example-Based-Synthesis-of-Stylized-Facial-Animations/inputImage/targetImage/target_chai.png", img)