import cv2
import numpy as np

# 顔写真から鼻付近での暗い部分を取得するコード


image_path = "hoge.png"  # 顔写真のパスを指定してください
image = cv2.imread(image_path)


# Haar-like特徴分類器を使用して顔を検出
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 検出された顔の中から鼻の範囲を特定
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    # Haar-like特徴分類器を使用して鼻を検出
    nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
    noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 鼻の範囲内で暗い部分を赤い点で表示
    for (nx, ny, nw, nh) in noses:
        nose_roi_color = roi_color[ny:ny+nh, nx:nx+nw]
        hsv = cv2.cvtColor(nose_roi_color, cv2.COLOR_BGR2HSV)
        dark_pixels = hsv[..., 2] < 150  # 明度が50未満のピクセルを選択
        nose_roi_color[dark_pixels] = (0, 0, 255)  # 赤い点で表示

# 結果の表示
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()