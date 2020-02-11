import cv2
import numpy as np

# resize
def resize(img):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (800, 600)
    resized_img = cv2.resize(img, dsize)
    return resized_img

# show img
def show(name, img):
    cv2.imshow(name, resize(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# src img
img = cv2.imread("fuehrerschein1.jpg")
img = resize(img)

show("original", img)

# img variants
grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur_gray = cv2.GaussianBlur(grayScale,(3, 3),cv2.BORDER_DEFAULT)

edged = cv2.Canny(blur_gray, 30, 200)

show("edged", edged)

# contours and rectangle detection
(cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    # approximate
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # rectangle
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
print(screenCnt)
cv2.waitKey(0)

show("img", img)

# homography
pts_src = np.array(screenCnt)

im_dst = cv2.imread('book1.jpg')
pts_dst = np.array([[800, 600], [800, 0], [0, 0], [0, 600]])

h, status = cv2.findHomography(pts_src, pts_dst)
    

im_out = cv2.warpPerspective(img, h, (800, 600))
    
cv2.imshow("Warped Source Image", im_out)

cv2.waitKey(0)
