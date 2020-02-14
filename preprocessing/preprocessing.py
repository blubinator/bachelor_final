import cv2
import numpy as np
from PIL import Image
import pytesseract

# resize
def resize(img):
    dsize = (1600, 1000)
    resized_img = cv2.resize(img, dsize)
    return resized_img

# show img
def show(name, img):
    cv2.imshow(name, resize(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_card(path): 
    # src img
    img = cv2.imread(path)
    img = resize(img)

    show("original", img)

    # img variants
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur_gray = cv2.GaussianBlur(hsv,(3, 3),cv2.BORDER_DEFAULT)

    edged = cv2.Canny(blur_gray, 30, 150)

    show("edged", edged)

    edged_blur = cv2.GaussianBlur(edged,(3, 3),cv2.BORDER_DEFAULT)

    show("edged_blur", edged_blur)

    # contours and rectangle detection
    (cnts, hier) = cv2.findContours(edged_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    show("contours", img)
    screenCnt = None
    area = 0
    area_index = 0
    index = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        # if len(approx) == 4:
        #     screenCnt = approx
        if area < cv2.contourArea(c) and len(approx) == 4:
            area = cv2.contourArea(c)
            screenCnt = approx

        index = index + 1

    rect = cv2.minAreaRect(screenCnt) # get a rectangle rotated to have minimal area
    box = cv2.boxPoints(rect) # get the box from the rectangle
    box = np.array(box, dtype="int")

    cv2.drawContours(img, screenCnt, -1, (0, 0, 255), 2)

    show("rectangle", img)

    # homography
    pts_src = np.array(screenCnt)

    pts_dst = np.array([[1600, 1000], [1600, 0], [0, 0], [0, 1000]])

    h, status = cv2.findHomography(pts_src, pts_dst)
        

    im_out = cv2.warpPerspective(img, h, (1600, 1000))
        
    show("Warped Source Image", im_out)
    return im_out

def extract_roi():
    print("asd")

if __name__ == "__main__":
    img = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures\\fuehrerschein0.jpg")

    print(pytesseract.image_to_string(img))
   

