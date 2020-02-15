import cv2
import numpy as np
from PIL import Image
import pytesseract
import imutils

# resize
def resize(img):
    dsize = (1920, 1080)
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

    #show("original", img)

    # img variants
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur_gray = cv2.GaussianBlur(hsv,(3, 3),cv2.BORDER_DEFAULT)

    edged = cv2.Canny(blur_gray, 30, 150)

    #show("edged", edged)

    edged_blur = cv2.GaussianBlur(edged,(3, 3),cv2.BORDER_DEFAULT)

    #show("edged_blur", edged_blur)

    # contours and rectangle detection
    (cnts, hier) = cv2.findContours(edged_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    #show("contours", img)
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

    pts_dst = np.array([[1920, 1080], [1920, 0], [0, 0], [0, 1080]])

    h, status = cv2.findHomography(pts_src, pts_dst)
        

    im_out = cv2.warpPerspective(img, h, (1920, 1080))
        
    show("Warped Source Image", im_out)
    return im_out

def extract_roi():
    print("asd")

def perform_ocr(img):
    blur = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)

    pil_img = Image.fromarray(blur, 'RGB')
    print(pytesseract.image_to_string(pil_img, lang = 'deu'))
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    h = pil_img.height
    w = pil_img.width
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show("ayy", img)


if __name__ == "__main__":
    ####### set tesseract path manually ########
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ############################################

    img = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures\\fuehrerschein0.jpg")
    perform_ocr(img)
   

