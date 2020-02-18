import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import imutils
import boto3
import io
from io import BytesIO
import sys
import math

# resize


def resize(img):
    dsize = (1600, 900)
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

    # img variants
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise reduction
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    # blur whole image
    blur_gray = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)

    edged = cv2.Canny(blur_gray, 30, 150)

    edged_blur = cv2.GaussianBlur(edged, (3, 3), cv2.BORDER_DEFAULT)

    show("edged_blur", edged_blur)

    # contours and rectangle detection
    (cnts, hier) = cv2.findContours(edged_blur,
     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    # show("contours", img)
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

    cv2.drawContours(img, screenCnt, -1, (0, 0, 255), 2)

    show("rectangle", img)

    # homography
    pts_src = np.array(screenCnt)

    pts_dst = np.array([[1600, 900], [1600, 0], [0, 0], [0, 900]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(img, h, (1600, 900))

    show("Warped Source Image", im_out)
    return im_out

# def perform_ocr(img):
#     #blur = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)
#     img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#     # Convert to gray
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # # Apply dilation and erosion to remove some noise
#     kernel = np.ones((1, 1), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
#     #img = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)
#     img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#     show("ayyy", img)

#     pil_img = Image.fromarray(img, 'L')
#     print(pytesseract.image_to_string(pil_img, lang = 'deu'))
#     data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
#     h = pil_img.height
#     w = pil_img.width
#     n_boxes = len(data['level'])
#     for i in range(n_boxes):
#         (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     show("ayy", img)


def perform_ocr_aws(img):
    # Read document content    
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    byte_im = im_buf_arr.tobytes()

    # Amazon Textract client
    textract = boto3.client('textract')

    # Call Amazon Textract
    response = textract.detect_document_text(Document={'Bytes': byte_im})

    return response

def crop_blocks(img, resp):
    # Get the text blocks
    blocks = resp['Blocks']
    height, width = img.shape
    poly = []
    index = 0
    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        if block['BlockType'] == 'LINE':
            # draw polygon
            poly = []
            temp = []
            for pt in block['Geometry']['Polygon']:
                x = pt['X']*width
                y = pt['Y']*height
                temp = [int(x),int(y)]
                poly.append(temp)

                # crop
                crop_poly = np.array(poly)
                rect = cv2.boundingRect(crop_poly)
                x,y,w,h = rect
                croped = img[y:y+h, x:x+w].copy()

                ## (2) make mask
                crop_poly = crop_poly - crop_poly.min(axis=0)

                mask = np.zeros(croped.shape[:2], np.uint8)
                cv2.drawContours(mask, [crop_poly], -1, (255, 255, 255), -1, cv2.LINE_AA)

                ## (3) do bit-op
                dst = cv2.bitwise_and(croped, croped, mask=mask)

                ## (4) add the white background
                bg = np.ones_like(croped, np.uint8)*255
                cv2.bitwise_not(bg,bg, mask=mask)
                dst2 = bg+ dst

                cv2.imwrite("shaizze" + str(index) + ".png", dst2)
                index = index + 1

                
            
    poly = np.array(poly)
    poly = poly.reshape((-1,1,2))
    cv2.polylines(img, [poly], True, (0,255,255))

    

    # Display the image
    show("4ecke", img)

def extract_variable_lines(img, resp):
        # Get the text blocks
    blocks = resp['Blocks']
    height, width = img.shape

    poly = []

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        if block['BlockType'] == 'LINE':
            # draw polygon
            
            for pt in block['Geometry']['Polygon']:
                x = pt['X']*width
                y = pt['Y']*height
                temp = [int(x),int(y)]
                poly.append(temp)
            
    poly = np.array(poly)
    poly = poly.reshape((-1,1,2))
    # cv2.polylines(img, [poly], True, (0,255,255))
    cv2.fillPoly(img, [poly], 255)

    show("black bars", img)

    #cr
if __name__ == "__main__":
    ####### set tesseract path manually ########
    pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ############################################

    img = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures\\fuehrerschein0.jpg")
    resp = perform_ocr_aws(img)
    crop_blocks(img, resp)
    extract_variable_lines(img, resp)
