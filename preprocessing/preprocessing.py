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
import re
import matplotlib.pyplot as plt


# resize


def resize(img):
    dsize = (1680, 1050)
    resized_img = cv2.resize(img, dsize)
    return resized_img

# show img
def show(name, img):
    #cv2.imshow(name, resize(img))
    cv2.imshow(name, img)
    cv2.resizeWindow(name, (1680, 1050))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# It's quite difficult. There is so little contrast, even the JPEG compression artifacts have more contrast than the object on the background.

# You would require a highly specialized deblock filter to eliminate the compression artifacts first. With knowledge about the block size of the used compression algorithm and the number of coefficients used per block you may be able to predict some of the edges.

# For edges you could predict in the previous step, you may try to filter these from the detected edges.

# All around the shirt in the photo, there are over-swings due to an excessively lossy compression. At the edge of each block, the over-swings form a hard edge which you also successfully detected.

# The block size is 8x8 pixels for JPEG (and hence also in this image), so every vertical or horizontal edge which falls directly onto position 8*n or 8n+1 X or Y is most likely just a compression artifact and can be ignored.

# This approach can only work though if the image hasn't been re-sampled after compression, respectively hasn't been re-compressed multiple times with potentially different block sizes. At that point, isolating the compression artifacts becomes nearly impossible.




def detect_card(path):
    # src img
    img = cv2.imread(path)
    img = resize(img)

    # img variants
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise reduction
    kernel = np.ones((1, 1), np.uint8)
    img_erode = cv2.erode(img_gray, kernel, iterations=1)
    img_dilate = cv2.dilate(img_erode, kernel, iterations=1)

    
    # blur whole image
    blur_gray = cv2.GaussianBlur(img_dilate, (3, 3), cv2.BORDER_DEFAULT)

    edged = cv2.Canny(blur_gray, 30, 150)

    edged_blur = cv2.GaussianBlur(edged, (3, 3), cv2.BORDER_DEFAULT)

    show("edged", edged)

    # contours and rectangle detection
    (cnts, hier) = cv2.findContours(edged_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    # show("contours", img)
    screenCnt = None
    area = 0
    area_index = 0
    index = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if area < cv2.contourArea(c) and len(approx) == 4:
            area = cv2.contourArea(c)
            screenCnt = approx

        index = index + 1

    cv2.drawContours(img, screenCnt, -1, (0, 0, 255), 2)

    show("rectangle", img)

    # detect orientation of card
    angle = math.degrees(math.atan2(screenCnt[0,:,1]-screenCnt[1,:,1], screenCnt[0,:,0]-screenCnt[1,:,0]))

    # homography
    pts_src = np.array(screenCnt)
    if angle > 0:
        pts_dst = np.array([[1680, 1050], [1680, 0], [0, 0], [0, 1050]])
 
    else:
        pts_dst = np.array([[0, 1050], [1680, 1050], [1600, 0], [0, 0]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(img, h, (1680, 1050))

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
    height, width, _ = img.shape
    
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

            # make mask
            crop_poly = crop_poly - crop_poly.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [crop_poly], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            # add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg + dst

            cv2.imwrite("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\preprocessing\\line" + str(index) + ".png", dst2)
            index = index + 1
          
            poly = np.array(poly)
            poly = poly.reshape((-1,1,2))
            cv2.polylines(img, [poly], True, (0,255,255))
            poly = []

    show("without blocks", img)
    return img




def extract_variable_lines(img, resp):
    # Get the text blocks
    blocks = resp['Blocks']
    height, width, _ = img.shape

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
            
            poly = np.array(poly)
            poly = poly.reshape((-1,1,2))
            # cv2.polylines(img, [poly], True, (0,255,255))
            cv2.fillPoly(img, [poly], (255, 255, 255))

    show("black bars", img)
    cv2.imwrite("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\preprocessing\\without_var.png", img)
    return img

def crop_face(img, cascPath):
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-50, y-100), (x+w+50, y+h+80), (255, 255, 255), cv2.FILLED)
        

    show("Faces found", img)
    cv2.imwrite("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\preprocessing\\no_face_square.png", img)

    # maybe test with edge detection


def idcard_check_nmbr(resp):
    blocks = resp['Blocks']

    for block in blocks:
        if block['BlockType'] == 'WORD':
            if re.match("^[LMNPRTVWXY][1234567890CFGHJKLMNPRTVWXYZ]{8}", block['Text']):
                print("length match")
    # maybe nummer mit prüfziffer auf rückseite checken!       


#def check_lines(img):




if __name__ == "__main__":
    ####### set tesseract path manually ########
    pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ############################################
    
    img = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures_idcard\\ausweis17.jpg")
    resp = perform_ocr_aws(img)
    img = crop_blocks(img, resp)
    img = extract_variable_lines(img, resp)
    img = crop_face(img, "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\haarcascade_frontalface_default.xml")
    idcard_check_nmbr(resp)
