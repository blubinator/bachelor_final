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
import operator
import re


# resize
def resize(img):
    dsize = (1920, 1080)
    resized_img = cv2.resize(img, dsize)
    return resized_img

# show img
def show(name, img):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, (1680, 1050))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ger_id_dict = ["BUNDESREPUBLIK DEUTSCHLAND", "REPUBLIQUE FEDERAL", "PERSONALAUSWEIS", "IDENTITY CARD"]

# It's quite difficult. There is so little contrast, even the JPEG compression artifacts have more contrast than the object on the background.

# You would require a highly specialized deblock filter to eliminate the compression artifacts first. With knowledge about the block size of the used compression algorithm and the number of coefficients used per block you may be able to predict some of the edges.

# For edges you could predict in the previous step, you may try to filter these from the detected edges.

# All around the shirt in the photo, there are over-swings due to an excessively lossy compression. At the edge of each block, the over-swings form a hard edge which you also successfully detected.

# The block size is 8x8 pixels for JPEG (and hence also in this image), so every vertical or horizontal edge which falls directly onto position 8*n or 8n+1 X or Y is most likely just a compression artifact and can be ignored.

# This approach can only work though if the image hasn't been re-sampled after compression, respectively hasn't been re-compressed multiple times with potentially different block sizes. At that point, isolating the compression artifacts becomes nearly impossible.




def detect_card(path):
    # src img
    img = cv2.imread(path)
    #img = resize(img)

    # img variants
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # noise reduction
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
    kernel = np.ones((3, 3), np.uint8)
    img_erode = cv2.erode(img_gray, (5, 5), iterations=1)
    img_dilate = cv2.dilate(img_erode, (3, 3), iterations=1)
    _, img_thr = cv2.threshold(img_dilate,127,255,cv2.THRESH_BINARY_INV)
    #show("img_thr", img_thr)
  
    # blur whole image
    blur_gray = cv2.GaussianBlur(img_thr, (5, 5), cv2.BORDER_DEFAULT)

    # thresholds for canny
    v = np.median(blur_gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))

    #detect edges
    edged_gray = cv2.Canny(blur_gray, lower, upper)

    blur = cv2.GaussianBlur(edged_gray, (9, 9), cv2.BORDER_DEFAULT)
    _, img_thr = cv2.threshold(blur,1,127,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img_thr, (5, 5), cv2.BORDER_DEFAULT)
    #show("img_thr", blur)

    # contours and rectangle detection
    (cnts, hier) = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 9)

    #show("contours", img)
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

    cv2.drawContours(img, screenCnt, -1, (0, 0, 255), 10)

    show("rectangle", img)

    # detect orientation of card
    screenCnt = np.sort(screenCnt, 1)
    ## get corners
    temp_array = screenCnt.copy()
    temp_array = sorted(temp_array, key=lambda x:x[:,1])
    upper_index = temp_array[0][0][1]
    lower_index = temp_array[1][0][1]
    

    angle = math.degrees(math.atan2(temp_array[1][0][1]-temp_array[0][0][1], temp_array[1][0][0]-temp_array[0][0][0]))
    print("found angle of : " + str(angle))

    # homography
    pts_src = np.array(temp_array)
    if angle > 90:
        pts_dst = np.array([[1680, 0], [0, 0],  [1680, 1050], [0, 1050]])
    else:
        pts_dst = np.array([[0, 0], [1680, 0], [0, 1050], [1680, 1050]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(img, h, (1680, 1050))

    #show("Warped Source Image", im_out)

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
    is_success, im_buf_arr = cv2.imencode(".png", img)
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
            if block['Text'].isupper() or re.match(r"[\d]{2}.[\d]{2}.[\d]{4}", block['Text']) or re.match(r"[0-9]+", block['Text']):
                if not any(block['Text'] in s for s in ger_id_dict):
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
        
            if block['Text'].isupper() or re.match(r"[\d]{2}.[\d]{2}.[\d]{4}", block['Text']) or re.match(r"[0-9]+", block['Text']):
                if not any(block['Text'] in s for s in ger_id_dict):
                    for pt in block['Geometry']['Polygon']:
                        x = pt['X']*width
                        y = pt['Y']*height
                        temp = [int(x),int(y)]
                        poly.append(temp)    

                    poly = np.array(poly)
                    poly = poly.reshape((-1,1,2))
                    # cv2.polylines(img, [poly], True, (0,255,255))
                    cv2.fillPoly(img, [poly], (255, 255, 255))

    show("white bars", img)
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
        #cv2.rectangle(img, (x-50, y-100), (x+w+50, y+h+80), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (x-50, y-100), (x+w+50, y+h+80), (255, 255, 255), cv2.FILLED)
        

    show("Faces found", img)
    cv2.imwrite("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\preprocessing\\no_face_square.png", img)

    # maybe use edge detection inside the rectangle


def idcard_check_nmbr(resp, resp_back):
    blocks = resp['Blocks']
    blocks_back = resp_back['Blocks']

    id_nmbr_front = ""
    id_nmbr_back = ""

    for block in blocks:
        if block['BlockType'] == 'WORD':
            if re.match("^[LMNPRTVWXY][1234567890CFGHJKLMNPRTVWXYZ]{8}", block['Text']):
                id_nmbr_front = block['Text']
                print("length match")

    # maybe nummer mit prüfziffer auf rückseite checken!
    for block in blocks_back:
        if block['BlockType'] == 'WORD':
            if re.match("^[LMNPRTVWXY][1234567890CFGHJKLMNPRTVWXYZ]{9}", block['Text']):
                id_nmbr_back = block['Text']
                print("length match back")
    if id_nmbr_back == "" or id_nmbr_front == "":
        print("Found no Id Numbers")
    else:
        print("found the numbers : " + id_nmbr_front + " and " + id_nmbr_back)

        weight = 7
        nmbr_sum = 0
        for c in id_nmbr_front:
            if weight == 7:
                nmbr_sum += get_alphab_nmbr(c)
                weight = 3

            if weight == 3:
                weight = 1
                nmbr_sum += get_alphab_nmbr(c)
            if weight == 1:
                weight = 7
                nmbr_sum += get_alphab_nmbr(c)

            if id_nmbr_back[len(id_nmbr_back) - 1] == int(str(nmbr_sum)[-1:]):
                print('id is valid')
            else:
                print('id is false')

def get_alphab_nmbr(char):
    if str.isdigit(char):
        return ord(char) - 96 + 9
    else:
        return char





def check_lines():
    # with hu moments evtl einzelne buchstaben ausschneiden
    img = cv2.imread("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\preprocessing\\line8.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    show("thresh line", th)

    moments = cv2.moments(th)
    huMoments = cv2.HuMoments(moments)

    for i in range(len(huMoments)):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))  
        print(huMoments[i])
    



if __name__ == "__main__":
    ####### set tesseract path manually ########
    pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ############################################
    
    # front
    img = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures_idcard\\ausweis17.png")
    resp = perform_ocr_aws(img)
    img = crop_blocks(img, resp)
    img = extract_variable_lines(img, resp)
    img = crop_face(img, "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\haarcascade_frontalface_default.xml")

    # back
    img_back = detect_card("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures_idcard\\ausweis_rueckseite3.png")
    resp_back = perform_ocr_aws(img_back)


    idcard_check_nmbr(resp, resp_back)
    check_lines()
