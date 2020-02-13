import cv2
import numpy as np

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

def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmax(diff)]
	rect[3] = pts[np.argmin(diff)]
	# return the ordered coordinates
	return rect
    
# src img
img = cv2.imread("C:\\Users\\tim.reicheneder\\Desktop\Bachelorthesis\\impl_final\\pictures\\fuehrerschein1.jpg")
img = resize(img)

show("original", img)

# img variants
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

blur_gray = cv2.GaussianBlur(hsv,(3, 3),cv2.BORDER_DEFAULT)

edged = cv2.Canny(blur_gray, 30, 150)

show("edged", edged)

# contours and rectangle detection
(cnts, hier) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cnts[-1], -1, (0, 255, 0), 2)
show("contours", img)
screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        screenCnt = approx

rect = cv2.minAreaRect(cnts[-1]) # get a rectangle rotated to have minimal area
box = cv2.boxPoints(rect) # get the box from the rectangle
box = np.array(box, dtype="int")

cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 2)

show("rectangle", img)

# homography
pts_src = np.array(screenCnt)

pts_dst = np.array([[1600, 900], [1600, 0], [0, 0], [0, 900]])

h, status = cv2.findHomography(pts_src, pts_dst)
    

im_out = cv2.warpPerspective(img, h, (1600, 900))
    
show("Warped Source Image", im_out)

# extract variable shit

