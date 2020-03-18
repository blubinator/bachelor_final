import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance



def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        noisy = np.zeros(image.shape,np.uint8)
        thres = 1 - 0.05 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.rand()
                if rdn < 0.05:
                    noisy[i][j] = 0
                elif rdn > thres:
                    noisy[i][j] = 255
                else:
                    noisy[i][j] = image[i][j]
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def change_contrast(src, filename):
    img = Image.open(src)
    enhancer = ImageEnhance.Brightness(img)

    img_dark = enhancer.enhance(0.5)
    img_dark.save("dark_" + filename)

    img_light = enhancer.enhance(1.5)
    img_light.save("light_" + filename)


# for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\id"):

#     src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\id\\" + filename  

#     img = cv2.imread(src)

#     altered_img = noisy("gauss", img)
#     cv2.imwrite(src + "_gauss.jpg", altered_img)

#     altered_img = noisy("s&p", img)
#     cv2.imwrite(src + "_s&p.jpg", altered_img)

#     altered_img = noisy("poisson", img)
#     cv2.imwrite(src + "_poisson.jpg", altered_img)


for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\id"):

    src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\id\\" + filename  

    change_contrast(src, filename)

# for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\licence"):

#     src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\licence\\" + filename  

#     img = cv2.imread(src)

#     altered_img = noisy("gauss", img)
#     cv2.imwrite(src + "_gauss.jpg", altered_img)

#     altered_img = noisy("s&p", img)
#     cv2.imwrite(src + "_s&p.jpg", altered_img)

#     altered_img = noisy("poisson", img)
#     cv2.imwrite(src + "_poisson.jpg", altered_img)


for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\licence"):

    src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train\\licence\\" + filename    

    change_contrast(src, filename)

#     altered_img = noisy("speckle", img)
#     cv2.imwrite(src + "_speckle.png", altered_img)


