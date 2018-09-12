#/usr/bin/python3

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from PIL import ImageTk, Image

import pytesseract
import os

import math
import numpy as np
import cv2
import time
from imutils.object_detection import non_max_suppression
from random import randrange, uniform

from platform import system

if system().lower() == "windows":
  pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def placeimage(image1, image2, x, y, w, h):
    image2 = cv2.resize(image2, (w, h))

    y1, y2 = y, y + image2.shape[0]
    x1, x2 = x, x + image2.shape[1]

    alpha_s = image2[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        image1[y1:y2, x1:x2, c] = (alpha_s * image2[:, :, c] +
                                  alpha_l * image1[y1:y2, x1:x2, c])
    return image1


def noise(image, hsv, mean, var):
    imagecv = np.array(image)
    if hsv:
        imagecv = cv2.cvtColor(imagecv, cv2.COLOR_RGB2HSV)

    row,col,ch = imagecv.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = imagecv + gauss
    if hsv:
        noisy = cv2.cvtColor(noisy.astype('uint8'), cv2.COLOR_HSV2RGB)
    image = Image.fromarray(noisy.astype('uint8'))
    return image


def ripple(image, xA, xw, yA, yw):
    imagecv = np.array(image)
    if xA == 0:
        Ax = 0
    else:
        Ax = imagecv.shape[1] / xA
    wx = xw / imagecv.shape[0]
    if yA == 0:
        Ay = 0
    else:
        Ay = imagecv.shape[0] / yA
    wy = yw / imagecv.shape[1]

    shiftx = lambda x: Ax * np.cos(2.0 * np.pi * x * wx)
    shifty = lambda x: Ay * np.cos(2.0 * np.pi * x * wy)

    for i in range(imagecv.shape[0]):
        imagecv[i, :] = np.roll(imagecv[i, :], int(shiftx(i)), 0)
    for i in range(imagecv.shape[1]):
        imagecv[:, i] = np.roll(imagecv[:, i], int(shifty(i)), 0)
    image = Image.fromarray(imagecv.astype('uint8'))
    return image

def Bulge(image, X, Y, R):
  img = np.array(image)

  map_x = np.zeros(img.shape[:2],np.float32)
  map_y = np.zeros(img.shape[:2],np.float32)
  rows,cols = img.shape[:2]
  R = 50

  for j in range(rows):
    for i in range(cols):
      map_x.itemset((j,i),i)
      map_y.itemset((j,i),j)

  for j in range(rows):
    for i in range(cols):
      r = math.sqrt(math.pow(i - X, 2) + math.pow(j - Y, 2))
      if r > R:
        continue
      a = 0
      if(j != Y):
        a = math.atan((i - X)/(j - Y))
      rn = -math.pow(r,2.5)/(10*R)
      if j < Y:
        map_x.itemset((j,i), rn*math.sin(a) + X)
        map_y.itemset((j,i), rn*math.cos(a) + Y)
      else:
        map_x.itemset((j,i), rn*math.sin(-a) + X)
        map_y.itemset((j,i), -rn*math.cos(a) + Y)

  dst = cv2.remap(img,map_x,map_y,cv2.INTER_LINEAR)
  return Image.fromarray(dst.astype('uint8'))

def Bemoji (imagesrc):
    image = np.array(imagesrc)

    # set B emoji scaling
    scalefactor = 2
    scalevar = (scalefactor - 1)/2

    # load the input image and grab the image dimensions
    min_conf = 0.1
    eastpath = "frozen_east_text_detection.pb"
    orig = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (640, 640)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(eastpath)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_conf:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    Bimage = cv2.imread("B.png", -1)
    b, g, r, a = cv2.split(Bimage)
    Bimage = cv2.merge((r, g, b, a))
    print("[INFO] loading Tesseract...")
    start = time.time()
    for (startX, startY, endX, endY) in boxes:
        roi = image[startY:endY, startX:endX]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        preprocess = "thresh"
        # check to see if we should apply thresholding to preprocess the
        # image
        if preprocess == "thresh":
            gray = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise
        elif preprocess == "blur":
            gray = cv2.medianBlur(gray, 3)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())

        if gray is not None:
            cv2.imwrite(filename, gray)
            text = pytesseract.image_to_boxes(Image.open(filename))
            os.remove(filename)
        else:
            text = ""

        print(text)
        text2 = str.split(text)

        if text2:
            dX = [int(text2[x*6 + 1]) for x in range (0, (int(len(text2)/6)))]
            dY = [int(text2[x * 6 + 2]) for x in range (0, (int(len(text2)/6)))]
            dW = [int(text2[x * 6 + 3])-int(text2[x*6 + 1]) for x in range (0, (int(len(text2)/6)))]
            dH = [int(text2[x * 6 + 4])-int(text2[x * 6 + 2]) for x in range (0, (int(len(text2)/6)))]
            #print(str(dW))
            startX = [int((startX + dX[x]) * rW) for x in range (0, (int(len(text2)/6)))]
            startY = [int((startY + dY[x]) * rH) for x in range (0, (int(len(text2)/6)))]

            letter = [text2[x * 6] for x in range (0, (int(len(text2)/6)))]

            for x in range (0, (int(len(text2)/6))):
                if (letter[x] == "G") | (letter[x] == "g") | (letter[x] == "B") | (letter[x] == "b"):
                    exp = 0
                    try:
                        placeimage(orig, Bimage, startX[x]-int(scalevar*(dW[x]*rH)),
                                   startY[x]-int(scalevar*(dH[x]*rW)), int(dW[x]*rH)*scalefactor,
                                   int(dH[x]*rW)*scalefactor)
                    except ValueError:
                        exp = 1
                    if exp:
                        scalecount = 1.9
                        switch = 0
                        while scalecount > 0:
                            try:
                                placeimage(orig, Bimage, startX[x] - int(((scalecount - 1) / 2) * (dW[x] * rH)),
                                           startY[x] - int(((scalecount - 1) / 2) * (dH[x] * rW)),
                                           int(dW[x] * rH * scalecount),
                                           int(dH[x] * rW * scalecount))
                            except ValueError:
                                scalecount = scalecount - 0.1
                            else:
                                break


    end = time.time()
    print("[INFO] letter detection took {:.6f} seconds".format(end - start))
    imagesrc = Image.fromarray(orig.astype('uint8'))
    return imagesrc

def facereg (imagesrc):
    # Get user supplied values
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = np.array(imagesrc)
    cryimage = cv2.imread("cri.png", -1)
    b, g, r, a = cv2.split(cryimage)
    cryimage = cv2.merge((r, g, b, a))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("[INFO] Found {0} face(s)".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        image = placeimage(image, cryimage, x, y, w, h)
    imagesrc = Image.fromarray(image.astype('uint8'))
    return imagesrc

def JPEG(imgsrc, quality):
    imagecv = np.array(imgsrc)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', imagecv, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    imagesrc = Image.fromarray(decimg.astype('uint8'))
    return imagesrc

def hunEmoji(imgsrc):

    imagecv = np.array(imgsrc)
    randsize = randrange(3, 10)
    cvW = imagecv.shape[1]
    cvH = imagecv.shape[0]
    rW = randrange(int(cvW/randsize), cvW - int(cvW/randsize))
    rH = randrange(int(cvH/randsize), cvH - int(cvH/randsize))

    hunimage = cv2.imread("100.png", -1)
    b, g, r, a = cv2.split(hunimage)
    hunimage = cv2.merge((r, g, b, a))

    image = placeimage(imagecv, hunimage, rW, rH, int(cvW/randsize), int(cvH/randsize))

    imagesrc = Image.fromarray(image.astype('uint8'))
    return imagesrc

def saturate(imagesrc):
    imagecv = np.array(imagesrc)
    (h, s, v) = cv2.split(imagecv)
    s = s * 2
    s = np.clip(s, 0, 255)
    image = cv2.merge([h, s, v])
    imagesrc = Image.fromarray(image.astype('uint8'))
    return imagesrc

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.init_window()

    def init_window(self):
        self.master.title("Fryer")

        self.pack(fill=BOTH, expand=1)
        FryButton  = Button(self, text="Fry one step", command=self.frystep, width=10, height=1)
        QuitButton = Button(self, text="Quit", command=self.quit, width=10, height=1)
        fileButton = Button(self, text="Open File", command=self.openfile, width=10, height=1)
        refryButton = Button(self, text="Refry", command=self.refry, width=10, height=1)
        self.HSV = IntVar()
        HSVButton = Checkbutton(self, text="HSV", variable=self.HSV, )
        self.scalemean = Scale(self, from_=0, to=10, orient=HORIZONTAL, tickinterval=0.25)
        self.scalevar = Scale(self, from_=0, to=1000, orient=HORIZONTAL)


        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=310, mode="determinate")
        self.imagesrc = Image.open("Fry.jpg")
        imageresized = self.imagesrc.copy()  # #copy is needed as image.thumbnail does not return
        imageresized.thumbnail((530, 530), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(imageresized)
        self.label = Label(image=self.image, width=530, height=530)
        self.label.image = self.image # keep a reference!
        self.label.place(x=10, y=10)

        T1 = Label(self)
        T2 = Label(self)
        T3 = Label(self)
        T4 = Label(self)
        T1.config(text = "mean")
        T2.config(text="variance")
        T3.config(text="ripple x")
        T4.config(text="ripple y")
        T1.place(x=10, y=620)
        T2.place(x=10, y=660)
        T3.place(x=250, y=620)
        T4.place(x=250, y=660)

        ##sin params
        self.xA = StringVar()
        self.xw = StringVar()
        self.yA = StringVar()
        self.yw = StringVar()
        self.ExA = Entry(self, width=5, textvariable=self.xA)
        self.Exw = Entry(self, width=5, textvariable=self.xw)
        self.EyA = Entry(self, width=5, textvariable=self.yA)
        self.Eyw = Entry(self, width=5, textvariable=self.yw)
        self.ExA.place(x=300, y=620)
        self.Exw.place(x=350, y=620)
        self.EyA.place(x=300, y=660)
        self.Eyw.place(x=350, y=660)
        self.xA.set(100)
        self.xw.set(3)
        self.yA.set(100)
        self.yw.set(3)

        FryButton.place(x=340, y=560)
        QuitButton.place(x=440, y=560)
        HSVButton.place(x=40, y=580)
        fileButton.place(x=340, y=580)
        refryButton.place(x=440, y=580)
        self.scalemean.place(x=80, y=610)
        self.scalevar.place(x=80, y=650)
        self.scalemean.set(int(1))
        self.scalevar.set(int(10))
        self.progress.place(x=10, y=562)

        self.progress["value"] = 0
        self.progress["maximum"] = 100

    def frystep(self):
        print("[INFO] Frying")
        if self.progress["value"] < 5:
            self.imagesrc = Bemoji(self.imagesrc)
        elif self.progress["value"] < 10:
            self.imagesrc = facereg(self.imagesrc)
        elif self.progress["value"] < 15:
            self.imagesrc = hunEmoji(self.imagesrc)
        elif self.progress["value"] < 20:
            self.imagesrc = noise(self.imagesrc, self.HSV.get(), self.scalemean.get(), self.scalevar.get())
        elif self.progress["value"] < 25:
            self.imagesrc = ripple(self.imagesrc, int(self.xA.get()), int(self.xw.get()), int(self.yA.get()), int(self.yw.get()))
        elif self.progress["value"] < 30:
            self.imagesrc = JPEG(self.imagesrc, 10)
        elif self.progress["value"] < 35:
            self.imagesrc = saturate(self.imagesrc)
        elif self.progress["value"] < 40:
            self.imagesrc = JPEG(self.imagesrc, 9)
        elif self.progress["value"] < 45:
            self.imagesrc = Bulge(self.imagesrc, randrange(100,400), randrange(100,400), 100)
        imageresized = self.imagesrc.copy()
        imageresized.thumbnail((530, 530), Image.ANTIALIAS)
        new_image = ImageTk.PhotoImage(imageresized)
        self.label.configure(image=new_image)
        self.label.image = new_image
        #print(type(new_image))
        self.progress["value"] += 5

    def openfile(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                     filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if filename:
            self.imagesrc = Image.open(filename)
            imageresized = self.imagesrc.copy()
            imageresized.thumbnail((530, 530), Image.ANTIALIAS)
            new_image = ImageTk.PhotoImage(imageresized)
            self.label.configure(image=new_image)
            self.label.image = new_image

    def refry(self):
        self.progress["value"] = 0

    def quit(self):
        exit()


if __name__ == '__main__':
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("550x700")
    app = Window(root)
    root.mainloop()
