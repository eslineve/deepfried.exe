#/usr/bin/python3

from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image

import numpy as np
import cv2

def noise(image, hsv, mean, var):
      imagecv = np.array(image)
      if(hsv):
          imagecv = cv2.cvtColor(imagecv, cv2.COLOR_RGB2HSV)

      row,col,ch = imagecv.shape
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = imagecv + gauss
      if (hsv):
          noisy = cv2.cvtColor(noisy.astype('uint8'), cv2.COLOR_HSV2RGB)

      #im = Image.fromarray(noisy)
      image = Image.fromarray(noisy.astype('uint8'))
      return image
def ripple(image, xA, xw, yA, yw):
    imagecv = np.array(image)
    if(xA == 0):
        Ax = 0
    else:
        Ax = imagecv.shape[1] / xA
    wx = xw / imagecv.shape[0]
    if(yA == 0):
        Ay = 0
    else:
        Ay = imagecv.shape[0] / yA
    wy = yw / imagecv.shape[1]

    shiftx = lambda x: Ax * np.cos(2.0 * np.pi * x * wx)
    shifty = lambda x: Ay * np.cos(2.0 * np.pi * x * wy)

    for i in range(imagecv.shape[0]):
        imagecv[i, :] = np.roll(imagecv[i, :], int(shiftx(i)), 0)
        imagecv[:, i] = np.roll(imagecv[:, i], int(shifty(i)), 0)
    image = Image.fromarray(imagecv.astype('uint8'))
    return image

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
        self.HSV = IntVar()
        HSVButton = Checkbutton(self, text="HSV", variable=self.HSV, )
        self.scalemean = Scale(self, from_=0, to=10, orient=HORIZONTAL, tickinterval=0.25)
        self.scalevar = Scale(self, from_=0, to=1000, orient=HORIZONTAL)


        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=310, mode="determinate")
        self.imagesrc = Image.open("Fry.jpg")
        self.image = ImageTk.PhotoImage(self.imagesrc)
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
        self.xA.set(50)
        self.xw.set(3)
        self.yA.set(0)
        self.yw.set(3)

        FryButton.place(x=340, y=560)
        QuitButton.place(x=440, y=560)
        HSVButton.place(x=40, y=580)
        self.scalemean.place(x=80, y=610)
        self.scalevar.place(x=80, y=650)
        self.scalemean.set(int(1))
        self.scalevar.set(int(10))
        self.progress.place(x=10, y=562)

        self.progress["value"] = 0
        self.progress["maximum"] = 100

    def frystep(self):
        print("Fry that picture")
        if (self.progress["value"] < 5):
            self.imagesrc = noise(self.imagesrc, self.HSV.get(), self.scalemean.get(), self.scalevar.get())
        elif (self.progress["value"] < 10):
            print("ripple")
            self.imagesrc = ripple(self.imagesrc, int(self.xA.get()), int(self.xw.get()), int(self.yA.get()), int(self.yw.get()))
        new_image = ImageTk.PhotoImage(self.imagesrc)
        self.label.configure(image=new_image)
        self.label.image = new_image
        #print(type(new_image))
        self.progress["value"] += 5

    def quit(self):
        exit()


if __name__ == '__main__':
    print("Hello World my old friend")

    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("550x700")
    app = Window(root)
    root.mainloop()
