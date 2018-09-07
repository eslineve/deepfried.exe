#/usr/bin/python3

from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image

import numpy as np
import cv2

def noise(image, hsv):
      imagecv = np.array(image)
      if(hsv):
          imagecv = cv2.cvtColor(imagecv, cv2.COLOR_RGB2HSV)

      row,col,ch = imagecv.shape
      mean = 1
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = imagecv + gauss
      if (hsv):
          noisy = cv2.cvtColor(noisy.astype('uint8'), cv2.COLOR_HSV2RGB)

      #im = Image.fromarray(noisy)
      image = Image.fromarray(noisy.astype('uint8'))
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

        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=310, mode="determinate")
        self.imagesrc = Image.open("Fry.jpg")
        self.image = ImageTk.PhotoImage(self.imagesrc)
        self.label = Label(image=self.image, width=530, height=530)
        self.label.image = self.image # keep a reference!
        self.label.place(x=10, y=10)

        FryButton.place(x=340, y=560)
        QuitButton.place(x=440, y=560)
        HSVButton.place(x=10, y=580)
        self.progress.place(x=10, y=562)

        self.progress["value"] = 0
        self.progress["maximum"] = 100

    def frystep(self):
        print("Fry that picture")
        if (self.progress["value"] < 5):
            self.imagesrc = noise(self.imagesrc, self.HSV.get())
        elif (self.progress["value"] < 10):
            print("ripple")
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
    root.geometry("550x650")
    app = Window(root)
    root.mainloop()
