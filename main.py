#/usr/bin/python3

from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image

import numpy as np
import cv2

def noise(image):
      imagecv = cv2.cvtColor(cv2.imread("Fry.jpg"), cv2.COLOR_BGR2RGB)

      row,col,ch = imagecv.shape
      mean = 1
      var = 1000
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = imagecv + gauss
      print(type(noisy))
      #im = Image.fromarray(noisy)
      imgtk = ImageTk.PhotoImage(image = Image.fromarray(noisy.astype('uint8')))
      return imgtk

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

        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=310, mode="determinate")

        self.image = ImageTk.PhotoImage(Image.open("Fry.jpg"))
        self.label = Label(image=self.image, width=530, height=530)
        self.label.image = self.image # keep a reference!
        self.label.place(x=10, y=10)

        FryButton.place(x=340, y=560)
        QuitButton.place(x=440, y=560)
        self.progress.place(x=10, y=562)

        self.progress["value"] = 0
        self.progress["maximum"] = 100

    def frystep(self):
        print("Fry that picture")
        new_image = noise(self.image)
        self.label.configure(image=new_image)
        self.label.image = new_image
        self.progress["value"] += 5

    def quit(self):
        exit()


if __name__ == '__main__':
    print("Hello World my old friend")

    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("550x600")
    app = Window(root)

    root.mainloop()
