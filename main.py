#/usr/bin/python3

from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image

import numpy as np

def noise(image):
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

class Window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.master = master

		self.init_window()

	def init_window(self):
		self.master.title("Fryer")

		self.pack(fill=BOTH, expand=1)
		FryButton  = Button(self, text="Fry one step", command=self.FryStep, width=10, height=1)
		QuitButton = Button(self, text="Quit", command=self.Quit, width=10, height=1)

		self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=310, mode="determinate")

		image = ImageTk.PhotoImage(Image.open("Fry.jpg"))
		label = Label(image=image, width=530, height=530)
		label.image = image # keep a reference!
		label.place(x=10, y=10)

		FryButton.place(x=340, y=560)
		QuitButton.place(x=440, y=560)
		self.progress.place(x=10, y=562)

		self.progress["value"] = 0
		self.progress["maximum"] = 100

	def FryStep(self):
		print("Fry that picture")

		self.progress["value"] += 5

	def Quit(self):
		exit()


if __name__ == '__main__':
	print("Hello World my old friend")

	root = Tk()
	root.resizable(width=False, height=False)
	root.geometry("550x600")
	app = Window(root)

	root.mainloop()
