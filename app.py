import os
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from turtle import width
import webbrowser
import route_finding as router

root = Tk()


class Window:
	master = None
	routesText = None

	src = StringVar()
	dest = StringVar()

	def viewRoutes(self):
		webbrowser.open_new_tab('file://' + os.path.realpath('index.html'))

	def run(self):
		self.routesText.configure(state=NORMAL)
		self.routesText.delete(1.0,END)
		self.routesText.insert(INSERT, "Generating Routes...")
		self.routesText.configure(state=DISABLED)
		
		src = int(self.src.get())
		dest = int(self.dest.get())

		self.routesText.configure(state=NORMAL)
		self.routesText.delete(1.0,END)
		self.routesText.insert(INSERT, router.runRouter(src, dest))
		self.routesText.configure(state=DISABLED)
		return
		

	def createWindow(self):
		self.master.title("Route Navigation")
		self.master.geometry("400x600")
		self.master.resizable(False, False)
		Label(self.master, text="Route Navigation", font='Helvetica 18 bold').pack()
	
	def renderElements(self):
		srcLbl = Label(self.master, width=20, text="Source:")
		srcInput = Entry(self.master, width=10, text=self.src)
		destLbl = Label(self.master, width=20, text="Destination:")
		destInput = Entry(self.master, width=10, text=self.dest)

		# add locations to window
		srcLbl.pack()
		srcInput.pack()
		destLbl.pack()
		destInput.pack()


		# Create the rest of the UI elements
		generateBtn = Button(self.master, text="Generate", command=self.run)


		self.routesText = ScrolledText(self.master, width=50, padx=0)

		displayBtn = Button(self.master, text="View", command=self.viewRoutes)

		# Add the elements to the window
		generateBtn.pack()
		self.routesText.pack()
		self.routesText.configure(state=DISABLED)

		displayBtn.pack()

	def __init__(self, master):
		self.master = master
		self.createWindow()
		self.renderElements()
		


# keep the window open on the mainloop
gui = Window(root)
root.mainloop()
