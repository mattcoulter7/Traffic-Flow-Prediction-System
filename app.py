import datetime
import os
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from turtle import width
import webbrowser
import route_finding as router
from TrafficData.TrafficFlowPredictor import *

root = Tk()


class Window:
	master = None
	routesText = None

	src = StringVar()
	dest = StringVar()
	pred = StringVar()
	model = StringVar()

	def viewRoutes(self):
		webbrowser.open_new_tab('file://' + os.path.realpath('index.html'))

	def setTextBox(self, text):
		self.routesText.configure(state=NORMAL)
		self.routesText.delete(1.0,END)
		self.routesText.insert(INSERT, text)
		self.routesText.configure(state=DISABLED)
		return

	def run(self):
		self.setTextBox("Generating Routes...")

		src = int(self.src.get())
		dest = int(self.dest.get())
		
		routes = router.runRouter(src, dest, self.model.get())
		self.setTextBox(routes)
		return
	
	def predictFlow(self):
		self.setTextBox("Predicting...")
		point = str(self.pred.get())
		predictor = TrafficFlowPredictor()
		flow = predictor.predict_traffic_flow(point, datetime.datetime.now(), 4, self.model.get())
		self.setTextBox(f"--Predicted Traffic Flow--\nSCATS:\t\t{point}\nTime:\t\t{datetime.datetime.now().strftime('%Y/%m/%d %I:%M:%S')}\nPrediction:\t\t{str(flow)}veh/hr")
		return

	def createWindow(self):
		self.master.title("Route Navigation")
		self.master.geometry("400x630")
		self.master.resizable(False, True)
		Label(self.master, text="Route Navigation", font='Helvetica 18 bold').pack()
	
	def renderElements(self):
		modelLbl = Label(self.master, width=20, text="Model:")
		self.model.set(TrafficFlowModelsEnum.LSTM.value)

		model_selection = OptionMenu(self.master, self.model, *[option.value for option in TrafficFlowModelsEnum])

		modelLbl.pack()
		model_selection.pack()

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

		# SCAT prediction ui
		predLbl = Label(self.master, width=20, text="Predict SCAT Traffic Flow:")
		predInput = Entry(self.master, width=10, text=self.pred)
		predBtn = Button(self.master, text="Predict", command=self.predictFlow)

		predLbl.pack()
		predInput.pack()
		predBtn.pack()


	def __init__(self, master):
		self.master = master
		self.createWindow()
		self.renderElements()
		


# keep the window open on the mainloop
gui = Window(root)
root.mainloop()
