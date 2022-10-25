import datetime
import os
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from turtle import width
import webbrowser
import route_finding as router
import datetime
from TrafficData.TrafficFlowPredictor import *

root = Tk()


class Window:
	master = None
	routesText = None

	src = StringVar()
	dest = StringVar()
	pred = StringVar()
	model = StringVar()
	date_string = StringVar()

	def viewRoutes(self):
		webbrowser.open_new_tab('file://' + os.path.realpath('index.html'))

	def setTextBox(self, text):
		self.routesText.configure(state=NORMAL)
		self.routesText.delete(1.0,END)
		self.routesText.insert(INSERT, text)
		self.routesText.configure(state=DISABLED)
		return

	def get_date(self):
		date_string = self.date_string.get()

		date = None
		try:
			date = parse_date(date_string)
		except:
			date = datetime.datetime.now()
		return date

	def run(self):
		self.setTextBox("Generating Routes...")

		src = self.src.get()
		dest = self.dest.get()

		if src == '' or dest == '':
			self.setTextBox("Please enter SCATS")
			return
		
		routes = router.runRouter(src, dest, self.get_date(), self.model.get())
		self.setTextBox(routes)
		return
	
	def predictFlow(self):
		self.setTextBox("Predicting...")
		point = str(self.pred.get())

		if point == '':
			self.setTextBox("Please enter SCATS")
			return

		predictor = TrafficFlowPredictor()
		date = self.get_date()
		try:
			flow = predictor.predict_traffic_flow(point, date, 4, self.model.get())
		except:
			self.setTextBox("Invalid SCATS")
			return
		self.setTextBox(f"--Predicted Traffic Flow--\nSCATS:\t\t{point}\nTime:\t\t{date.strftime('%Y/%m/%d %I:%M:%S')}\nPrediction:\t\t{str(flow)}veh/hr")
		return

	def createWindow(self):
		self.master.title("Route Navigation")
		self.master.geometry("400x750")
		self.master.resizable(False, True)
		Label(self.master, text="Route Navigation", font='Helvetica 18 bold').pack()
	
	def renderElements(self):
		modelLbl = Label(self.master, width=20, text="Model:")
		self.model.set(TrafficFlowModelsEnum.LSTM.value)

		model_selection = OptionMenu(self.master, self.model, *[option.value for option in TrafficFlowModelsEnum])

		modelLbl.pack()
		model_selection.pack()

		srcLbl = Label(self.master, width=20, text="Source:*")
		srcInput = Entry(self.master, width=10, text=self.src)
		destLbl = Label(self.master, width=20, text="Destination:*")
		destInput = Entry(self.master, width=10, text=self.dest)
		dateLbl = Label(self.master, width=20, text="Date/Time:")
		dateInput = Entry(self.master, width=20, text=self.date_string)

		# add locations to window
		srcLbl.pack()
		srcInput.pack()
		destLbl.pack()
		destInput.pack()
		dateLbl.pack()
		dateInput.pack()

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
		

# Date Format: 2006/10/02 2:45
def parse_date(date_string):
    date,time = date_string.split()
    year,month,day = date.split('/')
    hour,minute = time.split(':')
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))

# keep the window open on the mainloop
gui = Window(root)
root.mainloop()
