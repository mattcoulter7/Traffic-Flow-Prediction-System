# base Class of your App inherits from the App class.
from kivy.app import App
# GridLayout arranges children in a matrix.
from kivy.uix.gridlayout import GridLayout
# Label is used to label something
from kivy.uix.label import Label
# used to take input from users
from kivy.uix.textinput import TextInput



class App(GridLayout):
	def __init__(self, **var_args):
		
		super(LoginScreen, self).__init__(**var_args)
		# super function can be used to gain access
		# to inherited methods from a parent or sibling class
		# that has been overwritten in a class object.
		self.cols = 1	 # You can change it accordingly
		self.add_widget(Label(text ='User Name'))
		self.username = TextInput(multiline = True)
		
		# multiline is used to take
		# multiline input if it is true
		self.add_widget(self.username)
		self.add_widget(Label(text ='password'))
		self.password = TextInput(password = True, multiline = False)
		
		# password true is used to hide it
		# by * self.add_widget(self.password)
		self.add_widget(Label(text ='Comfirm password'))
		self.password = TextInput(password = True, multiline = False)
		self.add_widget(self.password)


# the Base Class of our Kivy App
class MyApp(App):
	def build(self):
		# return a LoginScreen() as a root widget
		return App()


if __name__ == '__main__':
	MyApp().run()
