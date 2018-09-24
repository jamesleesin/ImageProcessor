#!/usr/bin/kivy
from kivy.config import Config
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.dropdown import DropDown
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line, InstructionGroup

import numpy as np
import threading
import sys

# Classes
sys.path.insert(0, 'src')
from EditingPage import EditingScreen

# Set this variable here to turn debug mode on and off
debugMode = True

# Screens
class ScreenManager(ScreenManager):
	pass

class ImageProcessorApp(App):
	def build(self):
		# Get main monitor size
		Config.set('graphics', 'fullscreen', 0)
		Config.set('graphics', 'resizable', 1)
		#Config.set('graphics', 'window_state', 'maximized')
		Config.write()

		root = Builder.load_file("kv/GUIManager.kv")
		return root


if __name__ == "__main__":
    ImageProcessorApp().run()
