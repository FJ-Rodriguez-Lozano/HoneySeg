#------------------------------------------------------------
# Libraries importation
#------------------------------------------------------------
import tkinter as tk # GUI library imported
import tkinter.filedialog # to create open files windows and folders
import tkinter.ttk # for the processing progress bar
import os # to manage actions of the operating system
import keras # to use keras api
import tensorflow as tf # to use tensorflow image functions
from keras.models import Model, load_model # to load pre-trained models
import segmentation_models as sm # Segmentation Models: using `keras` framework.
from keras import backend as K
import cv2 # opencv library to process images
import PIL.Image, PIL.ImageTk # to manage images inside the interface
import numpy as np # to make calculations
import math # to make calculations
from pathlib import Path # to manage system paths (windows, linux, etc)
import shutil # to delete no empty folders

#------------------------------------------------------------
# App information
#------------------------------------------------------------

# app developed version
app_version = 1.0  

# last app developed version date 
last_app_released_date = "2025-07-20" 

# developers authors 
authors_text = "Francisco J. Rodriguez-Lozano - Dpto. Ingeniería Electrónica y de Computadores, Universidad de Córdoba (Spain)\n \
                José Luis Ávila Jiménez - Dpto. Ingeniería Electrónica y de Computadores, Universidad de Córdoba (Spain)\n \
                Sergio R. Geninatti - Dpto. de Ingeniería Electrónica, Universidad Nacional de Rosario (Argentina)\n \
                José M. Flores - Dpto. de Zoología, Universidad de Córdoba (Spain)\n \
                Manuel Ortiz-Lopez - Dpto. Ingeniería Electrónica y de Computadores, Universidad de Córdoba (Spain)\n"

# app license 
license_text = "MIT License\n\n \
                Copyright (c) 2025 PhD. Francisco J. Rodriguez Lozano\n\n \
                Permission is hereby granted, free of charge, to any person obtaining a copy\n \
                of this software and associated documentation files (the \"Software\"), to deal\n \
                in the Software without restriction, including without limitation the rights\n \
                to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n \
                copies of the Software, and to permit persons to whom the Software is\n \
                furnished to do so, subject to the following conditions:\n\n \
                The above copyright notice and this permission notice shall be included in all\n \
                copies or substantial portions of the Software.\n\n \
                THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n \
                IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n \
                FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n \
                AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n \
                LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n \
                OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n \
                SOFTWARE."


#------------------------------------------------------------
# Main class of the graphical interface. Contains the interface 
# elements as well as the functionality.
#------------------------------------------------------------
class HoneySegmentationToolGUI(tk.Tk):
  def __init__(self, image_path=None):
    super().__init__()

    self.honeySegmentationModelPath = Path("defaults/honeyModels/efficientnetb2-FPN.keras") # path to keras segmentation model

    self.imgPath = None # path to load image to process
    self.splittedPath = None # temporal path to split image into tiles and process

    self.opencvImg = None # opencv image to calculate reference in image
    self.opencvMask = None # opencv image to store the processed mask
    self.opencvMaskApply = None # opencv image to blend the image and the mask
    
    self.tkimage = None # Image used to display in tkinter label
    self.tkimagePreview = None # Image used to display in tkinter label
    self.tkimageSegmented = None # Image used to display in tkinter label
    
    self.appIcon = Path("defaults/icons/icon.png") # path of app icon 
    self.imgDefaultLoad = Path("defaults/previewImages/defaultImageLoad.png") # path to default preview image
    self.imgDefaultProcess = Path("defaults/previewImages/defaultImageProcess.png") # path to default processed image 

    self.cmEntry = 0 # Used to find relationship between centimeters and pixels 
    self.referenceValue = 0 # Given relationship between centimeters and pixels by the user
    self.cmToPixelRelation = 0 # Calculated relationship between centimeters and pixels by the user    
    
    self.areaOfHoney = 0 # Calculated area of honey in real world values    

    self.labelImgagePreview = None # preview default/loaded image
    self.lavelSegmentedImage = None # preview calculated segmentation image   
    self.varLabelInformationText = tk.StringVar() # to show information of the image
 
    self.processButtom = None # Buttom to start the segmentation process
    self.referenceButtom = None # Buttom to find the relationship between centimeters and pixels 
    self.saveButtom = None # Buttom to save the segmentation result

    self.progressBar = None # Progress var to see the percentage of completition of the process

    self.title('HoneySeg - A Honey Bee Segmetation Tool GUI') # main window title

    self.geometry("1366x768") # init window size
    self.minsize(1366, 768) # minimum window size
    self.maxsize(1366, 768) # maximum window size

    self.call('wm', 'iconphoto', self._w, PIL.ImageTk.PhotoImage(file=str(self.appIcon))) # default app icon
     
     
    self.display = tk.Label(self)
    self.display.pack(expand=True, fill=tk.BOTH)
 
    # main elements of the top menu of the app
    menuTabs = tk.Menu(self)
    self.config(menu=menuTabs)

    fileMenu = tk.Menu(menuTabs)
    menuTabs.add_cascade(label='File', menu=fileMenu)
    
    fileMenu.add_separator()
    fileMenu.add_command(label='Exit', command=self.quit)

    helpMenu = tk.Menu(menuTabs)
    menuTabs.add_cascade(label='Help', menu=helpMenu)
    helpMenu.add_command(label='About', command= self.aboutWindow) 
 


  #------------------------------------------------------------
  # Function to show app information in a new child window
  #
  # Params:
  #        None
  # Return:
  #        None
  #------------------------------------------------------------
  def aboutWindow(self): 
    # generate a new child window
    aboutMessageWindow = tk.Toplevel() 
    
    # keep focus in this new window and prevent to interact with the main window
    aboutMessageWindow.grab_set() 

    # fill the message to show in the new window
    message = "App version: " + str(app_version) + "\n" + \
              "App date: " + last_app_released_date + "\n" + \
              "This application was developed by:"  + "\n" + authors_text  + "\n" + license_text

    # add message information as text label to about window
    tk.Label(aboutMessageWindow, text=message).pack() 

    # add a button to go back to main window and close about window 
    tk.Button(aboutMessageWindow, text='Exit', command=aboutMessageWindow.destroy).pack() 
    

#main call to the class of the graphical interface
HoneySegmentationToolGUI().mainloop()
