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
# Auxiliary function to split/merge the images. This function 
# find the pints where is necesary to split or merge the img
#
# Params:
#        size: original size
#        split_size: desired size
#        overlap: pertentage of overlaping between tiles points. 
#                 Specified in range 0-1.
# Return:
#        List of points where each tile begins
#------------------------------------------------------------
def start_points(size, split_size, overlap=0):
  points = [0]
  stride = int(split_size * (1-overlap))
  counter = 1

  while True:
    pt = stride * counter

    if pt + split_size >= size:
      if split_size == size:
        break
      points.append(size - split_size)
      break

    else:
      points.append(pt)
    counter += 1
  return points

#------------------------------------------------------------
# Auxiliary function to obtain the names of the files of a directory
# without extension that shares a name.
#
# Params:
#        imagesPath: path to the files in the directory
#        str_value: shared name between files
# Return:
#        list of image names without extension
#------------------------------------------------------------
def get_file_names_with_strings(imagesPath, str_value):
  # let's get the entire list of files
  full_list = sorted((f for f in os.listdir(imagesPath) if not f.startswith(".")), key=str.lower)

  # filter the files and store only those with the value of str_value in their names
  final_list = [nm for nm in full_list if str_value in nm]

  return final_list

#------------------------------------------------------------
# Auxiliary function to read images in tensorflow format
#
# Params:
#        img_path: path of the image
#        IMG_HEIGHT: height size of the image
#        IMG_WIDTH: width size of the image
#        IMG_CHANNELS: channels of the image 
# Return:
#        image in tensorflow format
#------------------------------------------------------------
def read_image(img_path, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
  image = tf.io.read_file(img_path)
  image = tf.image.decode_image(image, channels=IMG_CHANNELS, expand_animations = False)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  return image

#------------------------------------------------------------
# Auxiliary class to manage opencv mouse events functions
#------------------------------------------------------------
class openCVmouseEventStore:
  def __init__(self):
    self.drawing = False # true if mouse is pressed
    self.s_x = None # start x coordinate
    self.s_y = None # start y coordinate
    self.e_x = None # end x coordinate
    self.e_y = None # end y coordinate

  #------------------------------------------------------------
  # Internal class function to draw lines on an image according to
  # mouse events. Based on opencv callback mouse function.
  #
  # Params:
  #        event: original size
  #        x: desired size
  #        y: pertentage of overlaping between tiles points. 
  #        flags: not used
  #        param: not used
  # Return:
  #        void
  #------------------------------------------------------------
  def line_drawing(self,event,x,y,flags,param):
    # if the left mouse button has been pressed and is held down
    if event==cv2.EVENT_LBUTTONDOWN:
      self.drawing=True
      self.s_x, self.s_y = x, y
      self.e_x, self.e_y = x, y
      
    #if the mouse is moving and drawing is true
    elif event==cv2.EVENT_MOUSEMOVE:
      if self.drawing==True:
        self.e_x, self.e_y = x, y
        
    #if the finger has been lifted from the left mouse button
    elif event==cv2.EVENT_LBUTTONUP:
      self.drawing=False
      self.e_x, self.e_y = x, y


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
    fileMenu.add_command(label='Select file', command=self.load_image)
    fileMenu.add_command(label='Load custom honey segmentation model', command=lambda: self.load_file_path(1))
    
    fileMenu.add_separator()
    fileMenu.add_command(label='Exit', command=self.quit)

    helpMenu = tk.Menu(menuTabs)
    menuTabs.add_cascade(label='Help', menu=helpMenu)
    helpMenu.add_command(label='About', command= self.aboutWindow) 
 

    # split app interface in two  areas. Left for load image, buttoms, etc. Right to show calulated segmentation 
    left_frame = tk.Frame(self, width=450, height=700, bg='grey')
    left_frame.pack(side='left',  fill='both',  padx=10,  pady=20,  expand=True)

    right_frame = tk.Frame(self, width=850, height=700, bg='grey')
    right_frame.pack(side='right',  fill='both',  padx=10,  pady=20,  expand=True)

    # Make a frame to pack the buttons horizontally
    openExitButtons = tk.Frame(left_frame)
    openExitButtons.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
 
    # Make buttons to load image, and app quit
    button = tk.Button(openExitButtons, text="LOAD IMAGE", command=self.load_image)
    button.pack(side=tk.LEFT, expand=True, fill=tk.X)

    button = tk.Button(openExitButtons, text="EXIT", command=self.destroy)
    button.pack(side=tk.LEFT, expand=True, fill=tk.X) 
 
    # preview of loaded image. By default a custom text informative image
    im = PIL.Image.open(str(self.imgDefaultLoad))
    resized_img = im.resize((256, 256))
    self.tkimagePreview = PIL.ImageTk.PhotoImage(resized_img)

    self.labelImgagePreview = tk.Label(left_frame, image=self.tkimagePreview)
    self.labelImgagePreview.pack(fill=tk.X, padx=5, pady=5) 

    # preview of calculated segmentation image. By default a custom text informative image
    im = PIL.Image.open(str(self.imgDefaultProcess))
    resized_img = im.resize((830, 680))
    self.tkimageSegmented = PIL.ImageTk.PhotoImage(resized_img)    
    self.lavelSegmentedImage = tk.Label(right_frame, image=self.tkimageSegmented)
    self.lavelSegmentedImage.pack(fill=tk.X, padx=5, pady=5) 
  
    # frame region of the buttom to calculate or specify the relationship between centimeters and pixels 
    referenceFrame = tk.Frame(left_frame)
    referenceFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
  
    self.referenceButtom = tk.Button(referenceFrame, text='Find reference', command=self.find_reference)
    self.referenceButtom.pack(side=tk.TOP, expand=True, fill=tk.X, pady=5) # add a button to go back to main window and       
    self.referenceButtom["state"] = "disabled" # disabled until an image is loaded

    tk.Label(referenceFrame, text="Specify cm² per pixel").pack(side=tk.LEFT, expand=True, fill=tk.X)
    self.cmToPixelRelation = tk.Entry(referenceFrame)
    self.cmToPixelRelation.delete(0,tk.END)
    self.cmToPixelRelation.insert(0,"0")
    self.cmToPixelRelation.pack(expand=True, fill=tk.X)
    referenceButtom = tk.Button(referenceFrame, text='Click to apply', command=lambda: self.get_entry(self.cmToPixelRelation, self.referenceValue))
    referenceButtom.pack(side=tk.LEFT, expand=True, fill=tk.X) # add a button to go back to main window and close about window 

    # frame region of the process buttom
    processFrame = tk.Frame(left_frame)
    processFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    self.processButtom = tk.Button(processFrame, text='Process Image     >>>', command=lambda: self.segmentationProcess(640, 640, 3))
    self.processButtom.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5,  pady=5) # add a button to go back to main window and close about window     
    self.processButtom["state"] = "disabled" # disabled  until the relationship between centimeters and pixels is specified.

    # frame region to show the information retrieved by the loaded image
    informationImageFrame = tk.Frame(left_frame)
    informationImageFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label=tk.Label(informationImageFrame, text="IMAGE INFORMATION")
    label.config(font=("bold"))
    label.pack(side=tk.TOP, expand=True, fill=tk.X)

    self.varLabelInformationText.set("Image Name: -\nImage Size: -x-\n Area of honey: - cm²")
    tk.Label(informationImageFrame, textvariable=self.varLabelInformationText,wraplength=450 - 20).pack(side=tk.BOTTOM, expand=True, fill=tk.X)

    # frame region of the save buttom
    saveFrame = tk.Frame(left_frame)
    saveFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    self.saveButtom = tk.Button(saveFrame, text='Save result', command=self.saveResults)
    self.saveButtom.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5,  pady=5) # add a button to go back to main window and close about window   
    self.saveButtom["state"] = "disabled" # dusabled until segmentation proccess is finished
    
    # frame region of the progressbar widget
    progressBarFrame = tk.Frame(left_frame)
    progressBarFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # Create a progressbar widget
    self.progressBar = tk.ttk.Progressbar(progressBarFrame, orient="horizontal", length=300, mode="determinate")
    self.progressBar.pack(fill=tk.X, padx=5,  pady=5)
    

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
    

  #------------------------------------------------------------
  # function to obtain a value entered in the interface. This 
  # function will be used to use the data entered by the user 
  # as the size in cm2 of a pixel.
  #
  # Params:
  #        entry: entry field of the interface
  #        saveEntryVar: variable to return the data
  # Return:
  #        None
  #------------------------------------------------------------
  def get_entry(self, entry, saveEntryVar): 
    saveEntryVar = entry.get() 
    

  #------------------------------------------------------------
  # Fuction to show the type of files to load. Keras models or
  # supported image formats files. This function will cover the
  # option to show all file formats for ask dialogs
  #
  # Params:
  #        typeOfFile: 1: keras model files. 2: image formats
  # Return:
  #        None
  #------------------------------------------------------------
  def load_file_path(self, typeOfFile):
    # supprted keras models formtas
    modelTypes = (
                 ('model files', '*.keras'),
                 ('All files', '*.*')
                 )
    # supported images formats             
    imageTypes = (
                 ('JPG', '*.jpg .JPG'),      
                 ('JPEG', '*.jpeg *.JPEG'),
                 ('PNG', '*.png *.PNG'),                              
                 ('All files', '*.*')
                 )    
    # type 1 to open a ask dialog for load a keras model    
    if typeOfFile == 1:
      self.honeySegmentationModelPath = tk.filedialog.askopenfilename(title='Select a model', filetypes=modelTypes)    

    # type 2 to open a ask dialog for load images    
    elif typeOfFile == 2:
      self.imgPath = tk.filedialog.askopenfilename(title='Select an image', filetypes=imageTypes)
    

  #------------------------------------------------------------
  # Function to load an image to process and previsualizate
  #
  # Params:
  #        None
  # Return:
  #        None
  #------------------------------------------------------------
  def load_image(self):
    # choose image type with the type specified in imageTypes of function "load_file_path"
    self.load_file_path(2)
    
    if self.imgPath:
      # Read image and convert to RGB
      self.opencvImg = cv2.imread(self.imgPath)
      self.opencvImg = cv2.cvtColor(self.opencvImg, cv2.COLOR_BGR2RGB)
 
      # Convert image to tkinter image format and display
      self.tkimage = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv2.resize(self.opencvImg, (640, 480))))
      
      # resieze the previsualization of the image
      resized_imgPreview = cv2.resize(self.opencvImg, (256, 256))
      self.tkimagePreview = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(resized_imgPreview))
      
      self.labelImgagePreview.config(image=self.tkimagePreview)
      
      # get information of the loaded image and update this information in the interface
      rows, cols, _ = self.opencvImg.shape
      self.varLabelInformationText.set("Image Name:" + self.imgPath + "\nImage Size: " + str(cols) + "x" + str(rows) + "\nArea of honey: - cm²")

      # enable the process and find reference buttons
      self.referenceButtom["state"] = "normal"
      self.processButtom["state"] = "normal"
      self.update()
    

  #------------------------------------------------------------
  # Auxiliary function to split the images
  #
  # Params:
  #        sX: start x coordinate of the first point of the line
  #        sY: start x coordinate of the first point of the line
  #        eX: end x coordinate of the first point of the line
  #        eY: end y coordinate of the first point of the line
  #        originalWidth: Width of the loaded image in memory.
  #        originalHeight: Height of the loaded image in memory
  #        paintedWidth: Width of the showed image. May be a 
  #                      value higher or lower than the original 
  #                      image.
  #        paintedHeight: Height of the showed image. 
  #                       May be a value higher or lower 
  #                       than the original image.   
  # Return:
  #        Euclidean distance between (sX, sY) and (eX, eY) 
  #        considering that the displayed image may be larger 
  #        or smaller than the original one.
  #------------------------------------------------------------
  def calculateDistanceResized(self, sX, sY, eX, eY, originalWidth, originalHeight, paintedWidth, paintedHeight):

    #obtain relation between showed image and loade image in memory (original image)
    Original_sX = (sX/paintedWidth)*originalWidth
    Original_sY = (sY/paintedHeight)*originalHeight
    
    Original_eX = (eX/paintedWidth)*originalWidth
    Original_eY = (eY/paintedHeight)*originalHeight
       
    difX = Original_eX-Original_sX
    difY = Original_eY-Original_sY

    # compute euclidean distance    
    euclideanDistance = math.sqrt(pow(difX,2) + pow(difY,2))

    return euclideanDistance
    

  #------------------------------------------------------------
  # Function to calculate the calue of a pixel in cm2 and update 
  # this information in the main interface
  #
  # Params:
  #        entry: centimeters of the reference in real life
  #        distance: distance between the start and end point of the line
  # Return:
  #        None
  #------------------------------------------------------------
  def calculeReferenceValue(self, entry, distance):
    # Get value of a pixel in cm2
    self.referenceValue = pow(float(entry.get()) / distance, 2)
    
    # Update the information on main interface.
    self.cmToPixelRelation.delete(0,tk.END)
    self.cmToPixelRelation.insert(0,str(self.referenceValue))
    

  #------------------------------------------------------------
  # Function to mark the reference in an image. This function opens
  # a windows to select the reference element in the image
  #
  # Params:
  #        None
  # Return:
  #        None
  #------------------------------------------------------------
  def find_reference(self):
    # title and instructions of the new opened window
    cv2.namedWindow("Select a line defining the reference in the image", cv2.WINDOW_NORMAL) 
    
    # callback to manage mouse clicks
    mouseLineCoordinates = openCVmouseEventStore()
    cv2.setMouseCallback('Select a line defining the reference in the image', mouseLineCoordinates.line_drawing)  

    # keep showing images on the windows until the windows is closed
    while True:
      clearImage = cv2.cvtColor(self.opencvImg, cv2.COLOR_RGB2BGR) # make a new copy of the image to draw the new position of the line
      cv2.line(clearImage, (mouseLineCoordinates.s_x, mouseLineCoordinates.s_y), (mouseLineCoordinates.e_x, mouseLineCoordinates.e_y), color=(0,0,255), thickness=12)
      cv2.imshow("Select a line defining the reference in the image", clearImage)
      cv2.waitKey(10)
      
      # if the window is not visible so it is closed, then exit the loop section
      if cv2.getWindowProperty("Select a line defining the reference in the image", cv2.WND_PROP_VISIBLE) <1:
        break
      (x, y, windowWidth, windowHeight) = cv2.getWindowImageRect("Select a line defining the reference in the image") 
          
    # call calculateDistanceResized fuction to calculate the distance between the start and end point of the line considering that the displayed image may be larger or smaller than the original one.
    distance = self.calculateDistanceResized(mouseLineCoordinates.s_x, mouseLineCoordinates.s_y, mouseLineCoordinates.e_x, mouseLineCoordinates.e_y, self.opencvImg.shape[1], self.opencvImg.shape[0], windowWidth, windowHeight)
    
    cv2.destroyWindow("Select a line defining the reference in the image") # destroy the image to select the reference

    
    insertCentimetersInRealLifeWindow = tk.Toplevel() # generate a new child window
    insertCentimetersInRealLifeWindow.grab_set() # keep focus in this new window and prevent to interact with the main window

    # calculate the relationship between the image and centimeters in real life.
    referenceFrame = tk.Frame(insertCentimetersInRealLifeWindow)
    referenceFrame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    tk.Label(referenceFrame, text="Measure in cm").pack(side=tk.LEFT, expand=True, fill=tk.X)
    self.cmEntry = tk.Entry(insertCentimetersInRealLifeWindow)
    self.cmEntry.pack(expand=True, fill=tk.X)
    
    # get the value of a pixel in cm2
    cmButtom = tk.Button(insertCentimetersInRealLifeWindow, text='Click to apply', command=lambda: self.calculeReferenceValue(self.cmEntry, distance))
    cmButtom.pack()
    
    tk.Button(insertCentimetersInRealLifeWindow, text='Exit', command=insertCentimetersInRealLifeWindow.destroy).pack() # add a button to go back to main window and close window 
    

  #------------------------------------------------------------
  # Function to split the images. This function make a temp folder
  # to store each image tile.
  #
  # Params:
  #        path: path of the image to make a temp floder
  #        split_width: desired width of the tile
  #        split_height: desired height of the tile  
  # Return:
  #        None
  #------------------------------------------------------------
  def splitImages(self, path, split_width, split_height):
    # parse the path to find the right format depending of OS
    pathParser = Path(path)

    pathFolder = pathParser.parents[0]
    imageName = pathParser.name

    valid_images_format = [".jpg",".jpeg",".png"] # supported image format
    self.splittedPath = pathFolder / "splittedImages" # temp folder to store the tiles
    
    nameadd = 'splitted' # partly name of the tiles
    frmt = 'JPG' # format of the tiles
    
    # code to create a directory. If the directory exists it throws an exception, so it is put in a try-except.
    try:
      os.makedirs(self.splittedPath)
    except FileExistsError:
      # directory already exists
      pass

    # obtaining the neame of the image to be used partially for the name of the tiles
    name = os.path.splitext(imageName)[0]
    img = cv2.imread(pathFolder/imageName)
    img_h, img_w, _ = img.shape

    # make the tiles using the auxiliary function start_points and save the result in the temp folder
    X_points = start_points(img_w, split_width, 0.5)
    Y_points = start_points(img_h, split_height, 0.5)

    count = 0

    for i in Y_points:
      for j in X_points:
        split = img[i:i+split_height, j:j+split_width]
        saveTileName = str(self.splittedPath / name) + '_{}_{:0=5}.{}'.format(nameadd, count, frmt)
        cv2.imwrite(saveTileName, split) 
        count += 1
    

  #------------------------------------------------------------
  # Function to merge the images
  #
  # Params:
  #        path: path of the image to load temp folder created 
  #              in the split function
  #        split_width: width of the tile
  #        split_height: height of the tile  
  # Return:
  #        None
  #------------------------------------------------------------
  def mergeImages(self, path, split_width, split_height):
    # parse the path to find the right format depending of OS  
    pathParser = Path(path)

    pathFolder = pathParser.parents[0]
    imageName = pathParser.name
    
    valid_images_format = [".jpg",".jpeg",".png"] # supported image format

    img = cv2.imread(pathFolder/imageName) # load the original image
    img_h, img_w, _ = img.shape # find the shape of the imahe
    final_image = np.zeros_like(img) # create a image filled with zeros

    # find the points where the marge is necesary
    X_points = start_points(img_w, split_width, 0.5)
    Y_points = start_points(img_h, split_height, 0.5)

    # obtaining the name of the image without extension and find all files that share the name
    image_name = os.path.splitext(imageName)[0] 
    image_ext = (os.path.splitext(imageName)[1])[1:]
    splitted_images_list = get_file_names_with_strings(str(pathFolder/"splittedImages"/"predictions"), image_name) 
    
    splitted_images_loaded_in_memory = [] # list of tiles to load in memory
  
    # fill the splitted_images_loaded_in_memory with all the tiles of the image
    for splitted_image in splitted_images_list:
      splitted_images_loaded_in_memory.append(cv2.imread(str(pathFolder/"splittedImages"/"predictions"/splitted_image)))

    # marge all the tiles
    index = 0
    for i in Y_points:
      for j in X_points:
        final_image[i:i+split_height, j:j+split_width] = cv2.bitwise_or(final_image[i:i+split_height, j:j+split_width],splitted_images_loaded_in_memory[index])
        index += 1        

    # save the mereg image in a new variable. This merge image is the mask that represent the pixles with honey
    self.opencvMask = final_image.copy()
    

  #------------------------------------------------------------
  # Function to calculate the surface of honey in an image. This
  # function update the interface information
  #
  # Params:
  #        mask: merged mask here positive values are pixels
  #              with honey
  # Return:
  #        None
  #------------------------------------------------------------
  def calculateAreaofHoney(self, mask):
    # count the number of piels that contains honey
    numberofWhitePixels = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    
    # find the surface in cm2 of honey with 4 decimals
    self.areaOfHoney = round(float(self.cmToPixelRelation.get()) * numberofWhitePixels, 4)
    
    # update the information in the interface
    self.varLabelInformationText.set("Image Name:" + self.imgPath + "\nImage Size: " + str(self.opencvImg.shape[0]) + "x" + str(self.opencvImg.shape[1]) + "\nArea of honey: " + str(self.areaOfHoney) + "cm²")
    self.update()
    

  #------------------------------------------------------------
  # Function to blend the mask and the loaded image. This function
  # update the interface to show the result of the segmentation process
  #
  # Params:
  #        img: original image
  #        mask: mask of pixels representing the honey
  # Return:
  #        None
  #------------------------------------------------------------
  def applySegmentation(self, img, mask):
    # make an array with a value of red pixel
    color = np.array([255,0,0], dtype='uint8')
    
    # replace all white pixel in mask with a red pixel
    masked_img = np.where(mask, color, img)

    # blend image and mask woth red pixels with 60% of the original image and 40% of the mask
    self.opencvMaskApply  = cv2.addWeighted(img, 0.6, masked_img, 0.4,0)

    # update the content of the image showed in the interface as the result of the segmentation
    self.imgDefaultProcess = cv2.resize(self.opencvMaskApply, (830, 680))
 
    # Convert image to tkinter image format and display
    self.tkimageSegmented = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.imgDefaultProcess))
    self.lavelSegmentedImage.config(image=self.tkimageSegmented)
    self.update()

  #------------------------------------------------------------
  # Function that saves the result of the segmentation process.
  # This function saves the mask and the image with the mask 
  # applied
  #
  # Params:
  #        None
  # Return:
  #        None
  #------------------------------------------------------------
  def saveResults(self):
    # ask for a folder to save the results
    saveFolder = Path(tk.filedialog.askdirectory())
       
    pathParser = Path(self.imgPath)
    imageName = pathParser.name
    
    # obtaining the name of the image without extension
    image_name = os.path.splitext(imageName)[0] #obtenemos el nombre de la imagen sin extension
    
    # obtaining the extension of the image without the dot
    image_ext = (os.path.splitext(imageName)[1])[1:]

    # save the mask and the image with the blended mask represented with red pixels with honey
    cv2.imwrite('{}_{}.{}'.format(str(saveFolder / image_name), "honey-Mask", image_ext), self.opencvMask)
    cv2.imwrite('{}_{}.{}'.format(str(saveFolder / image_name), "honey-Highlighted", image_ext), cv2.cvtColor(self.opencvMaskApply, cv2.COLOR_BGR2RGB))
    

  #------------------------------------------------------------
  # Function to perform the segmentation process. This function 
  # split an image into tiles, process each tile and merge the 
  # result of each tile into a new image with the same size of
  # the original loaded image
  #
  # Params:
  #        IMG_HEIGHT: Height of a tile.
  #        IMG_WIDTH: Width of a tile.
  #        IMG_CHANNELS: Number of channels of a tile
  # Return:
  #        None
  #------------------------------------------------------------
  def segmentationProcess(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    # init the progressbar
    self.progressBar.start()

    # split images in tiles
    self.splitImages(self.imgPath, IMG_HEIGHT, IMG_WIDTH)

    # fill the progressbar with 10%    
    self.progressBar['value'] = 10
    self.update_idletasks()  
    
    # path to store the tile predictions
    resultPath = self.splittedPath / "predictions"
    
    reconstructed_model = keras.models.load_model(self.honeySegmentationModelPath) # load the keras model to perform the segmentation
    valid_images_format = [".jpg",".jpeg",".png"] # supported image formats

    frmt = 'png' # format for saving processed tiles

    # code to create a directory. If the directory exists it throws an exception, so it is put in a try-except.
    try:
      os.makedirs(resultPath)
    except FileExistsError:
      # directory already exists
      pass

    image_list = [] # variable to save the list of images to process

    # we read the list of files in a directory
    for filename in os.listdir(self.splittedPath):
      ext = os.path.splitext(filename)[1] # we keep the extension of each of the files
  
      if ext.lower() not in valid_images_format: # we pass the extension to lowercase and check if the extension is in the list
        continue # if not on the list we continue
      image_list.append(filename) # if it is in the list of supported formats, we save the image name in the list

    cont = 1 # counter for the progressbar 
    
    # loop to iterate over all the tiles of the image
    for imageName in image_list:
      # load a tile
      test_image = read_image(str(self.splittedPath / imageName), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) 

      test_image = np.expand_dims(test_image, axis = 0)

      #predict the result
      prediction = reconstructed_model.predict(test_image)
      prediction = (prediction > 0.5).astype(np.uint8) 

      i3 = prediction[0, :, :, 0] 
      i3 = (i3*255).astype(np.uint8) # scale to 0-255  range and convert to int
      nameImageToSave = os.path.splitext(imageName)[0]
      cv2.imwrite('{}.{}'.format(str(resultPath / nameImageToSave), frmt), i3) #save the result of the segmentation process

      # fill the progressbar with values between 10% to 70% that is the computational core of the segmentation process
      self.progressBar['value'] = round(70.0 * cont / float(len(image_list)), 1) + 10
      self.update_idletasks()  

      cont = cont + 1

    # merge the result of the tiles
    self.mergeImages(self.imgPath, IMG_HEIGHT, IMG_WIDTH)
    
    # fill the progressbar with 80%    
    self.progressBar['value'] = 80
    self.update_idletasks()  
    self.applySegmentation(self.opencvImg, self.opencvMask)

    # fill the progressbar with 90%    
    self.progressBar['value'] = 90
    self.update_idletasks()  
    self.calculateAreaofHoney(self.opencvMask)
    
    # remove the temp directory of the tiles 
    shutil.rmtree(str(self.splittedPath), ignore_errors = True) 
    
    # fill the progressbar with 100%
    self.progressBar['value'] = 100
    self.update_idletasks()   
    
    # stop the progressbar animation
    self.progressBar.stop()
    
    # enable the save button
    self.saveButtom["state"] = "normal"


#main call to the class of the graphical interface
HoneySegmentationToolGUI().mainloop()
