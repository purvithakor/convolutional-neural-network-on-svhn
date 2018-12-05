README FOR CAFFE

There are 3 Python scripts & 2 prototxt files present here in this folder.

#########################################################################################################

Setting Environment Variables - 

PYTHONPATH=/home/ubuntu/caffe/python
DISPLAY=localhost:10.0

Run the scripts in the following order in your home directory - 

01_DataManipulation.py---(PYTHON 3)
02_CreateLMDB.py---------(PYTHON 2.7)
03_SVHNCaffe.py----------(PYTHON 2.7)

#########################################################################################################

01_DataManipulation.py - Pulls data from the Stanford University's website (http://ufldl.stanford.edu/housenumbers/) and cleans/preprocesses the dataset. 

02_CreateLMDB.py - Pulls the output of the previous script and converts the arrays to images and stores the images in the format in which Caffe expects for the LMDB structure.

03_SVHNCaffe.py - Pulls the output of the previous script and runs the CNN model.

#########################################################################################################