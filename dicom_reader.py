import math
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QListWidget, QVBoxLayout, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QPainter, QCursor, QPainterPath
from PyQt5.QtCore import pyqtSlot, QRect, Qt, QPoint, QPointF
from pydicom import dcmread
import qimage2ndarray
import cv2
import numpy as np
import os

W_WIDTH = 1800
W_HEIGHT = 1050
W_X = 10
W_Y = 30


class ImageLabel(QLabel):
    mx_pts = np.zeros(10)
    my_pts = np.zeros(10)
    mz_pts = np.zeros(10)
    n_pts = 0
    x_pixel_spacing = 0.0
    y_pixel_spacing = 0.0
    scale_factor = 4
    image_height = 0            # number of rows in pixel data
    image_width = 0             # number of cols in pixel data
    # http://dicomiseasy.blogspot.com/2012/08/chapter-12-pixel-data.html
    # if == ' "MONOCHROME2" meaning its grayscale and 0 should be interpreted as Black.'
    photometric_interpretation = ''
    samples_per_pixel = 0   # defines the number of color channels
    """ From: https://nipy.org/nibabel/dicom/dicom_orientation.html
     ‘Image Orientation Patient’ are the direction cosine for the ‘positive row axis’. 
     That is, they express the direction change in (x, y, z), in the DICOM patient coordinate system (DPCS), 
     as you move along the row. That is, as you move from one column to the next. That is, as the 
     column array index changes. Similarly, the second triplet of values of 
     ‘Image Orientation Patient’ (img_ornt_pat[3:] in Python), 
     are the direction cosine for the ‘positive column axis’, and express the direction you move, in the DPCS, 
     as you move from row to row, and therefore as the row index changes.
    """
    image_orientation_patient = np.zeros((3, 2))

    image_position_patient = np.zeros(3)
    """  From https://nipy.org/nibabel/dicom/dicom_orientation.html
    The Image Position (0020,0032) specifies the x, y, and z coordinates of the upper left hand corner
    of the image; it is the center of the first voxel transmitted.
    """
    """
    http://dicomiseasy.blogspot.com/2012/08/chapter-12-pixel-data.html
    Rows and Columns
Rows (0028,0010) and Columns (0028,0011) define the size of the image. 
Rows is the height (i.e. the Y) and Columns is the width (i.e. the X). In our example every frame is 1280 x 960 pixels. We'll see what is frame in a minute.
Samples Per Pixel
Samples per pixel (0028,0002)define the number of color channels. 
In grayscale images like CT and MR it is set to 1 for the single grayscale channel 
and for color images like in our case it is set to 3 for the three color channels Red, 
Green and Blue.
Photometric Interpretation
The photometric interpretation (0028,0004) element is rather unique to DICOM.
 It defines what does every color channel hold. You may refer it to the color space used to encode the image. In our example it is "RGB" meaning the first channel ir Red, the second is Green and the third is Blue. In grayscale images (like CT or MR) it is usually "MONOCHROME2" meaning its grayscale and 0 should be interpreted as Black. In some objects like some fluoroscopic images it may be "MONOCHROME1" meaning its grayscale and 0 should be interpreted as White. Other values may be "YBR_FULL" or "YBR_FULL_422" meaning the color channels are in the YCbCr color space that is used in JPEG.
Planar configuration
Planar configuration (0028,0006) defines how the color channels are arranged in the pixel data buffer. It is relevant only when Samples Per Pixel > 1 (i.e. for color images). It can be either 0 meaning the channels are interlaced which is the common way of serializing color pixels or 1 meaning its separated i.e. first all the reds, then all the greens and then all the blues like in print. The separated way is rather rare and when it is used its usually with RLE compression. The following image shows the two ways. BTW, If this element is missing, the default is interlaced.
    """

    def mousePressEvent(self, e):
        pos = e.pos()  # returns QtCore.QPoint()
        x = e.x()
        y = e.y()
        #self.lblMouseCoords.setText("Mouse Press Event X: {:.3f}   Y: {:.3f}".format(e.x(), e.y()))
        #p = QtCore.QPointF(e.posF.x(), e.posF.y())
        #p = QtCore.QPoint(e.pos(), e.pos())
        print("Mouse Press Event X: {:.3f}   Y: {:.3f}".format(e.x(), e.y()))
        self.mx_pts[self.n_pts] = pos.x()
        self.my_pts[self.n_pts] = pos.y()
        self.scale_point()
        print("Scaled (mm) X: {:.3f}   Y: {:.3f}   Z: {:.3f}".format(self.mx_pts[self.n_pts], self.my_pts[
            self.n_pts], self.mz_pts[self.n_pts]))
        self.n_pts += 1

    def scale_point(self):
        M = np.zeros((4, 4), dtype=np.float64)
        v = np.zeros((4, 1), dtype=np.float64)
        M[0, 0] = self.image_orientation_patient[3] * \
            self.y_pixel_spacing / self.scale_factor
        M[1, 0] = self.image_orientation_patient[4] * \
            self.y_pixel_spacing / self.scale_factor
        M[2, 0] = self.image_orientation_patient[5] * \
            self.y_pixel_spacing / self.scale_factor
        M[0, 1] = self.image_orientation_patient[0] * \
            self.x_pixel_spacing / self.scale_factor
        M[1, 1] = self.image_orientation_patient[1] * \
            self.x_pixel_spacing / self.scale_factor
        M[2, 1] = self.image_orientation_patient[2] * \
            self.x_pixel_spacing / self.scale_factor
        M[3, 0] = self.image_position_patient[0]
        M[3, 1] = self.image_position_patient[1]
        M[3, 2] = self.image_position_patient[2]
        v[0] = self.mx_pts[self.n_pts]
        v[1] = self.my_pts[self.n_pts]
        v[3] = 1.0
        C = np.dot(M, v)  # column vector x matrix
        self.mx_pts[self.n_pts] = C[0, 0]
        self.my_pts[self.n_pts] = C[1, 0]
        self.mz_pts[self.n_pts] = C[2, 0]
        # print(C)

    def read_dicom_image(self, file_name):
        path = 'D:/2022_Dicom_Sorted/nelson01082021/20210108/#-#_data_s168314_3tbcardiac_3tb7831/mid_sax_cine/'
        #path = 'C:/Users/mdr24/OneDrive - University of Texas at Arlington/3TB8554b_Healthy/'
        fpath = path + file_name
        ds = dcmread(fpath)

        # Normal mode:
        print()
        print(f"File path........: {fpath}")
        print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
        print()

        pat_name = ds.PatientName
        display_name = pat_name.family_name + ", " + pat_name.given_name
        print(f"Patient's Name...: {display_name}")
        print(f"Patient ID.......: {ds.PatientID}")
        print(f"Modality.........: {ds.Modality}")
        print(f"Study Date.......: {ds.StudyDate}")
        print(f"Image size.......: {ds.Rows} x {ds.Columns}")
        print(f"Pixel Spacing....: {ds.PixelSpacing}")
        print(f"Image Position Patient..: {ds.ImagePositionPatient}")
        print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

        data = ds.pixel_array.astype(float)                         # get the pixel information into a numpy array
        self.x_pixel_spacing = ds.PixelSpacing[0]
        self.y_pixel_spacing = ds.PixelSpacing[1]
        print('The image has {} x {} voxels'.format(
            data.shape[0], data.shape[1]))
        self.image_height = data.shape[0] * self.scale_factor           # number of rows is the height of image
        self.image_width = data.shape[1] * self.scale_factor            # number of cols is the width of image
        self.photometric_interpretation = ds.PhotometricInterpretation
        self.samples_per_pixel = ds.SamplesPerPixel                     # 1 is grayscale, 3 is RGB
        self.image_orientation_patient = ds.ImageOrientationPatient     # direct cosine matrix
        self.image_position_patient = ds.ImagePositionPatient           # x,y,z position of first voxel (0,0) upper left corner
        zoomed = cv2.resize(data, (data.shape[0] * self.scale_factor, data.shape[1] * self.scale_factor),
                            interpolation=cv2.INTER_CUBIC)
        q_img = qimage2ndarray.array2qimage(zoomed, True)
        return q_img



class MainFrame(QMainWindow):
    def __init__(self):
        super(MainFrame, self).__init__()
        self.init()
        self.show()

    def init(self):
        self.resize(W_WIDTH, W_HEIGHT)
        self.move(W_X, W_Y)
        self.setWindowTitle("Applied Physiology and Advanced Imaging Laboratory | apailab.uta.edu ")

        self.lbl_img = ImageLabel(self)

        self.list_view = QListWidget(self)
        self.list_view.setGeometry(20, 60, 400, 800)
        self.list_view.itemClicked.connect(self.onClicked)

        self.button = QPushButton("Read Files", self)
        self.button.move(20, 10)
        self.button.clicked.connect(self.read_files)

    def read_files(self):
        path = 'D:/2022_Dicom_Sorted/nelson01082021/20210108/#-#_data_s168314_3tbcardiac_3tb7831/mid_sax_cine/'
        dir_list = os.listdir(path)
        for fn in dir_list:
            self.list_view.addItem(fn)

    def onClicked(self, item):
        #QMessageBox.information(self, "Info", item.text())
        file_name = item.text()
        q_img = self.lbl_img.read_dicom_image(file_name)
        self.lbl_img.setGeometry(
            QRect(500, 10, self.lbl_img.image_width, self.lbl_img.image_height))
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setCursor(Qt.CrossCursor)
        self.lbl_img.setMouseTracking(True)


app = QApplication(sys.argv)
win = MainFrame()
sys.exit(app.exec_())
