#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox,QFileDialog, QComboBox, QDateTimeEdit,QListWidget,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,QGraphicsView,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,QGraphicsScene,
        QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtWidgets import *
import color_percent
import cv2
import os
import time
current_fname = ''

class ThreadClass(QThread):
    def __init__(self, parent = None):
        super(ThreadClass, self).__init__(parent)

    def run(self):
        # self.mainclass = WidgetGallery()
        # self.mainclass.process_image()
        val = 'E:/Work/dithered_color_percent/images/dithered.png'
        print("hello")
        # self.emit(QtCore.SIGNAL('img_path'), val)
        # time.sleep(1)

    def process_image(self):
        # print(self.list_of_files)
        # print(self.img_to_process)
        if len(self.list_of_files) < 1:
            QMessageBox.about(self, "Alert", "Please Choose Image File To Proceed Further !!")

        else:
            # print(self.list_of_files[0], number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity)

            if self.ditherRadioButton.isChecked():
                # print("dithering")
                color_n = self.ditherColor.value()
                if len(self.img_to_process) < 2:
                    self.img_to_process = self.list_of_files[0]

                p = self.img_to_process
                f_name, file_extension = os.path.splitext(p)
                if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.JPG':
                    temp = cv2.imread(p)
                    p = f_name + '.png'
                    cv2.imwrite(p, temp)
                try:
                    path = color_percent.dither(p, color_n)
                except:
                    QMessageBox.about(self, "Alert", "The Number of Colors defined in Dithering is more than the colors in Image Itself!!")
                    return

                # print('From here::', path)
                pixmap = QPixmap(path)
                dithersmaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
                self.segmented_img_label.setPixmap(dithersmaller_pixmap)
                self.superpixels(path)
            else:
                if len(self.img_to_process) < 2:
                    self.img_to_process = self.list_of_files[0]
                self.superpixels(self.img_to_process)

    def superpixels(self, img_path):
        try:
            number_of_segments = self.segments_number.value()
            sigma_value = self.sigma.value()
            compactness_value = self.compactness.value()
            color_pocket_number = self.quant_levels.value()
            if self.checkBox.isChecked():
                connectivity = True
            else:
                connectivity = False
            replaced_img, segmented_image, time_elapsed = color_percent.main(img_path, number_of_segments, connectivity,
                                                               compactness_value, sigma_value, color_pocket_number)
            # display_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
            display_replaced_image = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)

            pixmap = QPixmap(self.convert_to_QImage(display_replaced_image))
            smaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.final_image_label.setPixmap(smaller_pixmap)

        # pixmap2 = QPixmap(self.convert_to_QImage(display_segmented_image))
        # smaller_pixmap2 = pixmap2.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
        # self.segmented_img_label.setPixmap(smaller_pixmap2)
            info = "Time Taken to Process Image = " + str(time_elapsed)
            self.time.setText(info)

        except:
            QMessageBox.about(self, "Alert",
                              "This error may be due to large number of Colors in Input image. Try Reducing color by Dithering.")

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

        disableWidgetsCheckBox = QCheckBox("&Disable widgets")

        self.createImageShower()
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()
        # self.createProgressBar()

        styleComboBox.activated[str].connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        disableWidgetsCheckBox.toggled.connect(self.topLeftGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.topRightGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomLeftTabWidget.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomRightGroupBox.setDisabled)

        topLayout = QHBoxLayout()
        # topLayout.addWidget(styleLabel)
        # topLayout.addWidget(styleComboBox)
        # topLayout.addStretch(1)
        # topLayout.addWidget(self.useStylePaletteCheckBox)
        # topLayout.addWidget(disableWidgetsCheckBox)
        self.ImageShower.setFixedWidth(100)
        self.topLeftGroupBox.setFixedWidth(275)
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.ImageShower,1,0)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 1)
        mainLayout.addWidget(self.topRightGroupBox, 1, 2)
        mainLayout.addWidget(self.bottomLeftTabWidget, 1, 3)
        mainLayout.addWidget(self.bottomRightGroupBox, 1, 4)
        # mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        # mainLayout.setRowStretch(1, 1)
        # mainLayout.setRowStretch(2, 1)
        # mainLayout.setColumnStretch(0, 1)
        # mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)
        self.list_of_files = []
        self.img_to_process = ''
        self.setWindowTitle("Dithering")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def itemClicked(self,item):
        # print(item.text())
        location = int(item.text())
        # print(location)
        new_img_path  = self.list_of_files[location]
        # print(new_img_path)
        self.img_to_process = new_img_path
        # print(self.list_of_files[loacation])
        self.display_image(new_img_path)

        self.process_image()

    def print_fun(self, val):
        print(val)

    def open(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Image File", "",
                                                "All Files (*.*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        if files:
            self.list_of_files.clear()
            self.list_of_files = files
            self.img_to_process = files[0]
            self.display_image(files[0])
            self.ImageShowerList.clear()
            for i,each in enumerate(files):
                itm = QListWidgetItem(str(i))
                itm.setIcon(QIcon(each))
                self.ImageShowerList.addItem(itm)

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)
    def disable(self):
        self.processPushButton.setEnabled(False)

    def display_image(self, img_path):
        pixmap = QPixmap(img_path)
        smaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.img_label.setPixmap(smaller_pixmap)

    def convert_to_QImage(self,cvImg):
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def superpixels(self, img_path):
        try:
            print('Processing Image :', img_path)

            number_of_segments = self.segments_number.value()
            sigma_value = self.sigma.value()
            compactness_value = self.compactness.value()
            color_pocket_number = self.quant_levels.value()
            if self.checkBox.isChecked():
                connectivity = True
            else:
                connectivity = False
            replaced_img, segmented_image, time_elapsed = color_percent.main(img_path, number_of_segments, connectivity,
                                                               compactness_value, sigma_value, color_pocket_number)
            # display_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
            display_replaced_image = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)

            pixmap = QPixmap(self.convert_to_QImage(display_replaced_image))
            smaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.final_image_label.setPixmap(smaller_pixmap)

        # pixmap2 = QPixmap(self.convert_to_QImage(display_segmented_image))
        # smaller_pixmap2 = pixmap2.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
        # self.segmented_img_label.setPixmap(smaller_pixmap2)
            info = "Time Taken to Process Image = " + str(time_elapsed)
            self.time.setText(info)

        except:
            QMessageBox.about(self, "Alert",
                              "This error may be due to large number of Colors in Input image. Try Reducing color by Dithering.")

    def enable_dither_color(self):
        if self.ditherRadioButton.isChecked():
            self.ditherColor.setDisabled(False)
        else:
            self.ditherColor.setEnabled(False)

    def process_image(self):
        # print(self.list_of_files)
        # print(self.img_to_process)
        if len(self.list_of_files) < 1:
            QMessageBox.about(self, "Alert", "Please Choose Image File To Proceed Further !!")

        else:
            # print('Processing')
            # print(self.list_of_files[0], number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity)

            if self.ditherRadioButton.isChecked():
                # print("dithering")
                color_n = self.ditherColor.value()
                if len(self.img_to_process) < 2:
                    self.img_to_process = self.list_of_files[0]

                p = self.img_to_process
                f_name, file_extension = os.path.splitext(p)
                if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.JPG':
                    temp = cv2.imread(p)
                    p = f_name + '.png'
                    cv2.imwrite(p, temp)
                try:
                    path = color_percent.dither(p, color_n)
                except:
                    QMessageBox.about(self, "Alert", "The Number of Colors defined in Dithering is more than the colors in Image Itself!!")
                    return

                # print('From here::', path)
                pixmap = QPixmap(path)
                dithersmaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
                self.segmented_img_label.setPixmap(dithersmaller_pixmap)
                self.superpixels(path)
            else:
                if len(self.img_to_process) < 2:
                    self.img_to_process = self.list_of_files[0]
                self.superpixels(self.img_to_process)

    def process_all_images(self):
        for each in self.list_of_files:
            self.img_to_process = each
            self.process_image()
        self.display_image(self.img_to_process)
        print('Finished Processing Output are in stored in folders')

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)

    def createImageShower(self):
        self.ImageShower = QGroupBox("Loaded Images")
        self.ImageShowerList = QListWidget()
        self.ImageShowerList.itemClicked.connect(self.itemClicked)


        layout = QVBoxLayout()
        layout.addWidget(self.ImageShowerList)
        #
        # layout.addStretch(1)
        self.ImageShower.setLayout(layout)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Parameters")

        defaultPushButton = QPushButton("Choose Images")
        defaultPushButton.clicked.connect(self.open)

        self.ditherRadioButton = QRadioButton("Dither and then Process")
        self.ditherRadioButton.setChecked(True)
        self.ditherRadioButton.clicked.connect(self.enable_dither_color)
        self.ditherColor = QSpinBox()
        self.ditherColor.setMaximum(15)
        self.ditherColor.setMinimum(2)
        self.ditherColor.setValue(4)
        # self.ditherColor.setEnabled(False)

        self.checkBox = QCheckBox("Connectivity Between Segments")
        self.checkBox.setEnabled(True)

        self.segments_number = QSpinBox()
        self.segments_number.setMaximum(10000)
        self.segments_number.setValue(100)

        self.compactness = QSpinBox()
        self.compactness.setValue(3)
        self.compactness.setMinimum(0)

        self.sigma = QSpinBox()
        self.sigma.setValue(3)

        self.quant_levels = QSpinBox()
        self.quant_levels.setMaximum(5)
        self.quant_levels.setMinimum(1)
        self.quant_levels.setValue(4)

        dither_color_label = QLabel("Number of Colors to be Used in Dithering")
        segment_label = QLabel("Number of Segments")
        compactness_label = QLabel("Compactness:Larger Value,Makes Square Segments")
        sigma_label = QLabel("Sigma:Size of Gussian Filter Kernel")
        quant_label = QLabel("Color:Define Number of Pockets of Color to be Used")
        self.time = QLabel()

        self.processPushButton = QPushButton("Batch Process All Images")
        # self.processPushButton.clicked.connect(self.processPushButton.setDisabled)
        # self.processPushButton.clicked.connect(self.disable)
        self.processPushButton.clicked.connect(self.process_all_images)

        layout = QVBoxLayout()


        layout.addWidget(defaultPushButton)
        layout.addWidget(self.ditherRadioButton)
        layout.addWidget(dither_color_label)
        layout.addWidget(self.ditherColor)

        layout.addWidget(self.checkBox)
        layout.addWidget(segment_label)
        layout.addWidget(self.segments_number)
        layout.addWidget(compactness_label)
        layout.addWidget(self.compactness)
        layout.addWidget(sigma_label)
        layout.addWidget(self.sigma)
        layout.addWidget(quant_label)
        layout.addWidget(self.quant_levels)

        layout.addWidget(self.processPushButton)
        layout.addWidget(self.time)


        # layout.addStretch(0)
        self.topLeftGroupBox.setLayout(layout)

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Image To Process")

        self.img_label = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.img_label)

        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QGroupBox("Dithered Image")
        self.segmented_img_label = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.segmented_img_label)

        layout.addStretch(1)
        self.bottomLeftTabWidget.setLayout(layout)

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Final Result")
        self.final_image_label = QLabel(self)


        layout = QVBoxLayout()
        layout.addWidget(self.final_image_label)

        layout.addStretch(1)
        self.bottomRightGroupBox.setLayout(layout)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)



if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_()) 
