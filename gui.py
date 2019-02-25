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

from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox,QFileDialog, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,QGraphicsView,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,QGraphicsScene,
        QVBoxLayout, QWidget, QMessageBox)
import color_percent
import cv2
import os

class Thread(QThread):
    def run(self):
        QThread.sleep(2)

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

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        # mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)
        self.list_of_files = []
        self.setWindowTitle("Dithering")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def open(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose Image File", "",
                                                "All Files (*.*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        if files:
            self.list_of_files = files
            self.display_image(files[0])

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
            display_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
            display_replaced_image = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)

            pixmap = QPixmap(self.convert_to_QImage(display_replaced_image))
            smaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.final_image_label.setPixmap(smaller_pixmap)

            pixmap2 = QPixmap(self.convert_to_QImage(display_segmented_image))
            smaller_pixmap2 = pixmap2.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.segmented_img_label.setPixmap(smaller_pixmap2)
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


        if len(self.list_of_files) < 1:
            QMessageBox.about(self, "Alert", "Please Choose Image File To Proceed Further !!")

        else:
            # print('Processing')
            # print(self.list_of_files[0], number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity)

            if self.ditherRadioButton.isChecked():
                # print("dithering")
                color_n = self.ditherColor.value()
                p = self.list_of_files[0]
                f_name, file_extension = os.path.splitext(p)
                if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.JPG':
                    temp = cv2.imread(p)
                    p = f_name + '.png'
                    cv2.imwrite(p, temp)

                path = color_percent.dither(p, color_n)
                print(path)
                pixmap = QPixmap(path)
                dithersmaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
                self.img_label.setPixmap(dithersmaller_pixmap)
                self.superpixels(path)
            else:

                self.superpixels(self.list_of_files[0])

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Parameters")

        defaultPushButton = QPushButton("Choose Image")
        defaultPushButton.clicked.connect(self.open)

        self.ditherRadioButton = QRadioButton("Dither and then Process")
        self.ditherRadioButton.clicked.connect(self.enable_dither_color)
        self.ditherColor = QSpinBox()
        self.ditherColor.setMaximum(15)
        self.ditherColor.setMinimum(2)
        self.ditherColor.setValue(4)
        self.ditherColor.setEnabled(False)

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
        compactness_label = QLabel("Compactness: More the Value, Segments will be More Square")
        sigma_label = QLabel("Sigma: Size of Gussian Filter Kernel")
        quant_label = QLabel("Color Combination: Define Number of Pockets of Color to be Used")
        self.time = QLabel()

        self.processPushButton = QPushButton("Process Image")
        # self.processPushButton.clicked.connect(self.processPushButton.setDisabled)
        # self.processPushButton.clicked.connect(self.disable)
        self.processPushButton.clicked.connect(self.process_image)

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


        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)    

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Image To Process")


        self.img_label = QLabel(self)


        layout = QVBoxLayout()
        layout.addWidget(self.img_label)

        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QGroupBox("Segmented Image")
        self.segmented_img_label = QLabel(self)

        # self.bottomLeftTabWidget = QTabWidget()
        # self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
        #         QSizePolicy.Ignored)
        #
        # tab1 = QWidget()
        # tableWidget = QTableWidget(10, 10)
        #
        # tab1hbox = QHBoxLayout()
        # tab1hbox.setContentsMargins(5, 5, 5, 5)
        # tab1hbox.addWidget(tableWidget)
        # tab1.setLayout(tab1hbox)
        #
        # tab2 = QWidget()
        # textEdit = QTextEdit()
        #
        # textEdit.setPlainText("Twinkle, twinkle, little star,\n"
        #                       "How I wonder what you are.\n"
        #                       "Up above the world so high,\n"
        #                       "Like a diamond in the sky.\n"
        #                       "Twinkle, twinkle, little star,\n"
        #                       "How I wonder what you are!\n")
        #
        # tab2hbox = QHBoxLayout()
        # tab2hbox.setContentsMargins(5, 5, 5, 5)
        # tab2hbox.addWidget(textEdit)
        # tab2.setLayout(tab2hbox)
        #
        # self.bottomLeftTabWidget.addTab(tab1, "&Table")
        # self.bottomLeftTabWidget.addTab(tab2, "Text &Edit")

        layout = QVBoxLayout()
        layout.addWidget(self.segmented_img_label)

        layout.addStretch(1)
        self.bottomLeftTabWidget.setLayout(layout)

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Final Result")
        self.final_image_label = QLabel(self)
        # self.bottomRightGroupBox.setCheckable(True)
        # self.bottomRightGroupBox.setChecked(True)
        #
        # lineEdit = QLineEdit('s3cRe7')
        # lineEdit.setEchoMode(QLineEdit.Password)
        #
        # spinBox = QSpinBox(self.bottomRightGroupBox)
        # spinBox.setValue(50)
        #
        # dateTimeEdit = QDateTimeEdit(self.bottomRightGroupBox)
        # dateTimeEdit.setDateTime(QDateTime.currentDateTime())
        #
        # slider = QSlider(Qt.Horizontal, self.bottomRightGroupBox)
        # slider.setValue(40)
        #
        # scrollBar = QScrollBar(Qt.Horizontal, self.bottomRightGroupBox)
        # scrollBar.setValue(10)
        #
        # dial = QDial(self.bottomRightGroupBox)
        # dial.setValue(30)
        # dial.setNotchesVisible(True)
        #
        # layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)
        # layout.addWidget(spinBox, 1, 0, 1, 2)
        # layout.addWidget(dateTimeEdit, 2, 0, 1, 2)
        # layout.addWidget(slider, 3, 0)
        # layout.addWidget(scrollBar, 4, 0)
        # layout.addWidget(dial, 3, 1, 2, 1)
        # layout.setRowStretch(5, 1)
        # self.bottomRightGroupBox.setLayout(layout)

        layout = QVBoxLayout()
        layout.addWidget(self.final_image_label)

        layout.addStretch(1)
        self.bottomRightGroupBox.setLayout(layout)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        # timer = QTimer(self)
        # timer.timeout.connect(self.advanceProgressBar)
        # timer.start(1000)

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_()) 
