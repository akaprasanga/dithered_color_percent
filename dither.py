#!/usr/bin/env python

from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
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
import dither_algorithm
import dominant_color_track as clustering


class WidgetGallery(QDialog):
    add_post = pyqtSignal(str, str, str, str)
    reduce_color_Signal = pyqtSignal(str, str)
    sleep_Signal = pyqtSignal()

    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.another_process = WorkerThread()
        self.clustering_thread = Kmeans()
        self.delay_thread = SleepThread()
        self.timer_id = -1

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        self.createImageShower()
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()

        topLayout = QHBoxLayout()

        self.ImageShower.setFixedWidth(100)
        self.topLeftGroupBox.setFixedWidth(350)
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.ImageShower,1,1)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 2)
        mainLayout.addWidget(self.bottomLeftTabWidget, 1, 3)
        mainLayout.addWidget(self.bottomRightGroupBox, 1, 4)

        self.setLayout(mainLayout)
        self.list_of_files = []
        self.img_to_process = ''
        self.current_output_path = ''
        self.setWindowTitle("Dithering")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))

    def itemClicked(self, item):
        self.another_process.batch_process_flag = False
        self.disable_vitals()
        try:
            location = int(item.text())
        except AttributeError:
            return
        print("came here")
        new_img_path = self.list_of_files[location]
        self.another_process.current_path = new_img_path
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number)

        self.another_process.start()
        # self.disable_vitals()

        self.another_process.add_post.connect(self.tread_done)

        self.img_to_process = new_img_path
        self.display_image(new_img_path)
        # self.process_image()


    def itemDoubleClicked(self, column_no):
        # print(column_no)

        self.another_process.batch_process_flag = False
        self.disable_vitals()
        # location = int(item.text())
        new_img_path = self.img_to_process
        self.another_process.current_path = new_img_path
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number)

        self.another_process.start()
        # self.disable_vitals()

        self.another_process.add_post.connect(self.tread_done)

        self.img_to_process = new_img_path
        self.display_image(new_img_path)

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

    def get_status(self):
        dither_flag = False
        dither_color = 4
        if self.ditherRadioButton.isChecked():
            dither_flag = True
            dither_color = self.ditherColor.value()

        number_of_segments = self.segments_number.value()
        sigma_value = self.sigma.value()
        compactness_value = self.compactness.value()
        color_pocket_number = self.quant_levels.value()
        if self.checkBox.isChecked():
            connectivity = True
        else:
            connectivity = False

        if self.resizeButton.isChecked():
            resize_flag = True
            resize_factor = self.resizefactor.value()
        else:
            resize_flag = False
            resize_factor = 1
        reduce_color_number = self.kmeans_color_slider.value()

        return dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor, reduce_color_number

    def set_status_to_thread(self,dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor, reduce_color_number):
        self.another_process.dither_flag = dither_flag
        self.another_process.dither_color = dither_color
        self.another_process.number_of_segments = number_of_segments
        self.another_process.sigma_value = sigma_value
        self.another_process.compactness_value = compactness_value
        self.another_process.color_pocket_number = color_pocket_number
        self.another_process.connectivity = connectivity
        self.another_process.resize_flag = resize_flag
        self.another_process.resize_factor = resize_factor
        self.another_process.reduce_color_number = reduce_color_number

    def disable_vitals(self):
        # self.processPushButton.setDisabled(True)
        # self.defaultPushButton.setDisabled(True)
        self.topLeftGroupBox.setDisabled(True)

    def enable_vitals(self):
        # self.processPushButton.setEnabled(True)
        # self.defaultPushButton.setEnabled(True)
        self.topLeftGroupBox.setEnabled(True)

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    def display_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.img_label.adjustSize()
        w = self.topRightGroupBox.width()
        h = self.topRightGroupBox.height()
        # print(w,h)
        smaller_pixmap = pixmap.scaled(w-40,h-20,Qt.KeepAspectRatio, Qt.FastTransformation)
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
            w = self.topRightGroupBox.width()
            h = self.topRightGroupBox.height()
            smaller_pixmap = pixmap.scaled(w-40,h-20,Qt.KeepAspectRatio, Qt.FastTransformation)
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
        if len(self.list_of_files) < 1:
            QMessageBox.about(self, "Alert", "Please Choose Image File To Proceed Further !!")

        else:

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

                pixmap = QPixmap(path)
                w = self.topRightGroupBox.width()
                h = self.topRightGroupBox.height()
                dithersmaller_pixmap = pixmap.scaled(w-40, h-20, Qt.KeepAspectRatio, Qt.FastTransformation)
                self.segmented_img_label.setPixmap(dithersmaller_pixmap)


                self.superpixels(path)
            else:
                if len(self.img_to_process) < 2:
                    self.img_to_process = self.list_of_files[0]
                self.superpixels(self.img_to_process)

    def disable_resize_spin_box(self):
        if self.resizeButton.isChecked():
            self.resizefactor.setDisabled(False)
        else:
            self.resizefactor.setEnabled(False)

    def process_all_images(self):
        self.disable_vitals()
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value,
                                  color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number)

        self.another_process.batch_process_flag = True
        self.another_process.list_of_files = self.list_of_files
        self.another_process.start()
        self.another_process.add_post.connect(self.tread_done)

    @QtCore.pyqtSlot(str,str,str,str)
    def tread_done(self,processing_img_path, output_path, dither_path, time):

        self.enable_vitals()
        if output_path == 'e':
            print('ERROR')
            QMessageBox.about(self, "Alert",
                              "The Number of Colors defined in Dithering is more than the colors in Image Itself!!")
        else:
            # print(output_path, dither_path, time)
            self.time.setText('Time Taken to Process Image =' + time)
            self.current_output_path = output_path
            self.update_img_after_thread(processing_img_path,output_path, dither_path)

    def update_img_after_thread(self,processing_img_path,output_path, dithered_path):
        main_img = QPixmap(processing_img_path)
        pixmap_output = QPixmap(output_path)
        pixmap_reduced = QPixmap(dithered_path)
        w = self.topRightGroupBox.width()
        h = self.topRightGroupBox.height()
        mainsmaller_pixmap = main_img.scaled(w - 40, h - 20, Qt.KeepAspectRatio, Qt.FastTransformation)
        outputsmaller_pixmap = pixmap_output.scaled(w - 40, h - 20, Qt.KeepAspectRatio, Qt.FastTransformation)
        reducedsmaller_pixmap = pixmap_reduced.scaled(w - 40, h - 20, Qt.KeepAspectRatio, Qt.FastTransformation)

        self.img_label.setPixmap(mainsmaller_pixmap)
        self.segmented_img_label.setPixmap(outputsmaller_pixmap)
        self.final_image_label.setPixmap(reducedsmaller_pixmap)

    @QtCore.pyqtSlot(str, str)
    def kmeans_done(self,path, number):
        print('Kmeans finised==', path)
        w = self.topRightGroupBox.width()
        h = self.topRightGroupBox.height()
        pixmap_reducecolor = QPixmap(path)
        reduced_pixmap = pixmap_reducecolor.scaled(w - 40, h - 20, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.final_image_label.setPixmap(reduced_pixmap)
        self.reduced_number_of_color.setText('Number Of Colors in Image = ' + str(number))

    def reduce_color_final_img(self):
        self.clustering_thread.final_img_path = self.current_output_path
        self.clustering_thread.number_of_cluster = self.kmeans_color_slider.value()
        self.clustering_thread.start()
        self.clustering_thread.reduce_color_Signal.connect(self.kmeans_done)

    def createImageShower(self):
        self.ImageShower = QGroupBox("Loaded Images")
        self.ImageShowerList = QListWidget()
        self.ImageShowerList.setFocusPolicy(Qt.StrongFocus)
        # self.ImageShowerList.itemClicked.connect(self.itemClicked)
        self.ImageShowerList.itemSelectionChanged.connect(self.itemClicked)
        self.ImageShowerList.doubleClicked.connect(self.itemDoubleClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.ImageShowerList)
        #
        # layout.addStretch(1)
        self.ImageShower.setLayout(layout)

    # def receive_value_and_reduce(self):

    def info(self):
        print('After thread == ', self.kmeans_color_slider.value())

    def value_change(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(3000)
        # print('changed == ', self.kmeans_color_slider.value())

    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.info()
        self.reduce_color_final_img()

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Parameters")

        self.defaultPushButton = QPushButton("Choose Images")
        self.defaultPushButton.clicked.connect(self.open)

        self.ditherRadioButton = QRadioButton("Dither and then Process")
        self.ditherRadioButton.setChecked(True)

        self.resizeButton = QCheckBox("Resize and then Process")
        self.resizeButton.setChecked(False)
        self.resizeButton.clicked.connect(self.disable_resize_spin_box)

        self.ditherRadioButton.clicked.connect(self.enable_dither_color)
        self.ditherColor = QSpinBox()
        self.ditherColor.setMaximum(15)
        self.ditherColor.setMinimum(2)
        self.ditherColor.setValue(8)
        # self.ditherColor.setEnabled(False)

        self.resizefactor = QSpinBox()
        self.resizefactor.setMaximum(4)
        self.resizefactor.setMinimum(2)
        self.resizefactor.setValue(3)

        self.checkBox = QCheckBox("Connectivity Between Segments")
        self.checkBox.setEnabled(True)

        self.segments_number = QSpinBox()
        self.segments_number.setMaximum(10000)
        self.segments_number.setValue(100)

        self.compactness = QSpinBox()
        self.compactness.setValue(3)
        self.compactness.setMinimum(1)

        self.sigma = QSpinBox()
        self.sigma.setValue(2)

        self.quant_levels = QSpinBox()
        self.quant_levels.setMaximum(5)
        self.quant_levels.setMinimum(1)
        self.quant_levels.setValue(4)

        self.kmeans_color_slider = QSpinBox()
        self.kmeans_color_slider.setMaximum(30)
        self.kmeans_color_slider.setMinimum(4)
        self.kmeans_color_slider.setValue(8)
        self.kmeans_color_slider.valueChanged.connect(self.value_change)

        dither_color_label = QLabel("Number of Colors to be Used in Dithering")
        resize_img_label = QLabel("Upscaling Factor")

        segment_label = QLabel("Number of Segments")
        compactness_label = QLabel("Compactness:Larger Value,Makes Square Segments")
        sigma_label = QLabel("Lower the Number More Detailed Output")
        quant_label = QLabel("Color:Define Number of Pockets of Color to be Used")
        kmeans_label = QLabel("Reduce Final Image Color Number")

        self.time = QLabel()

        self.processPushButton = QPushButton("Batch Process All Images")
        self.processPushButton.clicked.connect(self.process_all_images)

        self.kmeans_push_button = QPushButton("Reduce Color")
        self.kmeans_push_button.clicked.connect(self.reduce_color_final_img)


        layout = QVBoxLayout()

        layout.addWidget(self.defaultPushButton)
        layout.addWidget(self.ditherRadioButton)
        layout.addWidget(dither_color_label)
        layout.addWidget(self.ditherColor)

        layout.addWidget(self.resizeButton)
        layout.addWidget(resize_img_label)
        layout.addWidget(self.resizefactor)
        layout.addWidget(self.checkBox)
        layout.addWidget(segment_label)
        layout.addWidget(self.segments_number)
        layout.addWidget(compactness_label)
        layout.addWidget(self.compactness)
        layout.addWidget(sigma_label)
        layout.addWidget(self.sigma)
        layout.addWidget(quant_label)
        layout.addWidget(self.quant_levels)
        layout.addWidget(kmeans_label)
        layout.addWidget(self.kmeans_color_slider)
        # layout.addWidget(self.kmeans_push_button)

        layout.addWidget(self.processPushButton)
        layout.addWidget(self.time)


        # layout.addStretch(0)
        self.topLeftGroupBox.setLayout(layout)

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Image To Process")

        self.img_label = QLabel(self)

        layout = QVBoxLayout()

        layout.addWidget(self.img_label)
        layout.maximumSize()
        layout.addStretch(1)
        # print(self.img_label.height(), self.img_label.width())

        self.topRightGroupBox.setLayout(layout)

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QGroupBox("Final Image")
        self.segmented_img_label = QLabel(self)
        # self.segmented_img_label.addStreach(1)

        layout = QVBoxLayout()
        layout.addWidget(self.segmented_img_label)

        layout.addStretch(1)

        self.bottomLeftTabWidget.setLayout(layout)

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Reduced Color")
        self.final_image_label = QLabel(self)
        self.reduced_number_of_color = QLabel(self)



        layout = QVBoxLayout()
        layout.addWidget(self.final_image_label)
        layout.addWidget(self.reduced_number_of_color)

        layout.addStretch(1)
        self.bottomRightGroupBox.setLayout(layout)


class WorkerThread(QThread):
    add_post = pyqtSignal(str,str,str,str)

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.current_path = ''
        self.dither_flag = False
        self.dither_color = 4
        self.number_of_segments = 100
        self.sigma_value = 3
        self.compactness_value = 3
        self.color_pocket_number = 3
        self.connectivity = False
        self.batch_process_flag = False
        self.list_of_files = []
        self.resize_flag = True
        self.resize_factor = 3
        self.reduce_color_number = 8

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self.batch_process_flag == True:
                for each in self.list_of_files:
                    output_path, dither_path, time = dither_algorithm.main(each, self.dither_flag,
                                                                           self.dither_color, self.number_of_segments,
                                                                           self.connectivity, self.compactness_value,
                                                                           self.sigma_value, self.color_pocket_number, self.resize_flag,
                                                                           self.resize_factor, self.reduce_color_number)
                    self.add_post.emit(each,output_path, dither_path, time)
            else:
                output_path, dither_path, time = dither_algorithm.main(self.current_path, self.dither_flag, self.dither_color,
                                                                       self.number_of_segments,self.connectivity, self.compactness_value,
                                                                       self.sigma_value, self.color_pocket_number,
                                                                       self.resize_flag, self.resize_factor, self.reduce_color_number)
                self.add_post.emit(self.current_path,output_path, dither_path, time)
        except:
            self.add_post.emit('e', 'e', 'e', 'e')

class Kmeans(QThread):
    reduce_color_Signal = pyqtSignal(str,str)

    def __init__(self, parent=None):
        super(Kmeans, self).__init__(parent)
        self.final_img_path = ''
        self.number_of_cluster = 8

    @QtCore.pyqtSlot()
    def run(self):
        try:
            path, cluster_used = clustering.get_dominant_color(self.final_img_path, self.number_of_cluster)
            self.reduce_color_Signal.emit(path, str(cluster_used))
        except:
            self.reduce_color_Signal.emit('e', 'e')


class SleepThread(QThread):
    sleep_Signal = pyqtSignal()

    def __init__(self, parent=None):
        super(SleepThread, self).__init__(parent)

    @QtCore.pyqtSlot()
    def run(self):
        time.sleep(5)
        self.sleep_Signal.emit()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.showMaximized()
    gallery.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint)
    gallery.show()
    sys.exit(app.exec_()) 
