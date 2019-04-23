from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PIL import Image
import color_percent
import os
import time
import dither_algorithm
from dither_from_dll import FunctionsFromDLL
import traceback
import glob
import shutil
from BlurAlgorithms import BlurFilters
import numpy as np
import cv2
from mixply import MixPLy

class WidgetGallery(QDialog):
    slic_thread_signal = pyqtSignal(list)
    reduce_color_signal = pyqtSignal(list)
    sleep_Signal = pyqtSignal()
    mixply_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.another_process = SlicWorkerThread()
        self.clustering_thread = Kmeans()
        self.delay_thread = SleepThread()
        self.mixply_thread = MixplyThread()
        self.timer_id = -1

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        self.create_image_listview()
        self.create_parameter_groupbox()
        self.create_main_img_groupbox()
        self.create_final_img_groupbox()
        self.create_reduced_color_groupbox()
        self.thread_signal_connect()


        self.ImageShower.setFixedWidth(100)
        self.parameter_group_box.setFixedWidth(350)
        mainLayout = QGridLayout()
        second_grid_layout = self.second_grid_layout()
        mainLayout.addWidget(self.parameter_group_box, 1, 0)
        mainLayout.addWidget(self.ImageShower, 1, 1)
        mainLayout.addWidget(second_grid_layout, 1, 2)
        # mainLayout.addWidget(self.topRightGroupBox, 1, 2)
        # mainLayout.addWidget(self.bottomLeftTabWidget, 1, 3)
        # mainLayout.addWidget(self.bottomRightGroupBox, 1, 4)
        # mainLayout.addWidget(self.mix_ply_img_lbl, 1, 5)
        self.setLayout(mainLayout)
        self.list_of_files = []
        self.img_to_process = ''
        self.current_output_path = ''
        self.reduced_img_path = ''
        self.current_img = None

        self.connect_signals()
        self.setWindowTitle("Dithering")
        self.changeStyle('Fusion')

    def create_parameter_groupbox(self):
        self.parameter_group_box = QGroupBox("Parameters")
        bluring_groupbox = self.create_bluring_groupbox()
        self.defaultPushButton = QPushButton("Choose Folder")
        self.defaultPushButton.clicked.connect(self.open_filedialog)

        self.grayscalebutton = QCheckBox("Process in GrayScale")
        self.grayscalebutton.setChecked(False)

        self.ditherRadioButton = QRadioButton("Dither and then Process")
        self.ditherRadioButton.setChecked(False)

        self.resizeButton = QCheckBox("Upscale and then Process")
        self.resizeButton.setChecked(False)
        self.resizeButton.clicked.connect(self.disable_resize_spin_box)

        self.ditherRadioButton.clicked.connect(self.enable_dither_color)
        self.ditherColor = QSpinBox()
        self.ditherColor.setMaximum(20)
        self.ditherColor.setMinimum(2)
        self.ditherColor.setValue(6)
        # self.ditherColor.setEnabled(False)

        self.resizefactor = QSpinBox()
        self.resizefactor.setMaximum(4)
        self.resizefactor.setMinimum(2)
        self.resizefactor.setValue(3)
        self.resizefactor.setEnabled(False)

        self.checkBox = QCheckBox("Connectivity Between Segments")
        self.checkBox.setEnabled(True)

        self.segments_number = QSpinBox()
        self.segments_number.setMaximum(10000)
        self.segments_number.setValue(250)

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
        self.kmeans_color_slider.setMaximum(64)
        self.kmeans_color_slider.setMinimum(2)
        self.kmeans_color_slider.setValue(36)
        self.kmeans_color_slider.valueChanged.connect(self.reduce_color_final_img)

        delete_folder_push_button = QPushButton('Delete All Output Folders')
        delete_folder_push_button.clicked.connect(self.delete_folders)

        dither_color_label = QLabel("Number of Colors to be Used in Dithering")
        resize_img_label = QLabel("Upscaling Factor")

        segment_label = QLabel("Number of Segments")
        compactness_label = QLabel("Compactness:Larger Value,Makes Square Segments")
        sigma_label = QLabel("Lower the Number More Detailed Output")
        # quant_label = QLabel("Color:Define Number of Pockets of Color to be Used")
        kmeans_label = QLabel("Reduce Final Image Color Number")

        self.time = QLabel()

        self.processPushButton = QPushButton("Batch Process All Images")
        self.processPushButton.clicked.connect(self.batch_process_images)

        self.kmeans_push_button = QPushButton("Reduce Color")
        self.kmeans_push_button.clicked.connect(self.reduce_color_final_img)

        self.resize_check_box = QCheckBox("Resize Image")
        self.resize_check_box.clicked.connect(self.disable_dimension)
        self.resize_check_box.setChecked(True)
        self.width_input = QLineEdit()
        self.width_input.setText('900')
        # self.LineEdit.setValidator(self.width_input)
        self.height_input = QLineEdit()
        self.height_input.setText('1200')

        width_label = QLabel("W:")
        height_label = QLabel("H:")

        mixply_groupbox = self.create_mixply_group_box()

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.resize_check_box)
        hlayout.addWidget(width_label)
        hlayout.addWidget(self.width_input)
        hlayout.addWidget(height_label)
        hlayout.addWidget(self.height_input)


        layout = QVBoxLayout()

        layout.addWidget(self.defaultPushButton)
        layout.addWidget(self.grayscalebutton)
        layout.addWidget(self.ditherRadioButton)
        layout.addWidget(dither_color_label)
        layout.addWidget(self.ditherColor)
        layout.addWidget(bluring_groupbox)
        layout.addLayout(hlayout)
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
        # layout.addWidget(quant_label)
        # layout.addWidget(self.quant_levels)
        layout.addWidget(kmeans_label)
        layout.addWidget(self.kmeans_color_slider)
        # layout.addWidget(self.kmeans_push_button)
        layout.addWidget(mixply_groupbox)
        layout.addWidget(self.processPushButton)
        layout.addWidget(self.time)
        layout.addWidget(delete_folder_push_button)


        # layout.addStretch(0)
        self.parameter_group_box.setLayout(layout)

    def create_bluring_groupbox(self):
        group_box = QGroupBox('PreProcessing Tools')
        grid_layout = QGridLayout()

        saturation_label = QLabel('Sat. V')
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setValue(25)

        motion_blur_label = QLabel('Motion. B')
        self.motion_blur_slider = QSlider(Qt.Horizontal)
        self.motion_blur_slider.setMinimum(2)

        sharpen_label = QLabel('Sharpen')
        self.sharpen_slider = QSpinBox()
        self.sharpen_slider.setMinimum(0)
        self.sharpen_slider.setMaximum(10)
        # self.sharpen_slider.setTickInterval(1)
        # self.sharpen_slider.setSingleStep(1)
        self.revert_blurred_img_btn = QPushButton('Revert to Original Image')
        self.process_blurred_img = QPushButton('Process Image')


        grid_layout.addWidget(saturation_label, 0, 0)
        grid_layout.addWidget(self.saturation_slider, 0, 1)
        grid_layout.addWidget(motion_blur_label, 1, 0)
        grid_layout.addWidget(self.motion_blur_slider, 1, 1)
        grid_layout.addWidget(sharpen_label, 2, 0)
        grid_layout.addWidget(self.sharpen_slider, 2, 1)
        grid_layout.addWidget(self.revert_blurred_img_btn, 3, 0)
        grid_layout.addWidget(self.process_blurred_img, 3, 1)

        group_box.setLayout(grid_layout)
        return group_box

    def create_image_listview(self):
        self.ImageShower = QGroupBox("Loaded Images")
        self.ImageShowerList = QListWidget()
        self.ImageShowerList.setFocusPolicy(Qt.StrongFocus)
        # self.ImageShowerList.itemClicked.connect(self.itemClicked)
        self.ImageShowerList.currentItemChanged.connect(self.itemClicked)
        self.ImageShowerList.doubleClicked.connect(self.itemDoubleClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.ImageShowerList)
        #
        # layout.addStretch(1)
        self.ImageShower.setLayout(layout)

    def create_mixply_group_box(self):
        mixply_groupbox = QGroupBox('Mixply Section')
        grid_layout = QGridLayout()

        self.color_space_selection = QComboBox()
        self.color_space_selection.addItem('HSV')
        self.color_space_selection.addItem('RGB')
        self.color_space_selection.addItem('LAB')
        self.mixply_intensity_stepsize_spinbox = QSpinBox()
        self.mixply_intensity_stepsize_spinbox.setMinimum(1)
        mixply_intensity_stepsize_lbl = QLabel('StepSize for Intensity')
        mixply_intensit_label = QLabel("Intensity of Mixply")
        self.mixply_intensity_slider = QSlider(Qt.Horizontal)
        self.mixply_intensity_slider.setMinimum(0)
        self.mixply_intensity_slider.setMaximum(100)

        self.mixply_intensity_slider.setTickInterval(20)
        # self.mixply_intensity_slider.setMouseTracking(True)
        self.mixply_intensity_slider.setTickPosition(QSlider.TicksBothSides)

        mixply_color_label = QLabel("Mixply Colors")
        self.mixply_color_spinbox = QSpinBox()
        self.mixply_color_spinbox.setMaximum(36)

        grid_layout.addWidget(self.color_space_selection, 0, 0)
        grid_layout.addWidget(mixply_intensity_stepsize_lbl, 1, 0)
        grid_layout.addWidget(self.mixply_intensity_stepsize_spinbox, 1, 1)
        grid_layout.addWidget(mixply_intensit_label, 2,0)
        grid_layout.addWidget(self.mixply_intensity_slider, 2,1)
        grid_layout.addWidget(mixply_color_label, 3, 0)
        grid_layout.addWidget(self.mixply_color_spinbox, 3, 1)

        mixply_groupbox.setLayout(grid_layout)
        return mixply_groupbox

    def create_main_img_groupbox(self):
        self.main_img_group_box = QGroupBox("Image To Process")

        self.img_label = QLabel(self)

        layout = QVBoxLayout()

        layout.addWidget(self.img_label)
        # layout.maximumSize()
        layout.addStretch(1)
        # print(self.img_label.height(), self.img_label.width())

        self.main_img_group_box.setLayout(layout)

    def create_final_img_groupbox(self):
        self.final_img_group_box = QGroupBox("Final Image")
        self.final_img_lbl = QLabel(self)
        # self.segmented_img_label.addStreach(1)

        layout = QVBoxLayout()
        layout.addWidget(self.final_img_lbl)

        layout.addStretch(1)

        self.final_img_group_box.setLayout(layout)

    def create_reduced_color_groupbox(self):
        self.reduced_color_group_box = QGroupBox("Reduced Color")
        self.reduced_color_img_lbl = QLabel(self)
        self.reduced_number_of_color = QLabel(self)



        layout = QVBoxLayout()
        layout.addWidget(self.reduced_color_img_lbl)
        layout.addWidget(self.reduced_number_of_color)

        layout.addStretch(1)
        self.reduced_color_group_box.setLayout(layout)

    def create_mixply_img_groupbox(self):
        mix_ply_img_lbl = QGroupBox("Image Using Mixply")

        self.mixply_img_lbl = QLabel(self)
        self.mixply_color_number = QLabel()

        layout = QVBoxLayout()

        layout.addWidget(self.mixply_img_lbl)
        layout.addWidget(self.mixply_color_number)
        # layout.maximumSize()
        layout.addStretch(1)
        # print(self.img_label.height(), self.img_label.width())

        mix_ply_img_lbl.setLayout(layout)
        return mix_ply_img_lbl

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))

    def second_grid_layout(self):
        second_vertical_box = QGroupBox('Images')
        second_grid_layout = QGridLayout()
        mix_ply_img_lbl = self.create_mixply_img_groupbox()
        second_grid_layout.addWidget(self.main_img_group_box, 0, 0)
        second_grid_layout.addWidget(self.final_img_group_box, 0, 1)
        second_grid_layout.addWidget(self.reduced_color_group_box, 1, 0)
        second_grid_layout.addWidget(mix_ply_img_lbl, 1, 1)
        second_vertical_box.setLayout(second_grid_layout)
        return second_vertical_box

    def connect_signals(self):
        self.saturation_slider.sliderReleased.connect(self.change_saturation)
        self.motion_blur_slider.sliderReleased.connect(self.motion_blur)
        self.sharpen_slider.valueChanged.connect(self.sharpen_image)
        self.mixply_intensity_slider.sliderReleased.connect(self.create_mixply_image)
        self.mixply_intensity_stepsize_spinbox.valueChanged.connect(self.create_mixply_image)
        self.color_space_selection.currentTextChanged.connect(self.create_mixply_image)
        self.mixply_color_spinbox.valueChanged.connect(self.value_change)
        self.revert_blurred_img_btn.clicked.connect(self.revert_blurred_image)
        self.process_blurred_img.clicked.connect(self.process_after_blur)
        # self.mixply_color_spinbox..connect(self.create_mixply_image)

    def revert_blurred_image(self):
        self.restore_blur_parametrs()
        self.current_img = np.asarray(Image.open(self.img_to_process).convert('RGB'), dtype='uint8')
        # self.current_img = cv2.imread(self.img_to_process)
        self.render_img_from_array(self.current_img, self.img_label)

    def restore_blur_parametrs(self):
        self.sharpen_slider.setValue(0)
        self.motion_blur_slider.setValue(0)
        self.saturation_slider.setValue(25)

    def itemClicked(self, item):
        self.another_process.batch_process_flag = False
        try:
            location = int(item.text())
        except AttributeError:
            print('Atrribute Error Handled')
            return
        self.disable_vitals()
        # print(self.list_of_files)
        new_img_path = self.list_of_files[location]
        self.another_process.current_path = new_img_path
        print(new_img_path, location)
        # self.current_img = cv2.imread(new_img_path)
        self.current_img = np.asarray(Image.open(new_img_path).convert('RGB'), dtype='uint8')
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,change_dim_flag,dim, grayscale_flag = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,change_dim_flag,dim,grayscale_flag)

        self.another_process.start()
        self.img_to_process = new_img_path
        # self.current_img = Image.open_filedialog(new_img_path)
        self.display_original_image(new_img_path)

    def itemselectionChanged(self):
        items = self.ImageShowerList.selectedItems()
        # print(str(items[0].text()))

    def thread_signal_connect(self):
        self.clustering_thread.reduce_color_signal.connect(self.kmeans_thread_complete)
        self.another_process.slic_thread_signal.connect(self.slic_thread_completed)
        self.mixply_thread.mixply_thread_signal.connect(self.mixply_thread_complete)
        # self.another_process.add_post.connect(self.slic_thread_completed)
        # self.another_process.add_post.connect(self.slic_thread_completed)

    def change_saturation(self):
        blurFilters = BlurFilters()
        slider_value = self.saturation_slider.value()
        saturated_img = blurFilters.increase_saturation(self.current_img, 4*(slider_value/100))
        self.current_img = saturated_img
        # self.display_main_image_from_array(saturated_img)
        self.render_img_from_array(self.current_img, self.img_label)

    def display_main_image_from_array(self, img):
        height = self.main_img_group_box.height()
        final_img = QPixmap(self.numpy_to_pixmap(img))
        final_img = final_img.scaledToHeight(height - 10, mode=Qt.FastTransformation)
        self.img_label.setPixmap(final_img)

    def create_mixply_image(self):
        print((int(self.mixply_intensity_slider.value())//20))
        self.mixply_thread.image_path = self.reduced_img_path
        self.mixply_thread.level_of_mixply = 5-int(self.mixply_intensity_slider.value())//20
        self.mixply_thread.mixply_colors = int(self.mixply_color_spinbox.value())
        self.mixply_thread.intensity_step_size  = int(self.mixply_intensity_stepsize_spinbox.value())
        self.mixply_thread.color_selection_space = str(self.color_space_selection.currentText())
        self.mixply_thread.start()

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def motion_blur(self):
        blurFilters = BlurFilters()
        img = blurFilters.motion_blur(self.current_img, self.motion_blur_slider.value())
        self.current_img = img
        self.display_main_image_from_array(self.current_img)

    def sharpen_image(self):
        blurFilter = BlurFilters()
        img = blurFilter.sharpening_filter(self.current_img, self.sharpen_slider.value())
        self.current_img = img
        # self.display_main_image_from_array(self.current_img)
        self.render_img_from_array(self.current_img, self.img_label)

    def itemDoubleClicked(self, column_no):

        self.another_process.batch_process_flag = False
        self.disable_vitals()
        # location = int(item.text())
        new_img_path = self.img_to_process
        # self.current_img = np.asarray(Image.open_filedialog())
        self.another_process.current_path = new_img_path
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag)

        self.another_process.start()
        self.img_to_process = new_img_path
        self.display_original_image(new_img_path)

    def process_after_blur(self):
        self.another_process.batch_process_flag = False
        self.disable_vitals()
        self.another_process.current_path = self.img_to_process
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag)
        self.another_process.start()

    def list_files_inside_folder(self, path_to_folder):
        # path_in_glob_format = path_to_folder + '/*' + '.png'
        list_of_files = []
        files = []
        for ext in ('*.gif', '*.png', '*.jpg', '*.bmp', '*JPG', '*JPEG'):
            files.extend(glob.glob(os.path.join(path_to_folder, ext)))

        for each in files:
            each = each.replace('\\', '/')
            list_of_files.append(each)

        return list_of_files

    def open_filedialog(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        # files, _ = QFileDialog.getOpenFileNames(self, "Choose Image File", "",
        #                                         "All Files (*.*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        folder_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        files = self.list_files_inside_folder(folder_name)
        if files:
            self.list_of_files.clear()
            self.list_of_files = files
            self.img_to_process = files[0]
            self.current_img = np.asarray(Image.open(self.img_to_process).convert('RGB'))
            # self.display_original_image(files[0])
            self.render_img_from_array(self.current_img, self.img_label)
            self.ImageShowerList.clear()
            for i,each in enumerate(files):
                itm = QListWidgetItem(str(i))
                itm.setIcon(QIcon(each))
                self.ImageShowerList.blockSignals(True)
                self.ImageShowerList.addItem(itm)
                self.ImageShowerList.blockSignals(False)

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

        if self.resize_check_box.isChecked():
            change_dim_flag = True
            w = self.width_input.text()
            h = self.height_input.text()
            try:
                w = int(w)
                h = int(h)
                dim = tuple((w,h))
            except Exception:
                QMessageBox.about(self, 'Error', 'Input can only be a number')
                pass
        else:
            change_dim_flag = False
            w = 0
            h = 0
            dim = tuple((w,h))
        if self.grayscalebutton.isChecked():
            grayscale_flag = True
        else:
            grayscale_flag = False

        return dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor, reduce_color_number,change_dim_flag,dim, grayscale_flag

    def set_status_to_thread(self, dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor, reduce_color_number, change_dim_flag, dim, grayscale_flag):
        self.another_process.img_to_process = self.current_img
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
        self.another_process.change_dim_flag = change_dim_flag
        self.another_process.dim = dim
        self.another_process.grayscale_flag = grayscale_flag
        self.restore_blur_parametrs()

    def disable_vitals(self):
        # self.processPushButton.setDisabled(True)
        # self.defaultPushButton.setDisabled(True)
        self.parameter_group_box.setDisabled(True)

    def enable_vitals(self):
        self.parameter_group_box.setEnabled(True)

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    def display_original_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.img_label.adjustSize()
        w = self.main_img_group_box.width()
        h = self.main_img_group_box.height()
        # print(w,h)
        smaller_pixmap = pixmap.scaled(w-40,h-20,Qt.KeepAspectRatio, Qt.FastTransformation)
        self.img_label.setPixmap(smaller_pixmap)

    def convert_to_QImage(self, cvImg):
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def enable_dither_color(self):
        if self.ditherRadioButton.isChecked():
            self.ditherColor.setDisabled(False)
        else:
            self.ditherColor.setEnabled(False)

    def disable_resize_spin_box(self):
        if self.resizeButton.isChecked():
            self.resizefactor.setDisabled(False)
        else:
            self.resizefactor.setEnabled(False)

    def batch_process_images(self):
        self.disable_vitals()
        dither_flag, dither_color, number_of_segments, sigma_value, compactness_value, color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag = self.get_status()
        self.set_status_to_thread(dither_flag, dither_color, number_of_segments, sigma_value, compactness_value,
                                  color_pocket_number, connectivity, resize_flag, resize_factor,reduce_color_number,dim_change_flag,dim, grayscale_flag)

        self.another_process.batch_process_flag = True
        self.another_process.list_of_files = self.list_of_files
        self.another_process.start()

    def convert_from_pil_to_numpy(self, img):
        return np.asarray(img)

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def number_of_color_in_img(self, filename):
        img = Image.open(filename).convert('RGB')
        return len(img.getcolors())

    @QtCore.pyqtSlot(list)
    def slic_thread_completed(self, slic_thread_list):
        print(len(slic_thread_list), slic_thread_list)
        processing_img_path, output_path, dither_path, time, k_number, mixply_img_path = slic_thread_list[0], slic_thread_list[1], slic_thread_list[2], slic_thread_list[3], slic_thread_list[4], slic_thread_list[5]

        self.enable_vitals()
        if output_path == 'e':
            print('ERROR')
            QMessageBox.about(self, "Alert",
                              "Something Went Wrong with Image="+str(processing_img_path))
            return
        else:
            self.time.setText('Time Taken to Process Image =' + time)
            self.current_output_path = output_path
            self.update_img_after_slic_thread(processing_img_path, output_path, dither_path, k_number, mixply_img_path)

    @QtCore.pyqtSlot(list)
    def kmeans_thread_complete(self, kmeans_signal_list):
        self.clustering_thread.terminate()
        # print('Kmeans finised==', path)
        w = self.main_img_group_box.width()
        h = self.main_img_group_box.height()
        img, path = kmeans_signal_list[0], kmeans_signal_list[1]
        self.reduced_img_path = path
        img = self.convert_from_pil_to_numpy(img)
        img = self.numpy_to_pixmap(img)
        pixmap_reducecolor = QPixmap(img)
        reduced_pixmap = pixmap_reducecolor.scaled(w - 40, h - 20, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.reduced_color_img_lbl.setPixmap(reduced_pixmap)
        self.reduced_number_of_color.setText('Number Of Colors in Image = ' + str(self.kmeans_color_slider.value()))

    @QtCore.pyqtSlot(list)
    def mixply_thread_complete(self, mixply_outputlist):
        mix_ply_img = mixply_outputlist[0]
        self.render_img_from_array(mix_ply_img, self.mixply_img_lbl)
        self.render_color_number_from_img(mix_ply_img, self.mixply_color_number)
        # print('finished')

    def update_img_after_slic_thread(self, processing_img_path, output_path, reduced_path, k_number, mixply_img_path):
        self.reduced_img_path = reduced_path
        main_img = QPixmap(processing_img_path)
        pixmap_output = QPixmap(output_path)
        pixmap_reduced = QPixmap(reduced_path)
        pixmap_mixply = QPixmap(mixply_img_path)
        screen = app.primaryScreen()
        h = screen.size().height()
        w = screen.size().width()
        mainsmaller_pixmap = main_img.scaledToHeight(h//2.75, Qt.FastTransformation)
        outputsmaller_pixmap = pixmap_output.scaledToHeight(h//2.75, Qt.FastTransformation)
        reducedsmaller_pixmap = pixmap_reduced.scaledToHeight(h//2.75, Qt.FastTransformation)
        mixply_img_pixmap = pixmap_mixply.scaledToHeight(h//2.75, Qt.FastTransformation)


        self.img_label.setPixmap(mainsmaller_pixmap)
        self.final_img_lbl.setPixmap(outputsmaller_pixmap)
        self.reduced_color_img_lbl.setPixmap(reducedsmaller_pixmap)
        self.reduced_number_of_color.setText('Number Of Colors in Image = ' + str(self.number_of_color_in_img(reduced_path)))
        self.mixply_img_lbl.setPixmap(mixply_img_pixmap)
        self.mixply_color_number.setText('Number Of Colors in Image = ' + str(self.number_of_color_in_img(mixply_img_path)))

    def render_img_from_array(self, img, place_holder):
        pixmap = QPixmap(self.numpy_to_pixmap(img))
        h = self.main_img_group_box.height()
        w = self.main_img_group_box.width()
        mainsmaller_pixmap = pixmap.scaled(w-5, h-10,Qt.KeepAspectRatio, Qt.FastTransformation)
        place_holder.setPixmap(mainsmaller_pixmap)

    def render_color_number_from_img(self, img, place_holder):
        img = Image.fromarray(img.astype('uint8'))
        numbers = len(img.getcolors())
        place_holder.setText('Number of Colors ='+str(numbers))

    def reduce_color_final_img(self):
        self.clustering_thread.final_img_path = self.current_output_path
        self.clustering_thread.number_of_cluster = self.kmeans_color_slider.value()
        self.clustering_thread.start()

    def info(self):
        print('Starting thread to Reduce Color == ', self.kmeans_color_slider.value())

    def value_change(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(500)
        # print('changed == ', self.kmeans_color_slider.value())

    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        # self.info()
        self.create_mixply_image()

    def disable_dimension(self):
        if self.resize_check_box.isChecked():
            self.width_input.setDisabled(False)
            self.height_input.setDisabled(False)
        else:
            self.width_input.setEnabled(False)
            self.height_input.setEnabled(False)

    def delete_folders(self):
        output_folders = ['A_MIXEDPLY_OUTPUT', 'A_DITHERED', 'A_Grayscaled', 'A_OUTPUT', 'A_REDUCE_COLOR', 'A_SEGMENTED', 'A_STICHED_OUTPUT','A_UPSCALED_INPUT']
        for each in output_folders:
            shutil.rmtree(each, ignore_errors=True)


class SlicWorkerThread(QThread):
    slic_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(SlicWorkerThread, self).__init__(parent)
        self.current_path = ''
        self.img_to_process = None
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
        self.change_dim_flag = False
        self.dim = (0,0)
        self.grayscale_flag = False

    @QtCore.pyqtSlot()
    def run(self):
        if self.batch_process_flag == True:
            for each in self.list_of_files:
                try:
                    signal_list = []
                    # self.img_to_process = cv2.imread(each)
                    self.img_to_process = np.asarray(Image.open(each).convert('RGB'), dtype='uint8')
                    output_path, dither_path, time, k_number, mixply_img_path = dither_algorithm.main(each, self.img_to_process, self.dither_flag,
                                                                           self.dither_color, self.number_of_segments,
                                                                           self.connectivity, self.compactness_value,
                                                                           self.sigma_value, self.color_pocket_number, self.resize_flag,
                                                                           self.resize_factor, self.reduce_color_number,self.change_dim_flag,self.dim,self.grayscale_flag)
                    # self.add_post.emit(each,output_path, dither_path, time, str(k_number))
                    signal_list.append(each)
                    signal_list.append(output_path)
                    signal_list.append(dither_path)
                    signal_list.append(time)
                    signal_list.append(str(k_number))
                    signal_list.append(mixply_img_path)

                    self.slic_thread_signal.emit(signal_list)
                except Exception as e:
                    signal_list = []
                    traceback.print_exc()
                    print('error and skipped image ==', each)
                    actual_name = os.path.splitext(each)[0]
                    actual_name = actual_name.split('/')
                    actual_name = actual_name[len(actual_name) - 1]
                    with open('ErrorLog.txt', 'a') as file:
                        file.writelines(actual_name)
                        file.writelines(str(e))
                        file.writelines(traceback.format_exc())
                    if self.list_of_files[-1] == each:
                        signal_list.append('e')
                        signal_list.append('e')
                        signal_list.append('e')
                        signal_list.append('e')
                        signal_list.append('e')
                        signal_list.append('e')

                        self.slic_thread_signal.emit(signal_list)


                    # self.add_post.emit(str(each), 'e', 'e', 'e')
        else:
            # print(self.change_dim_flag, self.dim)
            try:
                signal_list = []
                output_path, dither_path, time, k_number, mixply_img_path = dither_algorithm.main(self.current_path, self.img_to_process, self.dither_flag, self.dither_color,
                                                                       self.number_of_segments,self.connectivity, self.compactness_value,
                                                                       self.sigma_value, self.color_pocket_number,
                                                                       self.resize_flag, self.resize_factor, self.reduce_color_number,self.change_dim_flag, self.dim, self.grayscale_flag)
                signal_list.append(self.current_path)
                signal_list.append(output_path)
                signal_list.append(dither_path)
                signal_list.append(time)
                signal_list.append(str(k_number))
                signal_list.append(mixply_img_path)
                self.slic_thread_signal.emit(signal_list)
            except Exception as e:
                signal_list = []
                traceback.print_exc()
                actual_name = os.path.splitext(self.current_path)[0]
                actual_name = actual_name.split('/')
                actual_name = actual_name[len(actual_name) - 1]
                with open('ErrorLog.txt', 'a') as file:
                    file.writelines(actual_name)
                    file.writelines(str(e))
                    file.writelines(traceback.format_exc())
                signal_list.append('e')
                signal_list.append('e')
                signal_list.append('e')
                signal_list.append('e')
                signal_list.append('e')
                signal_list.append('e')

                self.slic_thread_signal.emit(signal_list)


class MixplyThread(QThread):
    mixply_thread_signal = pyqtSignal(list)
    def __init__(self, parent=None):
        super(MixplyThread, self).__init__(parent)
        self.image_path = ''
        self.img_to_process = None
        self.level_of_mixply = 0
        self.mixply_colors = 5
        self.color_selection_space = ''
        self.intensity_step_size = 1

    @QtCore.pyqtSlot()
    def run(self):
        img_list = []
        # print('Level = ', self.level_of_mixply, 'Colors =', self.mixply_colors)
        mixPly = MixPLy()
        # main_colors, remaining_colors = mixPly.get_colors_from_img(self.image_path, self.level_of_mixply, self.color_selection_space, self.intensity_step_size)
        # mix_ply_list = mixPly.create_combination_and_distance_table(main_colors, remaining_colors)
        # replacing_dict = mixPly.sort_and_pick(mix_ply_list, number_to_replace=self.mixply_colors)
        colors = Image.open(self.image_path).convert('HSV').getcolors()
        c = [x[1] for x in colors]
        replacing_dict = mixPly.one_color_at_atime(c, self.mixply_colors)
        mix_ply_img = mixPly.create_mixply_image(self.image_path, replacing_dict, self.color_selection_space)
        print(replacing_dict)
        img_list.append(mix_ply_img)
        self.mixply_thread_signal.emit(img_list)


class Kmeans(QThread):
    reduce_color_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(Kmeans, self).__init__(parent)
        self.final_img_path = ''
        self.number_of_cluster = 10

    @QtCore.pyqtSlot()
    def run(self):
        try:
            kmeans_signal_list = []
            dir_path = os.getcwd()
            dir_path = dir_path.replace('\\', '/')
            clustering = FunctionsFromDLL()
            path = clustering.reduce_color(self.final_img_path, self.number_of_cluster, dir_path)
            reduced_img = Image.open(path).convert('RGB')
            kmeans_signal_list.append(reduced_img)
            kmeans_signal_list.append(path)
            self.reduce_color_signal.emit(kmeans_signal_list)
        except:
            # traceback.print_exc()
            kmeans_signal_list = []
            kmeans_signal_list.append('e')
            kmeans_signal_list.append('e')
            self.reduce_color_signal.emit(kmeans_signal_list)


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
    screen = app.primaryScreen()
    h = screen.size().height()
    w = screen.size().width()

    gallery = WidgetGallery()
    # gallery.showMaximized()
    gallery.setFixedSize(int(w-w*0.025), int(h-h*0.1))
    gallery.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint)
    gallery.show()
    sys.exit(app.exec_()) 
