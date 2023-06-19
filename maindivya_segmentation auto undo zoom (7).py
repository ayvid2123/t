from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import inspect
from glob import glob
from math import sin as msin
import ast 
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from shapely.geometry import Polygon, MultiPolygon
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

from PIL.ImageQt import ImageQt

from PyQt5.QtCore import QBuffer
from PyQt5.QtWidgets import (
    QMainWindow,
    QAction,
    QApplication,
    QComboBox,
    QDesktopWidget,
    QStatusBar,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,  # displaying text or image
    QListWidget,
    QFileDialog,
    QFrame,
    QLineEdit,
    QListWidgetItem,
    QDockWidget,
    QMessageBox,
    QSlider,
    QToolBar,
    QToolButton,
    QMenu,
    QSpacerItem,
    QSizePolicy,
    QStyle,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsPolygonItem,
    QGraphicsPathItem,
    QGraphicsItem
)

import numpy as np
import pandas as pd
import cv2
import sys
import os
import time
from os.path import isfile, join
from PIL import Image as im
from PIL.ImageQt import ImageQt
import subprocess
import time
import json
import copy 
import list_color

path = os.path.dirname(os.path.realpath(r"..\dashcamcleaner\main.py"))
path2 = os.path.dirname(os.path.realpath(r"..\dashcamcleaner\src\blurrer.py"))
sys.path.append(path)
sys.path.append(path2)
# from main_2 import MainWindow
from blurrer import VideoBlurrer

curr_frame, curr_frame_dir = None, None
res_width, res_height, scale_width, scale_ht = 0, 0, 0, 0
delta_x, delta_y, offset_x, offset_y = 0, 0, 0, 0
img_x, img_y, img_w, img_h = 0, 0, 0, 0

handle = 4
selected_rectangle, selected_cube = [], []
edit_highlight = []
# highlights for different shapes
rectangle_highlight, polygon_highlight, polyline_highlight, cube_highlight = -1, -1, -1, -1

sys.setrecursionlimit(10 ** 9)

current_session = 0
categories = ['Car', 'Pedestrian', 'Truck', 'License Plate', 'Face', 'Bike', 'Lane']
transparent = 127
categories_color = list_color.get_colors(transparent)

def setup_toolbar(qt_obj):
    tools = {}

    names = [
        'Open Image',
        'Open Video',
        'Load Annotation',
        'Auto Mode',
        'Manual Mode',
        'Rectangle',
        'Polyline',
        'Polygon',
        'Cube',
        'Segmentation Mode',
        'Blur Filter',
        'Export Video',
        'Save Annotation',
        'Delete',
        'Undo',
        'Redo',
    ]

    icons = [
        'upload_image.png',
        'upload_video.png',
        'load_annotation.png',
        'auto.png',
        'manual.png',
        'rectangle.png',
        'polyline.png',
        'polygon.png',
        '3d-cube.png',
        'segmentation.png',
        'blur.png',
        'export.png',
        'save.png',
        'delete.png',
        'undo.png',
        'redo.png',
    ]

    methods = [
        qt_obj.upload_photos,
        qt_obj.upload_video,
        qt_obj.upload_annotation,
        qt_obj.auto_mode,
        qt_obj.edit_mode,
        qt_obj.draw_rectangle,
        qt_obj.draw_polyline,
        qt_obj.draw_polygon,
        qt_obj.draw_cube,
        qt_obj.segmentation_mode,
        qt_obj.apply_blur,
        qt_obj.export_work,
        qt_obj.save_changes,
        qt_obj.delete_element,
        qt_obj.undo_state,
        qt_obj.redo_state,
    ]

    tips = [
        'Select images from a folder and add them to the image list',
        'Add a video and convert it to frames and add them to the image list',
        'Load annotations from previous work',
        'Perform automatic anonymization using model',
        'Activate editor mode',
        'Draw rectangle',
        'Draw polyline',
        'Draw polygon',
        'Draw cube',
        'Segmentation',
        'Perform anonymization on the selected ROIs',
        'Export the anonymized video file',
        'Save the annotation work as .csv file',
        'Delete selected item form Image Label List',
        'Undo',
        'Redo',
    ]

    keys = 'OVLAMRPWCGBESDZY'

    tips = [f'Press Ctrl+{key}:  ' + tip for key, tip in zip(keys, tips)]
    key_shorts = [f'Ctrl+{key}' for key in keys]

    check_status = [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    # if number of elements same
    assert len(names) == len(icons) == len(methods) == len(tips) == len(key_shorts)

    for name, icon, method, tip, key, check in zip(
            names, icons, methods, tips, key_shorts, check_status
    ):
        tools[name] = [name, icon, method, tip, key, check]
    return tools

import copy 

class VertexCircle(QGraphicsEllipseItem):
    def __init__(self, x, y, vertex, i, parent=None):
        super(VertexCircle, self).__init__(parent)
        self.vertex = vertex
        self.vertex_ = copy.deepcopy(vertex)
        print("VVVVV",self.vertex)
        self.setRect(x-5, y-5, 10, 10)
        #self.setPen(Qt.NoPen)
        self.setBrush(Qt.black)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptDrops(True)
        self.parent = parent
        self.index = i

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange: # and self.scene():
            print("change",value, self.mapToScene(value), self.mapFromScene(value))
            #print(self.parent.mapToScene(value),self.parent.mapFromScene(value))
            #print(self.parent.pos(), self.mapToScene(self.parent.pos()))
            #print(self.parent.polygon()[0],self.parent.polygon()[1])
            print(self.vertex,self.vertex_)
            print(self.pos())
            #value = self.pos()
            polygon = self.parent.polygon()
            #value += self.vertex
            polygon[self.index] = value +self.vertex_ #+ self.vertex #self.mapToScene(value) #+ self.vertex #global_pos #local_pos #self.mapToScene(global_pos)
            self.parent.setPolygon(polygon)
            self.parent.update()
        return super(VertexCircle, self).itemChange(change, value)

class EllipseROI(QGraphicsEllipseItem):

    def __init__(self, viewer):
        QGraphicsItem.__init__(self)
        self._viewer = viewer
        pen = QPen(Qt.yellow)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.roiClicked(self)

# add one more class here
class RegularImageArea(QGraphicsView):
    """
    Display image/frame only area within the main interface.
    """

    leftMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    middleMouseButtonPressed = pyqtSignal(float, float)
    middleMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    viewChanged = pyqtSignal()
    mousePositionOnImageChanged = pyqtSignal(QPoint)
    roiSelected = pyqtSignal(int)

    def __init__(self, current_image, main_window, zoom_stack = [], rois = [], image_file_name=None, scene = None, zoom=None):
        """
        Initialize current image for display.
        Args:
            current_image: Path to target image.
            main_window: ImageLabeler instance.
        """
        super().__init__()
        if scene == None:
            self.scene = QGraphicsScene()
            self.setScene(self.scene)
        else:
            #self.scene = scene 
            self.scene = QGraphicsScene()
            #self.zoom = zoom
            # Set the zoom level of view2 to match view1
            #self.setTransform(QTransform.fromScale(self.zoom, self.zoom))
            self.setScene(self.scene)
        self.mode = "none"
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFocusPolicy(Qt.StrongFocus)
        self.current_image = current_image
        self.main_window = main_window
        self.row = None 
        global current_session
        self.main_window.session_data = self.main_window.session_data[:current_session]
        self.start_roi_point = QPoint()
        self.end_roi_point = QPoint()
        self.drag_roi = False
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.wheelZoomFactor = 1.25
        self.zoomStack = zoom_stack
        self._isZooming = False
        self._isPanning = False
        self._image = current_image
        self.regionZoomButton = Qt.MouseButton.LeftButton  # Drag a zoom box.
        self.zoomOutButton = None #Qt.MouseButton.RightButton  # Pop end of zoom stack (double click clears zoom stack).
        self.panButton = Qt.MouseButton.MiddleButton  # Drag to pan.
        self.ROIs = rois

        self._pixelPosition = QPoint()
        self._scenePosition = QPointF()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.image_file_name = image_file_name
        print("INIT BASE..........................")
        self.activate = False


        # temporary bounding box 

        self.rect_item = QGraphicsRectItem()
        self.rect_item.setPen(QPen(Qt.green))
        self.rect_item.setBrush(QBrush(Qt.transparent))
        self.initial_pos = None

        self.shape_items = []
        self.label_selected = False

        self.polygon = None
        self.polygon_point_list = []

        self.pen_cyan = QPen()
        self.pen_cyan.setColor(Qt.cyan)
        self.pen_cyan.setWidth(3)

        self.pen_green = QPen()
        self.pen_green.setColor(Qt.green)
        self.pen_green.setWidth(3)

        self.pen_red = QPen()
        self.pen_red.setColor(Qt.red)
        self.pen_red.setWidth(3)

        self.pen_blue = QPen()
        self.pen_blue.setColor(Qt.blue)
        self.pen_blue.setWidth(3)

        self.pen_none = QPen(Qt.NoPen)

        self.polyline = None 
        self.polyline_item = None
        self.cube_state = 0

    def hasImage(self):
        return self._image is not None

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack):
            self.fitInView(self.zoomStack[-1], self.aspectRatioMode)  # Show zoomed rect.
        else:
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image.

    def update_image(self, flag):
        if flag == "first":
            self.image_name = self.main_window.curr_frame_dir+"/"+self.main_window.curr_frame
            print("IMAGE NAME",self.image_name)
            self.pixmap = QPixmap(self.image_name)
            self.setPixmap(self.pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))

    def on_polygon_clicked(self, event):
        if event.button() == Qt.RightButton:
            for i,shape in enumerate(self.shape_items):
                if shape.contains(event.pos()):
                    print("Right-clicked inside polygon",i)
                    menu = QMenu(self)
                    delete_action = menu.addAction("Delete")
                    edit_action = menu.addAction("Edit")
                    action = menu.exec_(event.screenPos())
                    if action == delete_action:
                        print("delete")
                    elif action == edit_action:
                        # do something to edit the polygon
                        pass
                    break

    def keyPressEvent(self, event):
        global current_session
        if event.key() == Qt.Key_Shift:
            self.setCursor(QCursor(Qt.CrossCursor))
        elif event.key() == Qt.Key_Escape and (self.mode == "polygon" or self.mode == "segmentation") and self.label_selected:
            if self.polygon.size() < 3:
                return
            print("current_session,len(self.shape_items)")
            print(current_session,len(self.shape_items))
            self.polygon_item.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
            #self.polygon_item.setAcceptedMouseButtons(Qt.RightButton)
            #self.polygon_item.mousePressEvent = self.on_polygon_clicked
            if self.mode == "segmentation":
                self.polygon_item.setPen(self.pen_none)
                i = self.main_window.get_current_selection('selected_label')
                self.polygon_item.setBrush(categories_color[i-1])
            if current_session == len(self.shape_items):
                self.shape_items.append(self.polygon_item)
            else:
                self.shape_items[current_session] = self.polygon_item
            self.polygon_item.hide()
            self.store_coords_polygon()
            self.polygon_point_list.clear()
            self.polygon = None
        elif event.key() == Qt.Key_Escape and self.mode == "polyline" and self.label_selected:
            if self.polyline.elementCount() < 2:
                return
            print("current_session,len(self.shape_items)")
            print(current_session,len(self.shape_items))
            if current_session == len(self.shape_items):
                self.shape_items.append(self.polyline_item)
            else:
                self.shape_items[current_session] = self.polyline_item
            self.polyline_item.hide()
            self.store_coords_polyline()
            self.polyline = None 
            self.polyline_item = None
        elif event.key() == Qt.Key_Escape and self.mode == "segmentation edit":
            if current_session == len(self.shape_items):
                self.shape_items.append(self.polygon_item)
            else:
                self.shape_items[current_session] = self.polygon_item
            for circle in self.circles:
                circle.hide()
            self.store_coords_polygon(flag="edit")
            self.polygon_point_list.clear()
            self.polygon = None
            self.mode = "segmentation"
        elif event.key() == Qt.Key_E:
            self.zoomStack = [] 
            self.ROIs = [] 
            self.updateViewer()
            
    def store_coords_binary_mask(self, flag="append"):
        global current_session
        self.row = self.main_window.photo_row
        current_label_index = self.main_window.get_current_selection('selected_label')
        object_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        if current_label_index is None or current_label_index < 0:
            return
        frame_name = self.main_window.images[self.row]

        if flag == "append":
            self.main_window.annotation_id = current_session
            id = self.main_window.annotation_id
        elif flag == "edit":
            id = self.main_window.get_image_label_list_from_index(self.edited_index, attr="id")

        finalarray = [id, frame_name, object_name, current_label_index]

        #print("Storing Coordinates...")

        if self.mode == "polygon":
            finalarray.append("Polygon")
        elif self.mode == "segmentation" or self.mode == "segmentation edit":
            finalarray.append("Segmentation")
        finalarray.append(self.binary_mask)
        data = [finalarray]
        #print("Adding Coordinates...")
        #print(data)
        if flag == "append":
            current_session += 1
            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])
        elif flag == "edit":
            self.main_window.session_data.loc[self.edited_index] = data
            self.main_window.change_item_index(f'{data}', self.main_window.right_widgets['Image Label List'],self.edited_index)
        #print(self.main_window.right_widgets['Image Label List'].count())
        #if flag == "append":
        #    self.show_annotations(self.row)

    def store_coords_polygon(self, flag="append"):
        global current_session
        self.row = self.main_window.photo_row
        current_label_index = self.main_window.get_current_selection('selected_label')
        object_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        if current_label_index is None or current_label_index < 0:
            return
        frame_name = self.main_window.images[self.row]

        if flag == "append":
            self.main_window.annotation_id = current_session
            id = self.main_window.annotation_id
        elif flag == "edit":
            id = self.main_window.get_image_label_list_from_index(self.edited_index, attr="id")

        finalarray = [id, frame_name, object_name, current_label_index]

        #print("Storing Coordinates...")
        coordarray = []
        for i in range(self.polygon.size()):
            coordarray.append([int(self.polygon[i].x()),int(self.polygon[i].y())])

        if self.mode == "polygon":
            finalarray.append("Polygon")
        elif self.mode == "segmentation" or self.mode == "segmentation edit":
            finalarray.append("Segmentation")
        finalarray.append(coordarray)
        data = [finalarray]
        #print("Adding Coordinates...")
        #print(data)
        if flag == "append":
            current_session += 1
            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])
        elif flag == "edit":
            self.main_window.session_data.loc[self.edited_index] = data
            self.main_window.change_item_index(f'{data}', self.main_window.right_widgets['Image Label List'],self.edited_index)
        #print(self.main_window.right_widgets['Image Label List'].count())
        if flag == "append":
            self.show_annotations(self.row)

    def store_coords_polyline(self):
        global current_session
        self.row = self.main_window.photo_row
        current_label_index = self.main_window.get_current_selection('selected_label')
        object_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        if current_label_index is None or current_label_index < 0:
            return
        frame_name = self.main_window.images[self.row]
        self.main_window.annotation_id = current_session
        id = self.main_window.annotation_id
        finalarray = [id, frame_name, object_name, current_label_index]

        print("Storing polyLine Coordinates...")
        coordarray = []
        for i in range(self.polyline.elementCount()):
            coordarray.append([int(self.polyline.elementAt(i).x),int(self.polyline.elementAt(i).y)])

        finalarray.append("Polyline")
        finalarray.append(coordarray)
        data = [finalarray]
        print("Adding Coordinates...")
        to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
        self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
        current_session += 1
        self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])

        self.show_annotations(self.row)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def setImageFileName(self, image_file_name):
        print("SET IMAGE")
        pixmap = QPixmap(image_file_name)
        self.image_file_name = image_file_name
        if self.hasImage():
            print("SET IMAGE 1")
            self._image.setPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
        else:
            self._image = self.scene.addPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
            print("SET IMAGE 2")

    def setImageFrame(self, image_frame):
        print("SET IMAGE")
        height, width, channel = image_frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(image_frame.data, width, height, bytesPerLine, QImage.Format_BGR888)

        pixmap = QPixmap(qImg)
        if self.hasImage():
            print("SET IMAGE 1")
            self._image.setPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
        else:
            self._image = self.scene.addPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
            print("SET IMAGE 2")

    def switch_frame(self, frame_pixels):
        """
        Switch the current image displayed in the main window with the new one.
        Args:
            frame_pixels: Pixel array of new image to be displayed.

        Return:
            None
        """
        
        frame_pixels = cv2.cvtColor(frame_pixels, cv2.COLOR_BGR2RGB)
        self.current_image = frame_pixels
        #self.repaint()

        height, width, channels = frame_pixels.shape
        bytes_per_line = channels * width
        qimage = QImage(frame_pixels.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap(qimage)
        #self._image.setPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
        self._image = self.scene.addPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))


    def nparray_to_image(self, nparray):
        """
        Convert numpy array to QImage.
        Args:
            nparray: Pixel array of new image to display.

        Return:
            QImage
        """
        img = nparray[:, :, ::-1]
        # img = cv2.cvtColor(nparray, cv2.COLOR_BGR2RGB)
        img2 = im.fromarray(img.astype(np.uint8))

        qim = ImageQt(img2)
        # pix = QtGui.QPixmap.fromImage(qim)
        # return pix
        return qim

    def setMode(self, mode):
        self.mode = mode

    def mousePressEvent(self, event):
        """
        To get starting point of cursor for pan
        Args:
            event: QMouseEvent object.

        Return:
            None
        """
        global img_x, img_y, img_w, img_h
        print(event.button())
        #if event.button() == Qt.MiddleButton:
        #    print("MIDDLE BUTTON OK")
        #    return
        #else:
        #    print("NOT MIDDLE BUTTON")

        if (event.modifiers() == Qt.ShiftModifier):
            return

        if self.mode == "none":
            return

        if not self.check_label_selection():
            return
        else:
            self.label_selected = True 

        if self.mode == "cube":
            point = event.pos()
            multiplier = self.main_window.scaleFactor
            x = int((point.x() - img_x) * (1 / multiplier))
            y = int((point.y() - img_y) * (1 / multiplier))

            if event.button() == Qt.LeftButton:
                if not self.activate:
                    if self.cube_state == 0:
                        print("PRESS")
                        self.resizingDone = False
                        self.drag = False
                        # do regular work
                        self.right_click = False
                        self.start_point = event.pos()
                        self.begin = event.pos()
                        self.end = event.pos()

                        self.initial_pos = self.mapToScene(event.pos())
                        self.cube_rect_item = QGraphicsRectItem()
                        self.cube_rect_item.setZValue(1)
                        self.cube_rect_item.setPen(self.pen_cyan)
                        self.cube_rect_item.setRect(QRectF(self.initial_pos, QPointF(0, 0)))
                        self.scene.addItem(self.cube_rect_item)
                        #self.rect_item.show()
                        #self.update()

                elif self.activate:
                    # editing contour work
                    self.activatingCorners(x, y)

            elif event.button() == Qt.RightButton:
                self.right_click = True
                if self.activate:
                    if self.pointInBox(x, y, self.resized_x, self.resized_y, self.resized_w, self.resized_h):
                        self.disableHandlersAndEditing()
                        self.saveEditedContour()
                        self.show_annotations(self.row)
                        self.setCursor(QtCore.Qt.ArrowCursor)

                else:
                    self.menu_right = QMenu()
                    self.actionDelete = QAction()
                    self.actionDelete.setText("Delete")
                    icon1 = QtGui.QIcon()
                    icon1.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    self.actionDelete.setIcon(icon1)
                    self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                    self.menu_right.addAction(self.actionDelete)

                    # self.actionEdit = QAction()
                    # self.actionEdit.setText("Edit")
                    # icon2 = QtGui.QIcon()
                    # icon2.addPixmap(QtGui.QPixmap("../Icons/New/edit_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    # self.actionEdit.setIcon(icon2)
                    # self.actionEdit.triggered.connect(self.edit_selected_contour)
                    # self.menu_right.addAction(self.actionEdit)

                    rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                        self.labels_to_annotation('Delete')

                    if rectangle_contours and rectangle:
                        for contour in rectangle_contours:
                            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                            if check > -1:  # if the point is inside
                                # find contour coordinates in image label List
                                self.Id, self.label_id = self.main_window.find_matching_coordinates(contour, " 'Rectangle'")
                                # append the annotation id in delete_highlight
                                if self.Id > -1:
                                    rectangle_highlight = rectangle_contours.index(contour)
                                    edit_highlight.append(self.Id)
                                self.menu_right.popup(QtGui.QCursor.pos())  # show the menu

                self.show_annotations(self.row)
            rectangle_highlight = -1
        elif self.mode == "polyline":

            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                self.start_point = pos
                if not self.polyline_item:
                    self.polyline_item = QGraphicsPathItem()
                    self.polyline_item.setPen(self.pen_blue)
                    self.polyline_item.setZValue(1)
                    self.scene.addItem(self.polyline_item)
                self.polyline = self.polyline_item.path()
                if self.polyline.elementCount() == 0:
                    self.polyline.moveTo(pos)
                else:
                    self.polyline.lineTo(pos)
                self.polyline_item.setPath(self.polyline)


                #self.polyline_item_temp = QGraphicsPathItem()
                #self.polyline_item_temp.setPen(self.pen_blue)
                #self.polyline_item_temp.setZValue(1)
                #self.scene.addItem(self.polyline_item_temp)
                #self.polyline_temp = self.polyline_item_temp.path()
            elif event.button() == Qt.RightButton:
                self.right_click = True
                point = self.mapToScene(event.pos())

                x = point.x()
                y = point.y()

                rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                    self.labels_to_annotation('Delete')

                self.menu_right = QMenu()
                self.actionDelete = QAction()
                self.actionDelete.setText("Delete")
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.actionDelete.setIcon(icon)
                self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                self.menu_right.addAction(self.actionDelete)

                global polyline_highlight
                print("polyline",polyline)
                print(polyline_contours)
                if polyline_contours and polyline:
                    for contour in polyline_contours:
                        check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                        print("dist", check)
                        print("x")
                        if check == 1:  # if the point is on the line
                            print("polyline detected")
                            # find contour coordinates in image label List
                            ann_id = self.main_window.find_matching_coordinates(contour, " 'Polyline'")
                            # append the annotation id in delete_highlight
                            print(ann_id)
                            if ann_id > -1:
                                polyline_highlight = polyline_contours.index(contour)
                                edit_highlight.append(ann_id)
                            print("line highlight", polyline_highlight)
                            print("contour", contour)
                            self.menu_right.popup(QtGui.QCursor.pos())  # delete menu shown

                self.show_annotations(self.main_window.photo_row)
                polyline_highlight = -1
        elif self.mode == "polygon" or self.mode == "segmentation":
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                if not self.polygon:
                    self.polygon = QPolygonF([])
                    self.polygon.append(pos)
                    self.polygon_item = QGraphicsPolygonItem(self.polygon)
                    self.polygon_item.setPen(self.pen_red)
                    self.polygon_item.setZValue(1)
                    self.scene.addItem(self.polygon_item)
                else:
                    self.polygon.append(pos)
                    self.polygon_item.setPolygon(self.polygon)
                    #polygon_item.setPen(self.pen_red)
                    #polygon_item.setZValue(1)
                    #self.scene.addItem(polygon_item)

            
            elif event.button() == Qt.RightButton:
                pos = self.mapToScene(event.pos())
                for i,shape in enumerate(self.shape_items):
                    if shape.contains(pos):
                        print("Right-clicked inside polygon",i)
                        menu = QMenu(self)
                        delete_action = menu.addAction("Delete")
                        edit_action = menu.addAction("Edit")
                        action = menu.exec_(QPoint(event.screenPos().x(),event.screenPos().y())) #event.screenPos())
                        if action == delete_action:
                            #self.shape_items[i].hide()
                            #del self.shape_items[i]
                            #self.main_window.delete_from_list(self.main_window.right_widgets['Image Label List'], i)
                            self.main_window.delete_selected_contour(i)
                        elif action == edit_action:
                            self.mode = "segmentation edit"
                            self.edited_index = i
                            self.polygon = shape.polygon()
                            self.polygon_item = shape
                            self.circles = []
                            for j,point in enumerate(shape.polygon()):
                                circle = VertexCircle(point.x(), point.y(), point, j, shape)
                                circle.setBrush(Qt.white)
                                self.circles.append(circle)
                                self.scene.addItem(circle)
                        break

            '''
                self.right_click = True
                point = event.pos()
                self.update()

                #global polygon_highlight
                #global img_x, img_y, img_w, img_h
                multiplier = self.main_window.scaleFactor
                x = int((point.x() - img_x) * (1 / multiplier))
                y = int((point.y() - img_y) * (1 / multiplier))

                rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                    self.labels_to_annotation('Delete')

                self.menu_right = QMenu()
                self.actionDelete = QAction()
                self.actionDelete.setText("Delete")
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.actionDelete.setIcon(icon)
                self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                self.menu_right.addAction(self.actionDelete)

                if polygon_contours and polygon:
                    for contour in polygon_contours:
                        # print("polygon-contour", contour)
                        check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                        # print("dist", check)
                        if check > -1:  # if the point is inside
                            # find contour coordinates in image label List
                            ann_id = self.main_window.find_matching_coordinates(contour, " 'Polygon'")
                            # append the annotation id in highlight
                            if ann_id > -1:
                                print("poly id found")
                                edit_highlight.append(ann_id)
                                polygon_highlight = polygon_contours.index(contour)
                            print("in Polygon highlight", polygon_highlight)
                            print("contour", contour)
                            self.menu_right.popup(QtGui.QCursor.pos())

                self.show_annotations(self.main_window.photo_row)
                polygon_highlight = -1

            self.right_click = False
            '''
        elif self.mode == "rectangle":
            point = self.mapToScene(event.pos())
            multiplier = self.main_window.scaleFactor
            x = point.x()
            y = point.y()

            if event.button() == Qt.LeftButton:
                if not self.activate:
                    print("PRESS")
                    self.resizingDone = False
                    self.drag = False
                    # do regular work
                    self.right_click = False
                    self.start_point = event.pos()
                    self.begin = event.pos()
                    self.end = event.pos()

                    self.initial_pos = self.mapToScene(event.pos())
                    self.rect_item = QGraphicsRectItem()
                    self.rect_item.setZValue(1)
                    self.rect_item.setPen(self.pen_green)
                    self.rect_item.setRect(QRectF(self.initial_pos, QPointF(0, 0)))
                    self.scene.addItem(self.rect_item)
                    #self.rect_item.show()
                    #self.update()

                elif self.activate:
                    # editing contour work
                    self.activatingCorners(x, y)

            elif event.button() == Qt.RightButton:
                self.right_click = True
                if self.activate:
                    if self.pointInBox(x, y, self.resized_x, self.resized_y, self.resized_w, self.resized_h):
                        self.disableHandlersAndEditing()
                        self.saveEditedContour()
                        self.show_annotations(self.row)
                        self.setCursor(QtCore.Qt.ArrowCursor)

                else:
                    self.menu_right = QMenu()
                    self.actionDelete = QAction()
                    self.actionDelete.setText("Delete")
                    icon1 = QtGui.QIcon()
                    icon1.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    self.actionDelete.setIcon(icon1)
                    self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                    self.menu_right.addAction(self.actionDelete)

                    # self.actionEdit = QAction()
                    # self.actionEdit.setText("Edit")
                    # icon2 = QtGui.QIcon()
                    # icon2.addPixmap(QtGui.QPixmap("../Icons/New/edit_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    # self.actionEdit.setIcon(icon2)
                    # self.actionEdit.triggered.connect(self.edit_selected_contour)
                    # self.menu_right.addAction(self.actionEdit)
                    print(self.main_window.right_widgets['Image Label List'].count())
                    print("VVVVVVVVVVVVVVVVVVVV")
                    rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                        self.labels_to_annotation('Delete')

                    if rectangle_contours and rectangle:
                        for contour in rectangle_contours:
                            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                            if check > -1:  # if the point is inside
                                # find contour coordinates in image label List
                                self.Id, self.label_id = self.main_window.find_matching_coordinates(contour, " 'Rectangle'")
                                # append the annotation id in delete_highlight
                                print("IIIIIDDDDDD",self.Id)
                                if self.Id > -1:
                                    rectangle_highlight = rectangle_contours.index(contour)
                                    edit_highlight.append(self.Id)
                                self.menu_right.popup(QtGui.QCursor.pos())  # show the menu

                self.show_annotations(self.row)
            rectangle_highlight = -1


        self.zoomPanROIPress(event)


    def itemChange(self, change, value):
        print("fgggfgfgfgf",change, value)
        if change == QGraphicsItem.ItemPositionChange:
            # Do something when the position of the item changes
            pass
        elif change == QGraphicsItem.ItemRotationChange:
            # Do something when the rotation of the item changes
            pass
        elif change == QGraphicsItem.ItemScaleChange:
            # Do something when the scale of the item changes
            pass

        # Call the base class implementation of itemChange
        return super().itemChange(change, value)

    def zoomPanROIPress(self, event):
        print("zoom pan roi press")
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            return

        if (self.regionZoomButton is not None) and (event.button() == self.regionZoomButton) and (event.modifiers() == Qt.ShiftModifier):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            QGraphicsView.mousePressEvent(self, event)
            event.accept()
            self._isZooming = True
            return

        if (self.zoomOutButton is not None) and (event.button() == self.zoomOutButton):
            if len(self.zoomStack):
                self.zoomStack.pop()
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        # Start dragging to pan?
        if (self.panButton is not None) and (event.button() == self.panButton):
            self._pixelPosition = event.pos()  # store pixel position
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mousePressEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier
                                                     | Qt.KeyboardModifier.ControlModifier
                                                     | Qt.KeyboardModifier.AltModifier
                                                     | Qt.KeyboardModifier.MetaModifier)
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(event.pos()), Qt.MouseButton.LeftButton,
                                         event.buttons(), dummyModifiers)
                self.mousePressEvent(dummyEvent)
            sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
            self._scenePosition = sceneViewport.topLeft()
            event.accept()
            self._isPanning = True
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        """
        Calculating the displacement in image position
        Args:
            event: QMouseEvent object.

        Return:
            None
        """
        # pan
        self.zoomPanROIMove(event)

        if (event.modifiers() == Qt.ShiftModifier):
            return
        
        if self.mode == "none":
            return

        if not self.label_selected:
            return
        if self.mode == "polyline":
                #print("move polyline")
                current_pos = self.mapToScene(event.pos())
                
                #self.polyline_temp.moveTo(self.start_point)
                #self.polyline_temp.lineTo(current_pos)
                #self.polyline_item_temp.setPath(self.polyline_temp)

        elif self.mode == "rectangle":
                current_pos = self.mapToScene(event.pos())

                # Set the position and size of the QGraphicsRectItem
                self.rect_item.setRect(QRectF(self.initial_pos, current_pos))


        elif self.mode == "cube":
                current_pos = self.mapToScene(event.pos())

                # Set the position and size of the QGraphicsRectItem
                self.cube_rect_item.setRect(QRectF(self.initial_pos, current_pos))


    def zoomPanROIMove(self, event):
        if self._isPanning:
            QGraphicsView.mouseMoveEvent(self, event)
            if len(self.zoomStack) > 0:
                sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
                delta = sceneViewport.topLeft() - self._scenePosition
                self._scenePosition = sceneViewport.topLeft()
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(self.sceneRect())
                self.updateViewer()
                self.viewChanged.emit()

        scenePos = self.mapToScene(event.pos())
        if self.sceneRect().contains(scenePos):
            # Pixel index offset from pixel center.
            x = int(round(scenePos.x() - 0.5))
            y = int(round(scenePos.y() - 0.5))
            imagePos = QPoint(x, y)
        else:
            # Invalid pixel position.
            imagePos = QPoint(-1, -1)
        self.mousePositionOnImageChanged.emit(imagePos)

        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        """
        To signal end of movement of cursor for pan
        Args:
            event: QMouseEvent object.

        Return:
            None
        """
        global current_session 

        self.zoomPanROIRelease(event)

        if (event.modifiers() == Qt.ShiftModifier):
            return

        if self.mode == "none":
            return

        if not self.label_selected:
            return

        if self.mode == "rectangle":

            if not self.right_click and not self.activate:

                # Record the final position of the mouse
                final_pos = self.mapToScene(event.pos())
                print("RELEASE",self.initial_pos, final_pos)

                # Set the position and size of the QGraphicsRectItem
                self.rect_item.setRect(QRectF(self.initial_pos, final_pos))
                self.rect_item.hide()
                #self.scene.addItem(self.rect_item)
                print("current_session,len(self.shape_items)")
                print(current_session, len(self.shape_items))
                if current_session == len(self.shape_items):
                    self.shape_items.append(self.rect_item)
                else:
                    self.shape_items[current_session] = self.rect_item

                h = int(final_pos.y()-self.initial_pos.y())
                w = int(final_pos.x()-self.initial_pos.x())
                self.update_session_data_rectangle(int(self.initial_pos.x()), int(self.initial_pos.y()), w, h)
                print(self.main_window.right_widgets['Image Label List'].count())
                print("CCCCCCCCCCCCCCCCC")
                self.show_annotations(self.row)
        elif self.mode == "cube":

            if not self.right_click and not self.activate:
                final_pos = self.mapToScene(event.pos())
                if self.cube_state == 0:
                    self.cube_state = 1 
                    self.cube_h = int(final_pos.y()-self.initial_pos.y())
                    self.cube_w = int(final_pos.x()-self.initial_pos.x())
                    self.cube_start = self.initial_pos
                    self.cube_rect_item.setRect(QRectF(self.initial_pos, final_pos))
                else:
                    self.cube_state = 0
                    self.cube_rect_item.hide()

                    self.cube_polygon1 = QPolygonF([self.cube_start,final_pos,QPointF(final_pos.x()+self.cube_w,final_pos.y()),QPointF(self.cube_start.x()+self.cube_w,self.cube_start.y())])
                    self.cube_polygon1_item = QGraphicsPolygonItem(self.cube_polygon1)
                    self.cube_polygon1_item.setPen(self.pen_green)
                    self.cube_polygon1_item.setZValue(1)
                    self.scene.addItem(self.cube_polygon1_item)
                    self.cube_polygon1_item.hide()

                    self.cube_polygon2 = QPolygonF([QPointF(self.cube_start.x(),self.cube_start.y()+self.cube_h),QPointF(final_pos.x(),final_pos.y()+self.cube_h),QPointF(final_pos.x()+self.cube_w,final_pos.y()+self.cube_h),QPointF(self.cube_start.x()+self.cube_w,self.cube_start.y()+self.cube_h)])
                    self.cube_polygon2_item = QGraphicsPolygonItem(self.cube_polygon2)
                    self.cube_polygon2_item.setPen(self.pen_green)
                    self.cube_polygon2_item.setZValue(1)
                    self.scene.addItem(self.cube_polygon2_item)
                    self.cube_polygon2_item.hide()

                    self.cube_polygon3 = QPolygonF([final_pos,QPointF(final_pos.x()+self.cube_w,final_pos.y()),QPointF(final_pos.x()+self.cube_w,final_pos.y()+self.cube_h),QPointF(final_pos.x(),final_pos.y()+self.cube_h)])
                    self.cube_polygon3_item = QGraphicsPolygonItem(self.cube_polygon3)
                    self.cube_polygon3_item.setPen(self.pen_green)
                    self.cube_polygon3_item.setZValue(1)
                    self.scene.addItem(self.cube_polygon3_item)
                    self.cube_polygon3_item.hide()

                    self.cube_rect_item = QGraphicsRectItem()
                    self.cube_rect_item.setZValue(1)
                    self.cube_rect_item.setPen(self.pen_cyan)
                    self.cube_rect_item.setRect(QRectF(self.cube_start, QPointF(self.cube_start.x()+self.cube_w, self.cube_start.y()+self.cube_h)))
                    self.scene.addItem(self.cube_rect_item)
                    self.cube_rect_item.hide()

                    items = (self.cube_polygon1_item,self.cube_polygon2_item,self.cube_polygon3_item,self.cube_rect_item)
                    if current_session == len(self.shape_items):
                        self.shape_items.append(items)
                    else:
                        self.shape_items[current_session] = items
                    #self.polygon.append(pos)
                    #self.polygon_item.setPolygon(self.polygon)
                    self.update_session_data_cube(int(self.cube_start.x()), int(self.cube_start.y()), int(final_pos.x()), int(final_pos.y()), int(self.cube_w), int(self.cube_h))
                    #print("row",self.row)
                    self.show_annotations(self.row)

    def zoomPanROIRelease(self, event):
        dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        if event.modifiers() == dummyModifiers:
            QGraphicsView.mouseReleaseEvent(self, event)
            event.accept()
            return

        # Finish dragging a region zoom box?
        if (self.regionZoomButton is not None) and (event.button() == self.regionZoomButton):
            QGraphicsView.mouseReleaseEvent(self, event)
            zoomRect = self.scene.selectionArea().boundingRect().intersected(self.sceneRect())
            # Clear current selection area (i.e. rubberband rect).
            self.scene.setSelectionArea(QPainterPath())
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # If zoom box is 3x3 screen pixels or smaller, do not zoom and proceed to process as a click release.
            zoomPixelWidth = abs(event.pos().x() - self._pixelPosition.x())
            zoomPixelHeight = abs(event.pos().y() - self._pixelPosition.y())
            if zoomPixelWidth > 3 and zoomPixelHeight > 3:
                if zoomRect.isValid() and (zoomRect != self.sceneRect()):
                    self.zoomStack.append(zoomRect)
                    self.updateViewer()
                    self.viewChanged.emit()
                    event.accept()
                    self._isZooming = False
                    return

        # Finish panning?
        if (self.panButton is not None) and (event.button() == self.panButton):
            if self.panButton == Qt.MouseButton.LeftButton:
                QGraphicsView.mouseReleaseEvent(self, event)
            else:
                # ScrollHandDrag ONLY works with LeftButton, so fake it.
                # Use a bunch of dummy modifiers to notify that event should NOT be handled as usual.
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                dummyModifiers = Qt.KeyboardModifier(Qt.KeyboardModifier.ShiftModifier
                                                     | Qt.KeyboardModifier.ControlModifier
                                                     | Qt.KeyboardModifier.AltModifier
                                                     | Qt.KeyboardModifier.MetaModifier)
                dummyEvent = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(event.pos()),
                                         Qt.MouseButton.LeftButton, event.buttons(), dummyModifiers)
                self.mouseReleaseEvent(dummyEvent)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if len(self.zoomStack) > 0:
                sceneViewport = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
                delta = sceneViewport.topLeft() - self._scenePosition
                self.zoomStack[-1].translate(delta)
                self.zoomStack[-1] = self.zoomStack[-1].intersected(self.sceneRect())
                self.viewChanged.emit()
            event.accept()
            self._isPanning = False
            return

        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.middleMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

        QGraphicsView.mouseReleaseEvent(self, event)

    def wheelEvent(self, event):
        if self.wheelZoomFactor is not None:
            if self.wheelZoomFactor == 1:
                return
            if event.angleDelta().y() > 0:
                # zoom in
                if len(self.zoomStack) == 0:
                    self.zoomStack.append(self.sceneRect())
                elif len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() / self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() / self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                self.updateViewer()
                self.viewChanged.emit()
            else:
                # zoom out
                if len(self.zoomStack) == 0:
                    # Already fully zoomed out.
                    return
                if len(self.zoomStack) > 1:
                    del self.zoomStack[:-1]
                zoomRect = self.zoomStack[-1]
                center = zoomRect.center()
                zoomRect.setWidth(zoomRect.width() * self.wheelZoomFactor)
                zoomRect.setHeight(zoomRect.height() * self.wheelZoomFactor)
                zoomRect.moveCenter(center)
                self.zoomStack[-1] = zoomRect.intersected(self.sceneRect())
                if self.zoomStack[-1] == self.sceneRect():
                    self.zoomStack = []
                self.updateViewer()
                self.viewChanged.emit()
            event.accept()
            return

        QGraphicsView.wheelEvent(self, event)


    #def enterEvent(self, event):
    #    self.setCursor(Qt.CursorShape.CrossCursor)

    #def leaveEvent(self, event):
    #    self.setCursor(Qt.CursorShape.ArrowCursor)

    def addROIs(self, rois):
        for roi in rois:
            self.scene.addItem(roi)
            self.ROIs.append(roi)

    def deleteROIs(self, rois):
        for roi in rois:
            self.scene.removeItem(roi)
            self.ROIs.remove(roi)
            del roi

    def clearROIs(self):
        for roi in self.ROIs:
            self.scene.removeItem(roi)
        del self.ROIs[:]

    def roiClicked(self, roi):
        for i in range(len(self.ROIs)):
            if roi is self.ROIs[i]:
                self.roiSelected.emit(i)
                print(i)
                break

    def setROIsAreMovable(self, tf):
        if tf:
            for roi in self.ROIs:
                roi.setFlags(roi.flags() | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        else:
            for roi in self.ROIs:
                roi.setFlags(roi.flags() & ~QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

    def addSpots(self, xy, radius):
        for xy_ in xy:
            x, y = xy_
            spot = EllipseROI(self)
            spot.setRect(x - radius, y - radius, 2 * radius, 2 * radius)
            self.scene.addItem(spot)
            self.ROIs.append(spot)

    def labels_to_annotation(self, option):
        print("*annotating*")
        global current_session
        objtype_rectangle, objtype_polygon, objtype_polyline, objtype_cube = False, False, False, False
        Box_rectangle, Box_polygon, Box_polyline, Box_cube = [], [], [], []
        print("DEBUUUUUG",current_session)
        print(self.main_window.right_widgets['Image Label List'])
        print(self.main_window.right_widgets['Image Label List'].count())
        sequence = {"rect":[],"polygon":[],"polyline":[],"cube":[]}
        for i in range(current_session): #self.main_window.right_widgets['Image Label List'].count()):
            session_name = str((self.main_window.right_widgets['Image Label List'].item(i).text()))
            obj_name = session_name.split(',')
            obj_len = len(obj_name)
            typeofshape = obj_name[4]

            coords = str(obj_name[5:obj_len])
            coords = coords.replace("[", "")
            coords = coords.replace("]", "")
            coords = coords.replace("'", "")
            coords = coords.replace("\"", "")
            coords = coords.replace(" ", "")
            coords = coords.split(",")

            if typeofshape == " 'Rectangle'":
                sequence["rect"].append(i)
                # print("rectangle")
                pts, pts2 = [], []
                objtype_rectangle = True
                # print("coords", coords)  # ['581', '107', '347', '173']
                x = int(coords[0])
                y = int(coords[1])
                w = int(coords[2])
                h = int(coords[3])

                x1, y1 = x, y
                x2, y2 = x + w, y + h

                # all four coordinates for the rectangle
                i_tl, i_tr = (x, y), (x + w, y)
                i_br, i_bl = (x + w, y + h), (x, y + h)

                if option == 'Display':
                    print('display rectangle')
                    for item in [x1, y1, x2, y2]:
                        pts.append(item)

                elif option == 'Delete':
                    print('delete rectangle')
                    for item in [i_tl, i_tr, i_br, i_bl]:
                        pts.append(item)

                pts2 = pts.copy()
                Box_rectangle.append(pts2)
                pts.clear()

            elif typeofshape == " 'Cube'":
                sequence["cube"].append(i)
                pts, pts2 = [], []
                objtype_cube = True
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                w = int(coords[4])
                h = int(coords[5])

                # all four coordinates for the rectangle
                f_tl, f_tr = (x1, y1), (x1 + w, y1)
                f_br, f_bl = (x1 + w, y1 + h), (x1, y1 + h)

                b_tl, b_tr = (x2, y2), (x2 + w, y2)
                b_br, b_bl = (x2 + w, y2 + h), (x2, y2 + h)

                #   (b_tl)____________(b_tr)
                #       | \ d         |\
                #       |  \__________|_\(f_tl)
                #       |  |          |  |
                # (b_bl)|__|__________|  |
                #        \ |           \ |
                #   (f_br)\|____________\|(f_bl)

                if option == 'Display':
                    print('display cube')
                    for item in [f_tl, b_tl, w, h]:
                        pts.append(item)

                elif option == 'Delete':
                    print('delete cube')
                    for item in [b_bl, b_tl, b_tr, f_tr, f_br, f_bl]:
                        pts.append(item)

                pts2 = pts.copy()
                Box_cube.append(pts2)
                pts.clear()

            elif typeofshape == " 'Polygon'" or typeofshape == " 'Segmentation'":
                sequence["polygon"].append(i)
                objtype_polygon = True
                length = len(coords)
                finalcoords, finalcoords2 = [], []
                loop = 0
                while loop < length:
                    x_coord = int(coords[loop])
                    y_coord = int(coords[loop + 1])
                    a = [x_coord, y_coord]
                    finalcoords.append(a)
                    loop += 2

                finalcoords2 = finalcoords.copy()
                Box_polygon.append(finalcoords2)

            elif typeofshape == " 'Polyline'":
                sequence["polyline"].append(i)
                objtype_polyline = True
                length = len(coords)
                finalcoords, finalcoords_ = [], []
                loop = 0
                while loop < length:
                    x_coord = int(coords[loop])
                    y_coord = int(coords[loop + 1])
                    a = [x_coord, y_coord]
                    finalcoords.append(a)
                    loop += 2

                finalcoords_ = finalcoords.copy()
                Box_polyline.append(finalcoords_)
            else:
                pass
        if option == "Display" or option == "Delete":
            return objtype_rectangle, objtype_polygon, objtype_polyline, objtype_cube, Box_rectangle, Box_polygon, Box_polyline, Box_cube
        elif option == "Display2":
            return objtype_rectangle, objtype_polygon, objtype_polyline, objtype_cube, Box_rectangle, Box_polygon, Box_polyline, Box_cube, sequence

    def show_annotations(self, row):
        """
        Display the annotations on image
        Args:
            Current row in Image List.

        Return:
            None
        """
        rectangle, polygon, polyline, cube, ROI_rec, ROI_pg, ROI_pl, ROI_cube, seq = self.labels_to_annotation('Display2')
        # row = index of image in Image list
        if not self.main_window.video_flag:
            frame_read = self.main_window.image_pixels[row]
        else:
            frame_read = self.main_window.video_frame_pixels[row]
        to_draw = frame_read.copy()

        # if no shapes available
        if not ROI_rec and not ROI_cube and not ROI_pg and not ROI_pl:
            self.switch_frame(to_draw)

        global rectangle_highlight, polygon_highlight, polyline_highlight, cube_highlight

        global current_session 

        if rectangle:
            for i,roi in enumerate(ROI_rec):
                print("DEBUG rect",i)
                if ROI_rec.index(roi) is not rectangle_highlight:
                    # cv2.putText(to_draw,'Car', (roi[0], roi[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,255,0),2)
                    # cv2.rectangle(image, start_point, end_point, color, thickness)
                    print("999999999999999999999", roi)
                    #to_draw = cv2.rectangle(to_draw, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

                    #self.shape_items[i].hide()
                    #self.shape_items[i].setRect(QRectF(QPoint(roi[0],roi[1]), QPoint(roi[0]+roi[2],roi[1]+roi[3])))
                    print(i,current_session)
                    #if i < current_session:
                    self.shape_items[seq['rect'][i]].show()
                    #else:
                    #    self.shape_items[i].hide()
                else:
                    # if size of rectangle is small don't add handlers
                    global handle
                    t = 2
                    color = (0, 255, 0)
                    width, height = (roi[2] - roi[0]), (roi[3] - roi[1])

                    # Rectangle
                    to_draw = cv2.rectangle(to_draw, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), 2)
                    # Top-Left
                    to_draw = cv2.rectangle(to_draw, (roi[0] - handle, roi[1] - handle),
                                            (roi[0] + handle, roi[1] + handle), color, t)
                    # Top-Right
                    to_draw = cv2.rectangle(to_draw, (roi[2] - handle, roi[1] - handle),
                                            (roi[2] + handle, roi[1] + handle), color, t)
                    # Bottom-Left
                    to_draw = cv2.rectangle(to_draw, (roi[0] - handle, roi[3] - handle),
                                            (roi[0] + handle, roi[3] + handle), color, t)
                    # Bottom-Right
                    to_draw = cv2.rectangle(to_draw, (roi[2] - handle, roi[3] - handle),
                                            (roi[2] + handle, roi[3] + handle), color, t)
                    # Top-Mid
                    to_draw = cv2.rectangle(to_draw, (roi[0] + int(width / 2) - handle, roi[1] - handle),
                                            (roi[0] + int(width / 2) + handle, roi[1] + handle), color, t)
                    # Bottom-Mid
                    to_draw = cv2.rectangle(to_draw, (roi[0] + int(width / 2) - handle, roi[3] - handle),
                                            (roi[0] + int(width / 2) + handle, roi[3] + handle), color, t)
                    # Left-Mid
                    to_draw = cv2.rectangle(to_draw, (roi[0] - handle, roi[3] - int(height / 2) - handle),
                                            (roi[0] + handle, roi[1] + int(height / 2) + handle), color, t)
                    # Right-Mid
                    to_draw = cv2.rectangle(to_draw, (roi[2] - handle, roi[3] - int(height / 2) - handle),
                                            (roi[2] + handle, roi[1] + int(height / 2) + handle), color, t)
                self.switch_frame(to_draw)

        if polygon:
            for i,roi in enumerate(ROI_pg):
                '''
                print("DEBUG polygon",i)
                pts = np.array(roi, np.int32)
                pts = pts.reshape(-1, 1, 2)
                if ROI_pg.index(roi) is not polygon_highlight:
                    cv2.polylines(to_draw, [pts], True, (0, 255, 255), 2)
                else:
                    cv2.polylines(to_draw, [pts], True, (0, 0, 0), 2)
                    for p in pts:
                        cv2.circle(to_draw, (p[0][0], p[0][1]), 5, (0, 255, 0), thickness=2, lineType=5,
                                   shift=0)
                self.switch_frame(to_draw)
                '''
                self.shape_items[seq['polygon'][i]].show()

        if polyline:
            for i,roi in enumerate(ROI_pl):
                self.shape_items[seq['polyline'][i]].show()
                '''
                print("DEBUG polyline",i)
                pts = np.array(roi, np.int32)
                pts = pts.reshape(-1, 1, 2)
                if ROI_pl.index(roi) is not polyline_highlight:
                    cv2.polylines(to_draw, [pts], False, (255, 255, 0), 3)
                else:
                    cv2.polylines(to_draw, [pts], False, (0, 0, 0), 2)
                    for p in pts:
                        cv2.circle(to_draw, (p[0][0], p[0][1]), 5, (0, 255, 0), thickness=2, lineType=5,
                                   shift=0)
                self.switch_frame(to_draw)
                '''

        if cube:
            for i,roi in enumerate(ROI_cube):
                print("DEBUG cube",i)
                for j in range(4):
                    self.shape_items[seq['cube'][i]][j].show()
                '''
                t = 2
                bg_color = fr_color = []
                if ROI_cube.index(roi) is not cube_highlight:
                    bg_color = (0, 255, 0)
                    fr_color = (255, 255, 0)

                    #   roi[0]____________roi[1]
                    #       |              |
                    #       |              |
                    #       |              |
                    # roi[3]|______________|roi[2]

                    # faces
                    self.drawCubeFaces(to_draw, roi, fr_color, bg_color, t)
                else:
                    bg_color = (0, 0, 0)
                    handle_color = fr_color = (255, 255, 255)

                    # faces
                    self.drawCubeFaces(to_draw, roi, fr_color, bg_color, t)
                    self.drawCubeHandlers(to_draw, roi, handle_color, -t)
                '''
            self.switch_frame(to_draw)

    def drawCubeFaces(self, to_draw, roi, fr_color, bg_color, t):
        # roi = [f_tl, b_tl, w, h]
        x1, y1 = roi[0][0], roi[0][1]
        x2, y2 = roi[1][0], roi[1][1]
        w, h = roi[2], roi[3]
        """front face"""
        # clock wise drawn
        cv2.line(to_draw, (x1, y1), (x1 + w, y1), fr_color, t)
        cv2.line(to_draw, (x1 + w, y1), (x1 + w, y1 + h), fr_color, t)
        cv2.line(to_draw, (x1 + w, y1 + h), (x1, y1 + h), fr_color, t)
        cv2.line(to_draw, (x1, y1 + h), (x1, y1), fr_color, t)

        contours = np.array([(x1, y1), (x1 + w, y1), (x1 + w, y1 + h), (x1, y1 + h)])
        # cv2.fillPoly(to_draw, pts=[contours], color=[])

        """rare face"""
        cv2.line(to_draw, (x2, y2), (x2 + w, y2), bg_color, t)
        cv2.line(to_draw, (x2 + w, y2), (x2 + w, y2 + h), bg_color, t)
        cv2.line(to_draw, (x2 + w, y2 + h), (x2, y2 + h), bg_color, t)
        cv2.line(to_draw, (x2, y2 + h), (x2, y2), bg_color, t)

        """edges"""
        cv2.line(to_draw, (x1, y1), (x2, y2), bg_color, t)
        cv2.line(to_draw, (x1 + w, y1), (x2 + w, y2), bg_color, t)
        cv2.line(to_draw, (x1 + w, y1 + h), (x2 + w, y2 + h), bg_color, t)
        cv2.line(to_draw, (x1, y1 + h), (x2, y2 + h), bg_color, t)

    def drawCubeHandlers(self, to_draw, roi, handle_color, t):
        """handler on front face """
        # roi = [f_tl, b_tl, w, h]
        x1, y1 = roi[0][0], roi[0][1]
        x2, y2 = roi[1][0], roi[1][1]
        w, h = roi[2], roi[3]
        # top-left
        cv2.rectangle(to_draw, (x1 - handle, y1 - handle),
                      (x1 + handle, y1 + handle), handle_color, t)
        # top-right
        cv2.rectangle(to_draw, (x1 + w - handle, y1 - handle),
                      (x1 + w + handle, y1 + handle), handle_color, t)
        # bottom-right
        cv2.rectangle(to_draw, (x1 + w - handle, y1 + h - handle),
                      (x1 + w + handle, y1 + h + handle), handle_color, t)
        # bottom-left
        cv2.rectangle(to_draw, (x1 - handle, y1 + h - handle),
                      (x1 + handle, y1 + h + handle), handle_color, t)

        """ handler on front face edges"""
        # Top-Mid
        cv2.rectangle(to_draw, (x1 + int(w / 2) - handle, y1 - handle),
                      (x1 + int(w / 2) + handle, y1 + handle), handle_color, t)
        # Bottom-Mid
        cv2.rectangle(to_draw, (x1 + int(w / 2) - handle, y1 + h - handle),
                      (x1 + int(w / 2) + handle, y1 + h + handle), handle_color, t)
        # Right-Mid
        cv2.rectangle(to_draw, (x1 + w - handle, y1 + int(h / 2) - handle),
                      (x1 + w + handle, y1 + int(h / 2) + handle), handle_color, t)
        # Left-Mid
        cv2.rectangle(to_draw, (x1 - handle, y1 + int(h / 2) - handle),
                      (x1 + handle, y1 + int(h / 2) + handle), handle_color, t)

        """ handler on rare face """
        if x1 > x2:
            # left
            cv2.rectangle(to_draw, (x2 - handle, y2 - handle),
                          (x2 + handle, y2 + handle), handle_color, t)
            cv2.rectangle(to_draw, (x2 - handle, y2 + h - handle),
                          (x2 + handle, y2 + h + handle), handle_color, t)

        if x1 < x2:
            # right
            cv2.rectangle(to_draw, (x2 + w - handle, y2 - handle),
                          (x2 + w + handle, y2 + handle), handle_color, t)
            cv2.rectangle(to_draw, (x2 + w - handle, y2 + h - handle),
                          (x2 + w + handle, y2 + h + handle), handle_color, t)
        """ handler on rare edge"""
        return

    def check_label_selection(self):
        current_label_index = self.main_window.get_current_selection('selected_label')
        if current_label_index is None or current_label_index < 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Add label")
            msg.setInformativeText('Choose a label from labels list or add a new label to continue annotation')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False 
        return True


    def update_session_data_rectangle(self, x, y, w, h):
        """
        Add a row to session_data containing calculated ratios.
        Args:
            x: Start x coordinate.
            y: Start y coordinate.
            w: x direction width.
            h: y direction height.

        Return:
            None
        """
        global current_session 

        current_label_index = self.main_window.get_current_selection('selected_label')  # selected label

        # if no label return
        if current_label_index is None or current_label_index < 0:
            return

        label_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        frame_name = self.main_window.images[self.row]
        if w != 0 and h != 0 and not self.right_click:
            rectangle_coordinates = [x, y, w, h]

            self.main_window.annotation_id = current_session
            a_id = self.main_window.annotation_id
            data = [[a_id, frame_name, label_name, current_label_index, "Rectangle", [rectangle_coordinates]]]

            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            current_session += 1
            print('yyyyyyyyyyyyyyyyy', rectangle_coordinates)
            print(self.main_window.session_data)
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])

    def update_session_data_cube(self, x1, y1, x2, y2, w, h):
        global current_session
        current_label_index = self.main_window.get_current_selection('selected_label')
        # if no label return
        if current_label_index is None or current_label_index == 0:
            return

        label_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        frame_name = self.main_window.images[self.row]
        if w != 0 and h != 0 and not self.right_click:
            cube_coordinates = [x1, y1, x2, y2, w, h]

            self.main_window.annotation_id = current_session
            a_id = self.main_window.annotation_id
            data = [[a_id, frame_name, label_name, current_label_index, "Cube", [cube_coordinates]]]

            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            current_session += 1
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])


class ImageEditorArea(RegularImageArea):
    """
    Edit and display area within the main interface.
    """

    def __init__(self, current_image, main_window, zoom_stack = [], rois = [], image_file_name = None, scene=None, zoom = None):
        """
        Initialize current image for display.
        Args:
            current_image: Path to target image.
            main_window: ImageLabeler instance.
        """
        super().__init__(current_image, main_window, zoom_stack, rois, image_file_name, scene, zoom)
        self.main_window = main_window
        self.zoomStack = [] #zoom_stack
        self.ROIs = rois
        self._image = current_image 
        global current_session
        self.main_window.session_data = self.main_window.session_data[:current_session]
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.begin = QPoint()
        self.end = QPoint()
        self.right_click = False
        self.row = self.main_window.photo_row

        """ RESIZING related variables and flags"""
        # Return flag
        self.resizingDone = False  # for task completion
        self.activate = False  # contour activated -> deactivate primitive function like making rectangle using press,move, release events of mouse
        self.drag = False  # corner movement of rectangle

        # Corner flags for positions
        self.markerPos = 0

        # coords after resizing
        self.resized_x = 0
        self.resized_y = 0
        self.resized_w = 0
        self.resized_h = 0

        # anchor/holder
        self.anchor_x = 0
        self.anchor_y = 0
        self.anchor_w = 0
        self.anchor_h = 0

        # selected rectangle info
        self.Id = None
        self.label_id = None

        self.image_file_name = image_file_name

        # for undo and redo
        #self.current_session = 0
        print("ImageEditorArea", self.image_file_name, self.scene, self._image, self.hasImage(), self.ROIs, self.zoomStack)
        #self.setImageFileName(self.image_file_name)
        #pixmap = QPixmap(self.image_file_name)
        #self._image.setPixmap(pixmap)

        #self._image = self.scene.addPixmap(pixmap.scaled(self.width(), self.height(), aspectRatioMode = 1))
        #self.scene.addPixmap(self._image)
        #self.scene = QGraphicsScene()
        #self.setScene(self.scene)
        self._image = None
        self.updateViewer()
        self.setImageFileName(self.image_file_name)
        self.update()
        #self.setImageFileName(self.image_file_name)
        #self.viewChanged.emit()

        #self.updateViewer()

        self._isPanningEditor = False
    
    '''
    def paintEvent(self, event):
        """
        Adjust image size to current window and draw bounding box.
        Args:
            event: QPaintEvent object.

        Return:
            None
        """

        super().paintEvent(event)
        qp = QPainter(self)
        pen = QPen(Qt.black)
        pen.setWidth(1)
        qp.setPen(pen)
        qp.drawRect(QRect(self.begin, self.end))
    '''
    
    def pointInBox(self, pX, pY, rX, rY, rW, rH):
        """
            checks if the rectangle points are within the canvas
            return: bool
        """
        return (rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH))

    def activatingCorners(self, x, y):
        if self.resized_h != 0 and self.resized_w != 0:

            #  (rX1, rY1)               (rX2, rY2)
            #         TL(1)-----TM(2)----TR(3)
            #          |                    |
            #         LM(4)       hold    RM(5)
            #          |                    |
            #         BL(6)-----BM(7)----BR(8)
            #  (rX4, rY4)               (rX3, rY3)

            if self.pointInBox(x, y, self.resized_x - handle, self.resized_y - handle, handle * 2, handle * 2):
                self.markerPos = 1
                print("inside TL")
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
                return
            if self.pointInBox(x, y, self.resized_x + int(self.resized_w / 2) - handle, self.resized_y - handle,
                               handle * 2, handle * 2):
                self.markerPos = 2
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("inside TM")
                return
            if self.pointInBox(x, y, self.resized_x + self.resized_w - handle, self.resized_y - handle, handle * 2,
                               handle * 2):
                self.markerPos = 3
                self.setCursor(QtCore.Qt.SizeBDiagCursor)
                print("inside TR")
                return
            if self.pointInBox(x, y, self.resized_x - handle,
                               self.resized_y + self.resized_h - int(self.resized_h / 2) - handle, handle * 2,
                               handle * 2):
                self.markerPos = 4
                self.setCursor(QtCore.Qt.SizeHorCursor)
                print("inside LM")
                return
            if self.pointInBox(x, y, self.resized_x + self.resized_w - handle,
                               self.resized_y + self.resized_h - int(self.resized_h / 2) - handle,
                               handle * 2, handle * 2):
                self.markerPos = 5
                self.setCursor(QtCore.Qt.SizeHorCursor)
                print("inside RM")
                return
            if self.pointInBox(x, y, self.resized_x - handle, self.resized_y + self.resized_h - handle, handle * 2,
                               handle * 2):
                self.markerPos = 6
                self.setCursor(QtCore.Qt.SizeBDiagCursor)
                print("inside BL")
                return
            if self.pointInBox(x, y, self.resized_x + int(self.resized_w / 2) - handle,
                               self.resized_y + self.resized_h - handle, handle * 2, handle * 2):
                self.markerPos = 7
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("inside BM")
                return
            if self.pointInBox(x, y, self.resized_x + self.resized_w - handle, self.resized_y + self.resized_h - handle,
                               handle * 2, handle * 2):
                self.markerPos = 8
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
                print("inside BR")
                return

            # click inside the Box to activate hold ⚓
            # anchor will lock the pos
            if self.pointInBox(x, y, self.resized_x, self.resized_y, self.resized_w, self.resized_h):
                print("hold it and drag")
                self.setCursor(QtCore.Qt.OpenHandCursor)
                self.markerPos = 9
                self.anchor_x = x - self.resized_x
                self.anchor_w = self.resized_w - self.anchor_x
                self.anchor_y = y - self.resized_y
                self.anchor_h = self.resized_h - self.anchor_y
                return

    def mousePressEvent(self, event):
        """
        Start drawing the box.
        Args:
            event: QMouseEvent object.

        Return:
            None
        """
        self.zoomPanROIPress(event)

        if event.button() == Qt.MouseButton.MiddleButton:
            print("MIDDLE PRESSED")
            self._isPanningEditor = True
            return 

        if (event.modifiers() == Qt.ShiftModifier):
            return

        point = event.pos()
        #self.update()

        global rectangle_highlight
        global img_x, img_y, img_w, img_h
        multiplier = self.main_window.scaleFactor
        x = int((point.x() - img_x) * (1 / multiplier))
        y = int((point.y() - img_y) * (1 / multiplier))

        if event.button() == Qt.LeftButton:
            if not self.activate:

                self.resizingDone = False
                self.drag = False
                # do regular work
                self.right_click = False
                self.start_point = event.pos()
                self.begin = event.pos()
                self.end = event.pos()
                self.update()

            elif self.activate:
                # editing contour work
                self.activatingCorners(x, y)

        elif event.button() == Qt.RightButton:
            self.right_click = True
            if self.activate:
                if self.pointInBox(x, y, self.resized_x, self.resized_y, self.resized_w, self.resized_h):
                    self.disableHandlersAndEditing()
                    self.saveEditedContour()
                    self.show_annotations(self.row)
                    self.setCursor(QtCore.Qt.ArrowCursor)

            else:
                self.menu_right = QMenu()
                self.actionDelete = QAction()
                self.actionDelete.setText("Delete")
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.actionDelete.setIcon(icon1)
                self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                self.menu_right.addAction(self.actionDelete)

                # self.actionEdit = QAction()
                # self.actionEdit.setText("Edit")
                # icon2 = QtGui.QIcon()
                # icon2.addPixmap(QtGui.QPixmap("../Icons/New/edit_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                # self.actionEdit.setIcon(icon2)
                # self.actionEdit.triggered.connect(self.edit_selected_contour)
                # self.menu_right.addAction(self.actionEdit)

                rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                    self.labels_to_annotation('Delete')

                if rectangle_contours and rectangle:
                    for contour in rectangle_contours:
                        check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                        if check > -1:  # if the point is inside
                            # find contour coordinates in image label List
                            self.Id, self.label_id = self.main_window.find_matching_coordinates(contour, " 'Rectangle'")
                            # append the annotation id in delete_highlight
                            if self.Id > -1:
                                rectangle_highlight = rectangle_contours.index(contour)
                                edit_highlight.append(self.Id)
                            self.menu_right.popup(QtGui.QCursor.pos())  # show the menu

            self.show_annotations(self.row)
        rectangle_highlight = -1

    def mouseMoveEvent(self, event):
        """
        Update size with mouse move.
        Args:
            event: QMouseEvent object.

        Return:
            None
        """
        self.zoomPanROIMove(event)



        if (event.modifiers() == Qt.ShiftModifier):
            return

        if self._isPanningEditor:
            print("MIDDLE MOVED")
            return

        if not self.right_click and not self.activate:
            self.end = event.pos()
            #self.update()

        elif self.activate:
            self.resize_it(event)

    def mouseReleaseEvent(self, event):
        """
        Calculate coordinates of the bounding box, display a message, update session data.
        Args:
            event: QMouseEvent object.

        Return:
            None
        """

        self.zoomPanROIRelease(event)



        if (event.modifiers() == Qt.ShiftModifier):
            return

        if self._isPanningEditor:
            print("MIDDLE RELEASED", event.button(), Qt.MouseButton.MiddleButton)
            self._isPanningEditor = False
            event.accept()
            return
        print("MIDDLE RELEASED 222", event.button(), Qt.MouseButton.MiddleButton)

        if event.button() == Qt.MouseButton.MiddleButton:
            return 

        if not self.right_click and not self.activate:
            self.begin = event.pos()
            self.end = event.pos()
            self.end_point = event.pos()
            x1, y1, x2, y2 = (
                self.start_point.x(),
                self.start_point.y(),
                self.end_point.x(),
                self.end_point.y(),
            )
            self.update()

            current_label_index = self.main_window.get_current_selection('selected_label')
            if current_label_index is None or current_label_index == 0:
                self.check_label_selection()
                return

            global img_x, img_y, img_w, img_h, offset_x, offset_y

            # Get the size of the widget
            widget_width = self.size().width()
            widget_height = self.size().height()

            # Get the position of the image
            image_x = img_x  # x-coordinate of top-left corner of image in widget coordinates
            image_y = img_y  # y-coordinate of top-left corner of image in widget coordinates

            # Get the size of the image
            image_height, image_width, _ = self.current_image.shape

            # Calculate the scaling factors for the image coordinates
            scale_x = image_width / widget_width
            scale_y = image_height / widget_height

            # Calculate the coordinates of the rectangle in the image coordinates
            widget_x1 = x1  # x-coordinate of top-left corner of rectangle in widget coordinates
            widget_y1 = y1  # y-coordinate of top-left corner of rectangle in widget coordinates
            widget_x2 = x2  # x-coordinate of bottom-right corner of rectangle in widget coordinates
            widget_y2 = y2  # y-coordinate of bottom-right corner of rectangle in widget coordinates

            offset_start = self.mapFromGlobal(self.start_point) - self.rect().topLeft()
            offset_end = self.mapFromGlobal(self.end_point) - self.rect().topLeft()

            img_x1 = int((offset_start.x()) * scale_x)#-int(x1/img_w)
            img_y1 = int((offset_start.y()) * scale_y)#-int(y1/img_h)
            img_x2 = int((offset_end.x()) * scale_x)#-int(x2/img_w)
            img_y2 = int((offset_end.y()) * scale_y)#-int(y2/img_h)
            print(x1,x1,y1,y2)
            print(img_x1,img_x2,img_y1,img_y2)
            print(image_x, image_y)
            print(image_height,image_width,widget_width,widget_height)
            print("SCALEEEE",scale_x,scale_y,self.end_point,self.mapFromGlobal(self.end_point))
            #self.update_session_data(il_x, il_y, i_bw, i_bh)

            #print("......",img_x,img_y,offset_x,offset_y)
            #print(x1, y1, x2-x1, y2-y1)
            #print(img_w, img_h, win_width, win_ht)
            #print(win_width-img_w, win_ht-img_h)
            self.update_session_data(img_x1, img_y1, img_x2-img_x1, img_y2-img_y1)
            self.show_annotations(self.row)

    def saveEditedContour(self):
        global rectangle_highlight
        self.main_window.session_data.sort_values(by=['ID', 'Image'], inplace=True)
        for i in range(self.main_window.right_widgets['Image Label List'].count()):
            session_name = str((self.main_window.right_widgets['Image Label List'].item(i).text()))
            obj_name = session_name.split(',')
            ann_id = str(obj_name[0])

            ann_id = ann_id.replace("'", "")
            ann_id = ann_id.replace(" ", "")
            ann_id = ann_id.replace("[", "")
            ann_id = int(ann_id)
            if ann_id == self.Id:
                print("found")
                # modifying in the Image Label List
                frame = self.main_window.images[self.row]
                label_name = self.main_window.right_widgets['Labels'].item(self.label_id).text()
                new_coords = [self.resized_x, self.resized_y, self.resized_w, self.resized_h]

                new_data = ([[self.Id, frame, label_name, self.label_id, "Rectangle", [new_coords]]])
                self.main_window.right_widgets['Image Label List'].item(i).setText(str(new_data))

                # modifying session_data
                id = self.Id - 1
                self.main_window.session_data.loc[id, 'Coordinates'][0][0] = self.resized_x
                self.main_window.session_data.loc[id, 'Coordinates'][0][1] = self.resized_y
                self.main_window.session_data.loc[id, 'Coordinates'][0][2] = self.resized_w
                self.main_window.session_data.loc[id, 'Coordinates'][0][3] = self.resized_h
                print(self.main_window.session_data)

                edit_highlight.clear()
                rectangle_highlight = -1
                selected_rectangle.clear()
                self.saveNewCoordinateMsg()

                # self.main_window.statusBar().showMessage("New coordinates saved")

    def saveNewCoordinateMsg(self):
        print("Saving New Coordinates")
        msg = QMessageBox()
        msg.setStyleSheet(
            "QLabel{min-height:20px; font-size: 12px;} QPushButton{ width:15px; font-size: 10px; }");
        msg.setIcon(QMessageBox.NoIcon)
        msg.setText("Saving new coordinates")
        msg.setWindowTitle("Save")
        msg.exec_()
        self.show_annotations(self.row)

    def RedrawRectangle(self):
        if not self.main_window.video_flag:
            frame = self.main_window.image_pixels[self.main_window.photo_row]
        else:
            frame = self.main_window.video_frame_pixels[self.main_window.photo_row]

        global handle
        color = [0, 0, 0]
        t = 2
        draw = frame.copy()
        cv2.rectangle(draw, (self.resized_x, self.resized_y),
                      (self.resized_x + self.resized_w, self.resized_y + self.resized_h), (0, 255, 0), 2)
        self.drawHandlers(draw, color, t)
        self.switch_frame(draw)

    def drawHandlers(self, draw, color, t):
        # Top-Left
        cv2.rectangle(draw, (self.resized_x - handle, self.resized_y - handle),
                      (self.resized_x + handle, self.resized_y + handle), color, t)
        # Top-Right
        cv2.rectangle(draw, (self.resized_x + self.resized_w - handle, self.resized_y - handle),
                      (self.resized_x + self.resized_w + handle, self.resized_y + handle), color, t)
        # Bottom-Left
        cv2.rectangle(draw, (self.resized_x - handle, self.resized_y + self.resized_h - handle),
                      (self.resized_x + handle, self.resized_y + self.resized_h + handle), color, t)
        # Bottom-Right
        cv2.rectangle(draw, (self.resized_x + self.resized_w - handle, self.resized_y + self.resized_h - handle),
                      (self.resized_x + self.resized_w + handle, self.resized_y + self.resized_h + handle), color, t)

        # Top-Mid
        cv2.rectangle(draw, (self.resized_x + int(self.resized_w / 2) - handle, self.resized_y - handle),
                      (self.resized_x + int(self.resized_w / 2) + handle, self.resized_y + handle), color, t)
        # Bottom-Mid
        cv2.rectangle(draw,
                      (self.resized_x + int(self.resized_w / 2) - handle, self.resized_y + self.resized_h - handle),
                      (self.resized_x + int(self.resized_w / 2) + handle, self.resized_y + self.resized_h + handle),
                      color, t)
        # Left-Mid
        cv2.rectangle(draw, (self.resized_x - handle, self.resized_y + int(self.resized_h / 2) - handle),
                      (self.resized_x + handle, self.resized_y + int(self.resized_h / 2) + handle), color, t)
        # Right-Mid
        cv2.rectangle(draw,
                      (self.resized_x + self.resized_w - handle, self.resized_y + int(self.resized_h / 2) - handle),
                      (self.resized_x + self.resized_w + handle, self.resized_y + int(self.resized_h / 2) + handle),
                      color, t)


    def resize_it(self, event):
        """
            Resizing Box using corners
        """
        point = event.pos()
        self.update()

        global img_x, img_y, img_w, img_h
        global win_width, win_ht

        multiplier = self.main_window.scaleFactor
        # get current click position
        eX = int((point.x() - img_x) * (1 / multiplier))
        eY = int((point.y() - img_y) * (1 / multiplier))

        if self.markerPos < 10:
            self.setCursor(QtCore.Qt.CrossCursor)

        # if self.hold:
        if self.markerPos == 9:
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.resized_x = eX - self.anchor_x
            self.resized_y = eY - self.anchor_y
            if self.resized_x < img_x:
                self.resized_x = img_x
            if self.resized_y < img_y:
                self.resized_y = img_y

            # if (self.resized_x + self.resized_w) > (img_x + win_width - 1):
            #     self.resized_x = img_x + win_width - 1 - self.resized_w
            # if (self.resized_y + self.resized_h) > (img_y + win_ht - 1):
            #     self.resized_y = img_y + win_ht - 1 - self.resized_h

            self.RedrawRectangle()
            return

        # if self.TL
        if self.markerPos == 1:
            self.resized_w = self.resized_x + self.resized_w - eX
            self.resized_h = self.resized_y + self.resized_h - eY
            self.resized_x = eX
            self.resized_y = eY
            self.RedrawRectangle()
            return
        # if self.BR:
        if self.markerPos == 8:
            self.resized_w = eX - self.resized_x
            self.resized_h = eY - self.resized_y
            self.RedrawRectangle()
            return
        # if self.TR:
        if self.markerPos == 3:
            self.resized_h = (self.resized_y + self.resized_h) - eY
            self.resized_y = eY
            self.resized_w = eX - self.resized_x
            self.RedrawRectangle()
            return
        # if self.BL:
        if self.markerPos == 6:
            self.resized_w = (self.resized_x + self.resized_w) - eX
            self.resized_x = eX
            self.resized_h = eY - self.resized_y
            self.RedrawRectangle()
            return

        # if self.TM:
        if self.markerPos == 2:
            self.resized_h = (self.resized_y + self.resized_h) - eY
            self.resized_y = eY
            self.RedrawRectangle()
            return
        # if self.BM:
        if self.markerPos == 7:
            self.resized_h = eY - self.resized_y
            self.RedrawRectangle()
            return
        # if self.LM:
        if self.markerPos == 4:
            self.resized_w = (self.resized_x + self.resized_w) - eX
            self.resized_x = eX
            self.RedrawRectangle()
            return
        # if self.RM:
        if self.markerPos == 5:
            self.resized_w = eX - self.resized_x
            self.RedrawRectangle()
            return

    def disableHandlersAndEditing(self):
        self.resizingDone = True
        self.activate = False
        self.drag = False
        self.markerPos = 0

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        self.update()

        global rectangle_highlight
        global img_x, img_y, img_w, img_h
        multiplier = self.main_window.scaleFactor
        # to get current click position
        x = int((point.x() - img_x) * (1 / multiplier))
        y = int((point.y() - img_y) * (1 / multiplier))

        if not self.activate and not self.resizingDone:
            print("mouse Double clicked")
            rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                self.labels_to_annotation('Delete')
            if len(rectangle_contours) > 0 and rectangle:
                for contour in rectangle_contours:
                    check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                    if check > -1:
                        # can get more info for rectangle resize and put back again
                        self.Id, self.label_id = self.main_window.find_matching_coordinates(contour, " 'Rectangle'")
                        if self.Id > -1:
                            rectangle_highlight = rectangle_contours.index(contour)
                            edit_highlight.append(self.Id)
                self.activate = True
                self.drag = True

                # assigning value to resizable contour
                if len(selected_rectangle) > 0 and self.activate:
                    it_l, it_r, ib_r, ib_l = selected_rectangle
                    rX1, rY1 = it_l[0], it_l[1]
                    rX2, rY2 = it_r[0], it_r[1]
                    rX3, rY3 = ib_r[0], ib_r[1]
                    rX4, rY4 = ib_l[0], ib_l[1]
                    # w, h = abs(rX1 - rX2), abs(rY4 - rY1)
                    self.resized_h = abs(rY4 - rY1)
                    self.resized_w = abs(rX1 - rX2)
                    self.resized_x = rX1
                    self.resized_y = rY1
                else:
                    pass

            self.show_annotations(self.row)

    def update_session_data(self, x, y, w, h):
        """
        Add a row to session_data containing calculated ratios.
        Args:
            x: Start x coordinate.
            y: Start y coordinate.
            w: x direction width.
            h: y direction height.

        Return:
            None
        """
        global current_session 

        current_label_index = self.main_window.get_current_selection('selected_label')  # selected label
        # if no label return
        if current_label_index is None or current_label_index < 0:
            return

        label_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        frame_name = self.main_window.images[self.row]
        if w != 0 and h != 0 and not self.right_click:
            rectangle_coordinates = [x, y, w, h]

            self.main_window.annotation_id = current_session
            a_id = self.main_window.annotation_id
            data = [[a_id, frame_name, label_name, current_label_index, "Rectangle", [rectangle_coordinates]]]

            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            current_session += 1
            print('yyyyyyyyyyyyyyyyy', rectangle_coordinates)
            print(self.main_window.session_data)
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])


class CubeEditorArea(ImageEditorArea):
    """
        making 3D Bounding Box on the 2D image
    """

    def __init__(self, current_image, main_window):
        super().__init__(current_image, main_window)

        """ Resizing related variables and flags"""
        self.cubeActive = False
        self.cubeResizingDone = False
        self.cubeMarkerPos = 0
        self.cubeResizedPos = [0] * 8  # [p1,p2,p3,p4, p1,p2,p3,p4] pi=(x,y)

        """ Anchor """
        # front face
        self.cubeAnchorPos = [0] * 5  # [x,y,w,h,d]
        self.cubeRareAnchorPos = [0] * 5  # [x,y,w,h,d]

        # selected cube info
        self.cubeID = None
        self.cubeLabelID = None

        # cube orientation info
        self.cubeShape = 0  # [1, 2, 3, 4]

    def pointOnCube(self, x, y, cX, cY, cW, cH):
        return (cX <= x <= (cX + cW) and cY <= y <= (cY + cH))

    def activateCubeHandlers(self, x, y):
        """
            Activates the handlers based on the clicked coordinates passed as x, y
        """
        # cubeResizedPos = f_tl, f_tr, f_br, f_bl, b_tl, b_tr, b_br, b_bl

        #      (x5,y5)          (x6, y6)
        #       10____________12
        #        |\____________\
        #        |(x1,y1)    (x2,y2)
        #   [16] | |    (9)     |
        # (x8,y8)| |[14](x7,y7) |
        #         \|___________\|
        #       (x4, y4)       (x3, y3)

        # empty
        if not self.cubeResizedPos:
            self.cubeActive = False
            return
        point = self.cubeResizedPos
        x1, y1 = point[0][0], point[0][1]  # 1
        x2, y2 = point[1][0], point[1][1]  # 3
        x3, y3 = point[2][0], point[2][1]  # 5
        x4, y4 = point[3][0], point[3][1]  # 7
        x5, y5 = point[4][0], point[4][1]  # 10
        x6, y6 = point[5][0], point[5][1]  # 12
        x7, y7 = point[6][0], point[6][1]  # 14
        x8, y8 = point[7][0], point[7][1]  # 15

        w, h = abs(x2 - x1), abs(y2 - y3)
        self.cubeRareOrientation(x, y)
        if w != 0 and h != 0:
            # top-left
            if self.pointOnCube(x, y, x1 - handle, y1 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 1
                print("top-left active")
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
                return
            # top-right
            if self.pointOnCube(x, y, x2 - handle, y2 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 3
                print("top-right active")
                self.setCursor(QtCore.Qt.SizeBDiagCursor)
                return
            # bottom-right
            if self.pointOnCube(x, y, x3 - handle, y3 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 5
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
                print("bottom-right active")
                return
            # bottom-left
            if self.pointOnCube(x, y, x4 - handle, y4 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 7
                self.setCursor(QtCore.Qt.SizeBDiagCursor)
                print("bottom-left active")
                return
            # top-mid
            if self.pointOnCube(x, y, x1 + int(w / 2) - handle, y1 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 2
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("top-mid active")
                return
            # bottom-mid
            if self.pointOnCube(x, y, x4 + int(w / 2) - handle, y4 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 6
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("Bottom-mid active")
                return
            # left-mid
            if self.pointOnCube(x, y, x1 - handle, y1 + int(h / 2) - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 8
                self.setCursor(QtCore.Qt.SizeHorCursor)
                print("left-mid active")
                return
            # right-mid
            if self.pointOnCube(x, y, x2 - handle, y2 + int(h / 2) - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 4
                self.setCursor(QtCore.Qt.SizeHorCursor)
                print("right-mid active")
                return
            # click inside the Box to activate hold ⚓
            if self.pointOnCube(x, y, x1, y1, w, h):
                print("hold cube & Drag")
                width = x2 - x1
                height = y4 - y1
                self.setCursor(QtCore.Qt.OpenHandCursor)
                self.cubeMarkerPos = 9
                self.cubeAnchorPos[0] = x - self.cubeResizedPos[0][0]
                self.cubeAnchorPos[1] = y - self.cubeResizedPos[0][1]
                self.cubeAnchorPos[2] = width - self.cubeAnchorPos[0]
                self.cubeAnchorPos[3] = height - self.cubeAnchorPos[1]
                return

            """Rare Face"""
            if self.pointOnCube(x, y, x8 - handle, y8 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 16
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("rare bottom left active")
                return
            if self.pointOnCube(x, y, x5 - handle, y5 - handle, handle * 2, handle * 2):
                self.cubeMarkerPos = 10
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("rare top left active")
                return

            if self.pointOnCube(x, y, x6-handle, y6-handle, handle*2, handle*2):
                self.cubeMarkerPos = 12
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("rare top right active")
                return

            if self.pointOnCube(x, y, x7-handle, y7-handle, handle*2, handle*2):
                self.cubeMarkerPos = 14
                self.setCursor(QtCore.Qt.SizeVerCursor)
                print("rare bottom right active")
                return

            """ holding and dragging the rare face"""
            self.checkPointOnCubeContour(x, y)

    def cubeRareOrientation(self, x, y):
        # empty
        if not self.cubeResizedPos:
            self.cubeActive = False
            return
        point = self.cubeResizedPos
        x1, y1 = point[0][0], point[0][1]
        x2, y2 = point[4][0], point[4][1]

        """cubeShape signifies the shape of cube shown
              ______________                        ______________
            +| \____________\                      /____________/ |+                             ____________                         ____________
            +| |            |                      |            | |+                          +/ |            |                      |            |\+
            +| |     [1]    |                      |    [2]     | |+                          +| |    [3]     |                      |    [4]     | |+
            +| |            |                      |            | |+                          +| |            |                      |            | |+
              \|____________|                      |____________|/                            +| |____________|                      |____________| |+
             + back face                                                                      +|/____________/                        \____________\|+
        """

        if x1 > x2:
            if y1 > y2:
                self.cubeShape = 1
            else:
                self.cubeShape = 3
        else:
            if y1 > y2:
                self.cubeShape = 2
            else:
                self.cubeShape = 4


    def checkPointOnCubeContour(self, x, y):
        if not self.cubeResizedPos:
            self.cubeActive = False
            return
        point = self.cubeResizedPos
        if self.cubeShape == 1:
            contour = [point[0], point[1], point[5], point[4], point[7], point[3]]
            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
            if check == 1:
                print("cube shape 1")
                self.cubeMarkerPos = 20
                self.setCubeRareAnchor(x, y)
                self.setCursor(QtCore.Qt.SizeAllCursor)
            return
        if self.cubeShape == 2:
            contour = [point[0], point[1], point[2], point[6], point[5], point[4]]
            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
            if check == 1:
                print("cube shape 2")
                self.cubeMarkerPos = 20
                self.setCubeRareAnchor(x, y)
                self.setCursor(QtCore.Qt.SizeAllCursor)
            return
        if self.cubeShape == 3:
            contour = [point[0], point[3], point[2], point[6], point[7], point[4]]
            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
            if check == 1:
                print("cube shape 3")
                self.cubeMarkerPos = 20
                self.setCubeRareAnchor(x, y)
                self.setCursor(QtCore.Qt.SizeAllCursor)
            return
        if self.cubeShape == 4:
            contour = [point[1], point[2], point[3], point[4], point[6], point[5]]
            check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
            if check == 1:
                print("cube shape 4")
                self.cubeMarkerPos = 20
                self.setCubeRareAnchor(x, y)
                self.setCursor(QtCore.Qt.SizeAllCursor)
            return

    def setCubeRareAnchor(self, x, y):
        x1, y1 = self.cubeResizedPos[4][0], self.cubeResizedPos[4][1]
        x2, y2 = self.cubeResizedPos[5][0], self.cubeResizedPos[5][1]
        x3, y3 = self.cubeResizedPos[6][0], self.cubeResizedPos[6][1]
        width = x2 - x1
        height = y3 - y1
        self.cubeRareAnchorPos[0] = x - self.cubeResizedPos[4][0]
        self.cubeRareAnchorPos[1] = y - self.cubeResizedPos[4][1]
        self.cubeRareAnchorPos[2] = width - self.cubeRareAnchorPos[0]
        self.cubeRareAnchorPos[3] = height - self.cubeRareAnchorPos[1]
        return

    def disableCubeEditing(self):
        self.cubeActive = False
        self.cubeMarkerPos = 0
        self.cubeResizingDone = True
        self.cubeShape = 0
        self.cubeID = None

    def mousePressEvent(self, event):
        point = event.pos()
        self.update()
        global cube_highlight
        global img_x, img_y, img_w, img_h
        multiplier = self.main_window.scaleFactor
        x = int((point.x() - img_x) * (1 / multiplier))
        y = int((point.y() - img_y) * (1 / multiplier))

        if event.button() == Qt.LeftButton:
            if not self.cubeActive:
                self.cubeResizingDone = False
                self.cubeMarkerPos = 0
                self.right_click = False
                self.start_point = event.pos()
                self.begin = event.pos()
                self.end = event.pos()
            elif self.cubeActive:
                self.activateCubeHandlers(x, y)
            else:
                pass

        elif event.button() == Qt.RightButton:
            self.right_click = True
            if self.cubeActive:
                # e = self.cubeResizedPos
                # # front-face of the cube
                # if self.pointOnCube(x, y, e[0][0], e[0][1], e[2][0], e[2][1]):
                self.saveEditedCube()
                self.disableCubeEditing()
                self.show_annotations(self.row)
                self.setCursor(QtCore.Qt.ArrowCursor)

            else:
                self.menu_right = QMenu()
                self.actionDelete = QAction()
                self.actionDelete.setText("Delete")
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.actionDelete.setIcon(icon1)
                self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
                self.menu_right.addAction(self.actionDelete)

                rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                    self.labels_to_annotation('Delete')

                if cube_contours and cube:
                    for contour in cube_contours:
                        check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                        if check > -1:
                            self.cubeID, self.cubeLabelID = \
                                self.main_window.find_matching_coordinates(contour, " 'Cube'")
                            if self.cubeID > -1:
                                cube_highlight = cube_contours.index(contour)
                                edit_highlight.append(self.cubeID)
                            self.menu_right.popup(QtGui.QCursor.pos())  # show the menu
            self.show_annotations(self.row)
        cube_highlight = -1

    def cube_resize(self, event):
        """
            Resizing and reorienting the Cube
        """
        point = event.pos()
        self.update()

        global img_x, img_y, img_w, img_h
        global win_width, win_ht

        multiplier = self.main_window.scaleFactor
        # get current click position
        eX = int((point.x() - img_x) * (1 / multiplier))
        eY = int((point.y() - img_y) * (1 / multiplier))

        point = self.cubeResizedPos
        width = point[1][0] - point[0][0]
        height = point[3][1] - point[0][1]
        dx, dy = point[4][0] - point[0][0],  point[4][1] - point[0][1]

        """hold"""
        if self.cubeMarkerPos == 9:
            self.cubeResizedPos[0] = (eX - self.cubeAnchorPos[0], eY - self.cubeAnchorPos[1])
            self.cubeResizedPos[4] = (self.cubeResizedPos[0][0] + dx, self.cubeResizedPos[0][1] + dy)

            self.cubeResizedPos[1] = (self.cubeResizedPos[0][0] + width, self.cubeResizedPos[0][1])
            self.cubeResizedPos[5] = (self.cubeResizedPos[1][0] + dx, self.cubeResizedPos[1][1] + dy)

            self.cubeResizedPos[2] = (self.cubeResizedPos[0][0] + width, self.cubeResizedPos[0][1] + height)
            self.cubeResizedPos[6] = (self.cubeResizedPos[2][0] + dx, self.cubeResizedPos[2][1] + dy)

            self.cubeResizedPos[3] = (self.cubeResizedPos[0][0], self.cubeResizedPos[0][1] + height)
            self.cubeResizedPos[7] = (self.cubeResizedPos[3][0] + dx, self.cubeResizedPos[3][1] + dy)
            self.redrawCube()

        """top left"""
        if self.cubeMarkerPos == 1:
            # front \n rare
            if eY > point[3][1] or eX > point[2][0]:
                return
            point[0] = (eX, eY)
            point[4] = (eX + dx, eY + dy)
            point[1] = (point[1][0], eY)
            point[5] = (point[5][0], eY + dy)
            point[3] = (eX, point[3][1])
            point[7] = (eX + dx, point[7][1])
            self.redrawCube()
        """top mid"""
        if self.cubeMarkerPos == 2:
            if eY > point[3][1]:
                return
            point[0] = (point[0][0], eY)
            point[4] = (point[4][0], eY + dy)
            point[1] = (point[1][0], eY)
            point[5] = (point[5][0], eY + dy)
            self.redrawCube()
        """top right"""
        if self.cubeMarkerPos == 3:
            if eY > point[3][1] or eX < point[0][0]:
                return
            point[1] = (eX, eY)
            point[5] = (eX + dx, eY + dy)
            point[0] = (point[0][0], eY)
            point[4] = (point[4][0], eY + dy)
            point[2] = (eX, point[2][1])
            point[6] = (eX + dx, point[6][1])
            self.redrawCube()
        """right mid"""
        if self.cubeMarkerPos == 4:
            if eX < point[0][0]:
                return
            point[1] = (eX, point[1][1])
            point[5] = (eX + dx, point[5][1])
            point[2] = (eX, point[2][1])
            point[6] = (eX + dx, point[6][1])
            self.redrawCube()
        """right bottom"""
        if self.cubeMarkerPos == 5:
            if eX < point[3][0] or eY < point[1][1]:
                return
            point[2] = (eX, eY)
            point[6] = (eX + dx, eY + dy)
            point[1] = (eX, point[1][1])
            point[5] = (eX + dx, point[5][1])
            point[3] = (point[3][0], eY)
            point[7] = (point[7][0], eY + dy)
            self.redrawCube()
        """bottom mid"""
        if self.cubeMarkerPos == 6:
            if eY < point[1][1]:
                return
            point[2] = (point[2][0], eY)
            point[6] = (point[6][0], eY + dy)
            point[3] = (point[3][0], eY)
            point[7] = (point[7][0], eY + dy)
            self.redrawCube()
        """bottom left"""
        if self.cubeMarkerPos == 7:
            if eX > point[2][0] or eY < point[0][1]:
                return
            point[3] = (eX, eY)
            point[7] = (eX + dx, eY + dy)
            point[0] = (eX, point[0][1])
            point[4] = (eX + dx, point[4][1])
            point[2] = (point[2][0], eY)
            point[6] = (point[6][0], eY + dy)
            self.redrawCube()
        """left mid"""
        if self.cubeMarkerPos == 8:
            if eX > point[2][0]:
                return
            point[0] = (eX, point[0][1])
            point[4] = (eX + dx, point[4][1])
            point[3] = (eX, point[3][1])
            point[7] = (eX + dx, point[7][1])
            self.redrawCube()

        """ *** Rare faces *** """
        """top left corner"""
        if self.cubeMarkerPos == 10 and self.cubeShape % 2 == 1:  # cubeShape either 1 or 3
            # only Y-direction movement
            if eY > point[7][1]:
                return
            point[4] = (point[4][0], eY)
            point[0] = (point[0][0], eY - dy)
            point[5] = (point[5][0], eY)
            point[1] = (point[1][0], eY - dy)
            self.redrawCube()

        """bottom left corner"""
        if self.cubeMarkerPos == 16 and self.cubeShape % 2 == 1:  # cubeShape either 1 or 3
            # only Y-direction movement
            if eY < point[4][1]:
                return
            point[7] = (point[7][0], eY)
            point[3] = (point[3][0], eY - dy)
            point[6] = (point[6][0], eY)
            point[2] = (point[2][0], eY - dy)
            self.redrawCube()

        """top right corner"""
        if self.cubeMarkerPos == 12 and self.cubeShape % 2 == 0:  # cubeShape either 2 or 4
            if eY > point[7][1]:
                return
            point[5] = (point[5][0], eY)
            point[1] = (point[1][0], eY - dy)
            point[4] = (point[4][0], eY)
            point[0] = (point[0][0], eY - dy)
            self.redrawCube()

        """bottom right corner"""
        if self.cubeMarkerPos == 14 and self.cubeShape % 2 == 0:  # cubeShape either 2 or 4
            if eY < point[4][1]:
                return
            point[6] = (point[6][0], eY)
            point[2] = (point[2][0], eY-dy)
            point[7] = (point[7][0], eY)
            point[3] = (point[3][0], eY-dy)
            self.redrawCube()
            pass

        """ hold and drag rare face"""
        if self.cubeMarkerPos == 20:
            self.cubeResizedPos[4] = (eX - self.cubeRareAnchorPos[0], eY - self.cubeRareAnchorPos[1])
            self.cubeResizedPos[5] = (self.cubeResizedPos[4][0] + width, self.cubeResizedPos[4][1])
            self.cubeResizedPos[6] = (self.cubeResizedPos[4][0] + width, self.cubeResizedPos[4][1] + height)
            self.cubeResizedPos[7] = (self.cubeResizedPos[4][0], self.cubeResizedPos[4][1] + height)
            self.redrawCube()

    def redrawCube(self):
        if not self.main_window.video_flag:
            frame = self.main_window.image_pixels[self.main_window.photo_row]
        else:
            frame = self.main_window.video_frame_pixels[self.main_window.photo_row]

        global handle
        drawOn = frame.copy()
        # structure
        self.cubeFacesNEdges(drawOn)
        self.switch_frame(drawOn)

    def cubeFacesNEdges(self, drawOn):
        bg_color = (0, 0, 0)
        handle_color = fr_color = (255, 255, 255)
        t = 2
        for i in range(4):
            """front face"""
            cv2.line(drawOn, self.cubeResizedPos[i % 4], self.cubeResizedPos[(i + 1) % 4], fr_color, t)
            """rare face"""
            cv2.line(drawOn, self.cubeResizedPos[(i % 4) + 4], self.cubeResizedPos[((i + 1) % 4) + 4], bg_color, t)
            """edges"""
            cv2.line(drawOn, self.cubeResizedPos[i % 4], self.cubeResizedPos[(i % 4) + 4], bg_color, t)

        """handlers"""
        self.drawCubeNewHandlers(drawOn, handle_color, -t)

    def drawCubeNewHandlers(self, drawOn, handle_color, t):
        # top-left
        point = self.cubeResizedPos
        cv2.rectangle(drawOn, (point[0][0] - handle, point[0][1] - handle),
                      (point[0][0] + handle, point[0][1] + handle), handle_color, t)
        # top-right
        cv2.rectangle(drawOn, (point[1][0] - handle, point[1][1] - handle),
                      (point[1][0] + handle, point[1][1] + handle), handle_color, t)
        # bottom-right
        cv2.rectangle(drawOn, (point[2][0] - handle, point[2][1] - handle),
                      (point[2][0] + handle, point[2][1] + handle), handle_color, t)
        # bottom-left
        cv2.rectangle(drawOn, (point[3][0] - handle, point[3][1] - handle),
                      (point[3][0] + handle, point[3][1] + handle), handle_color, t)

        h = int(point[3][1] - point[0][1])
        w = int(point[1][0] - point[0][0])
        edge_handle = (255, 255, 0)
        # top-mid
        cv2.rectangle(drawOn, (point[0][0] + int(w / 2) - handle, point[0][1] - handle),
                      (point[0][0] + int(w / 2) + handle, point[0][1] + handle),
                      edge_handle, t)
        # left-mid
        cv2.rectangle(drawOn, (point[0][0] - handle, point[0][1] + int(h / 2) - handle),
                      (point[0][0] + handle, point[0][1] + int(h / 2) + handle),
                      edge_handle, t)
        # right-mid
        cv2.rectangle(drawOn, (point[1][0] - handle, point[1][1] + int(h / 2) - handle),
                      (point[1][0] + handle, point[1][1] + int(h / 2) + handle),
                      edge_handle, t)
        # bottom-mid
        cv2.rectangle(drawOn, (point[3][0] + int(w / 2) - handle, point[3][1] - handle),
                      (point[3][0] + int(w / 2) + handle, point[3][1] + handle),
                      edge_handle, t)

        """ rare face handlers"""
        if self.cubeShape % 2 == 0:
            cv2.rectangle(drawOn, (point[5][0] - handle, point[5][1] - handle),
                          (point[5][0] + handle, point[5][1] + handle), handle_color, t)
            cv2.rectangle(drawOn, (point[6][0] - handle, point[6][1] - handle),
                          (point[6][0] + handle, point[6][1] + handle), handle_color, t)
        elif self.cubeShape % 2 == 1:
            cv2.rectangle(drawOn, (point[4][0] - handle, point[4][1] - handle),
                          (point[4][0] + handle, point[4][1] + handle), handle_color, t)
            cv2.rectangle(drawOn, (point[7][0] - handle, point[7][1] - handle),
                          (point[7][0] + handle, point[7][1] + handle), handle_color, t)

    def mouseMoveEvent(self, event):
        if not self.right_click and not self.cubeActive:
            self.end = event.pos()
            self.update()

        elif self.cubeActive:
            self.cube_resize(event)

    def mouseReleaseEvent(self, event):
        global current_session
        if not self.right_click and not self.cubeActive:
            self.begin = event.pos()
            self.end = event.pos()
            self.end_point = event.pos()
            x1, y1, x2, y2 = (
                self.start_point.x(),
                self.start_point.y(),
                self.end_point.x(),
                self.end_point.y(),
            )
            self.update()

            current_label_index = self.main_window.get_current_selection('selected_label')
            if current_label_index is None or current_label_index == 0:
                self.check_label_selection()
                return

            global img_x, img_y, img_w, img_h

            # In window co-ordinates (top left and bottom right)
            wl_x, wl_y = min(x2, x1), min(y2, y1)
            wr_x, wr_y = max(x2, x1), max(y2, y1)

            # In scaled image co-ordinates
            sl_x, sl_y = (wl_x - img_x), (wl_y - img_y)
            sr_x, sr_y = (wr_x - img_x), (wr_y - img_y)

            multiplier = self.main_window.scaleFactor
            # In original image co-ordinates
            il_x = int(sl_x * (1 / multiplier))
            il_y = int(sl_y * (1 / multiplier))
            ir_x = int(sr_x * (1 / multiplier))
            ir_y = int(sr_y * (1 / multiplier))

            s_bw = abs(x2 - x1)  # BB width and height for scaled image
            s_bh = abs(y2 - y1)
            i_bw = int(s_bw * (1 / multiplier))  # BB width and height for original image
            i_bh = int(s_bh * (1 / multiplier))
            depth = 20
            """rare face coordinate"""
            rl_x, r_ly = il_x + depth, il_y - depth
            self.update_session_data(il_x, il_y, rl_x, r_ly, i_bw, i_bh)
            print("row",self.row)
            self.show_annotations(self.row)

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        self.update()

        global cube_highlight
        global img_x, img_y, img_w, img_h
        multiplier = self.main_window.scaleFactor
        # to get current click position
        x = int((point.x() - img_x) * (1 / multiplier))
        y = int((point.y() - img_y) * (1 / multiplier))

        print("cubeActive", self.cubeActive)
        print("cubeResizingDone", self.cubeResizingDone)

        if not self.cubeActive and not self.cubeResizingDone:
            print("mouse Double clicked")
            rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                self.labels_to_annotation('Delete')
            if len(cube_contours) > 0 and cube:
                for contour in cube_contours:
                    check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                    if check > -1:
                        self.cubeID, self.cubeLabelID = self.main_window.find_matching_coordinates(contour, " 'Cube'")
                        if self.cubeID > 0:
                            cube_highlight = cube_contours.index(contour)
                            edit_highlight.append(self.cubeID)
                self.cubeActive = True

                # assigning value to resizable contour
                if len(selected_cube) > 0 and self.cubeActive:
                    # f_tl, f_tr, f_br, f_bl, b_tl, b_tr, b_br, b_bl
                    self.cubeResizedPos = selected_cube
            self.show_annotations(self.row)

    def saveEditedCube(self):
        global selected_cube, cube_highlight
        self.main_window.session_data.sort_values(by=['ID', 'Image'], inplace=True)
        for i in range(self.main_window.right_widgets['Image Label List'].count()):
            session_name = str((self.main_window.right_widgets['Image Label List'].item(i).text()))
            obj_name = session_name.split(',')
            ann_id = str(obj_name[0])

            ann_id = ann_id.replace("'", "")
            ann_id = ann_id.replace(" ", "")
            ann_id = ann_id.replace("[", "")
            ann_id = int(ann_id)
            if ann_id == self.cubeID:
                print("found")
                frame = self.main_window.images[self.row]
                label_name = self.main_window.right_widgets['Labels'].item(self.cubeLabelID).text()
                x1 = self.cubeResizedPos[0][0]
                y1 = self.cubeResizedPos[0][1]
                x2 = self.cubeResizedPos[4][0]
                y2 = self.cubeResizedPos[4][1]
                w = self.cubeResizedPos[1][0] - self.cubeResizedPos[0][0]
                h = self.cubeResizedPos[3][1] - self.cubeResizedPos[1][1]
                print("cube resized", x1, y1, x2, y2, w, h)
                new_coords = [x1, y1, x2, y2, w, h]

                new_data = ([[self.cubeID, frame, label_name, self.cubeLabelID, "Cube", [new_coords]]])
                self.main_window.right_widgets['Image Label List'].item(i).setText(str(new_data))

                # modifying session_data
                id = self.cubeID - 1
                print("save to id", self.cubeID, id)
                self.main_window.session_data.loc[id, 'Coordinates'][0][0] = x1
                self.main_window.session_data.loc[id, 'Coordinates'][0][1] = y1
                self.main_window.session_data.loc[id, 'Coordinates'][0][2] = x2
                self.main_window.session_data.loc[id, 'Coordinates'][0][3] = y2
                self.main_window.session_data.loc[id, 'Coordinates'][0][4] = w
                self.main_window.session_data.loc[id, 'Coordinates'][0][5] = h
                print(self.main_window.session_data)

                edit_highlight.clear()
                cube_highlight = -1
                selected_cube.clear()
                self.disableCubeEditing()
                self.saveNewCoordinateMsg()

    def update_session_data(self, x1, y1, x2, y2, w, h):
        global current_session
        current_label_index = self.main_window.get_current_selection('selected_label')
        # if no label return
        if current_label_index is None or current_label_index == 0:
            self.check_label_selection()
            return

        label_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        frame_name = self.main_window.images[self.row]
        if w != 0 and h != 0 and not self.right_click:
            cube_coordinates = [x1, y1, x2, y2, w, h]

            self.main_window.annotation_id = current_session
            a_id = self.main_window.annotation_id
            data = [[a_id, frame_name, label_name, current_label_index, "Cube", [cube_coordinates]]]

            to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
            self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
            current_session += 1
            self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])


class PolygonImageEditor(RegularImageArea):
    """
    Edit and display area within the main interface.
    """

    def __init__(self, current_image, main_window):
        """
        Initialize current image for display.
        Args:
            current_image: Path to target image.
            main_window: ImageLabeler instance.
        """
        super().__init__(current_image, main_window)
        self.main_window = main_window
        global current_session
        self.main_window.session_data = self.main_window.session_data[:current_session]
        self.point = QPoint()
        # self.end_point = QPoint()
        # self.begin = QPoint()
        # self.end = QPoint()
        self.right_click = False
        self.pointlist = []
        self.setFocusPolicy(Qt.StrongFocus)
    '''
    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        pen = QPen(Qt.black)
        pen.setWidth(2)
        qp.setPen(pen)
        qp.drawEllipse(self.point, 2, 1)
        if len(self.pointlist) > 1:
            points = QPolygon(self.pointlist)
            qp.drawPolygon(points)
    '''
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.check_label_selection()
            self.point = QPoint()
            self.point = event.pos()
            self.pointlist.append(self.point)
            self.update()
            # print(self.point)

        elif event.button() == Qt.RightButton:
            self.right_click = True
            point = event.pos()
            self.update()

            global polygon_highlight
            global img_x, img_y, img_w, img_h
            multiplier = self.main_window.scaleFactor
            x = int((point.x() - img_x) * (1 / multiplier))
            y = int((point.y() - img_y) * (1 / multiplier))

            rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                self.labels_to_annotation('Delete')

            self.menu_right = QMenu()
            self.actionDelete = QAction()
            self.actionDelete.setText("Delete")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionDelete.setIcon(icon)
            self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
            self.menu_right.addAction(self.actionDelete)

            if polygon_contours and polygon:
                for contour in polygon_contours:
                    # print("polygon-contour", contour)
                    check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                    # print("dist", check)
                    if check > -1:  # if the point is inside
                        # find contour coordinates in image label List
                        ann_id = self.main_window.find_matching_coordinates(contour, " 'Polygon'")
                        # append the annotation id in highlight
                        if ann_id > -1:
                            print("poly id found")
                            edit_highlight.append(ann_id)
                            polygon_highlight = polygon_contours.index(contour)
                        print("in Polygon highlight", polygon_highlight)
                        print("contour", contour)
                        self.menu_right.popup(QtGui.QCursor.pos())

            self.show_annotations(self.main_window.photo_row)
            polygon_highlight = -1

        self.right_click = False

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    def keyPressEvent(self, event):
        if event.key() == (Qt.Key_Control and Qt.Key_Z):
            self.pointlist.clear()
            msg = QMessageBox()
            msg.setStyleSheet("QLabel{min-height:20px; font-size: 12px;} QPushButton{ width:15px; font-size: 10px; }");
            msg.setIcon(QMessageBox.NoIcon)
            msg.setText("Annotation Removed")
            msg.setWindowTitle("Polygon")
            msg.exec_()

        elif event.key() == Qt.Key_Escape:
            self.check_label_selection()
            if len(self.pointlist) < 3:
                return
            self.store_coords()
            self.pointlist.clear()

    def store_coords(self):
        global current_session
        self.row = self.main_window.photo_row
        current_label_index = self.main_window.get_current_selection('selected_label')
        object_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        if current_label_index is None or current_label_index < 0:
            return
        frame_name = self.main_window.images[self.row]

        self.main_window.annotation_id = current_session
        id = self.main_window.annotation_id
        finalarray = [id, frame_name, object_name, current_label_index]

        print("Storing Coordinates...")
        y = len(self.pointlist)
        z = 0
        coordarray = []
        while z < y:
            point = self.pointlist[z]
            x_coord = point.x()
            y_coord = point.y()
            x_coord, y_coord = self.processcoords(x_coord, y_coord)
            coordarray.append([x_coord, y_coord])
            z += 1

        finalarray.append("Polygon")
        finalarray.append(coordarray)
        data = [finalarray]
        print("Adding Coordinates...")
        to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
        self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
        current_session += 1
        self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])

        self.show_annotations(self.row)

    def processcoords(self, x_coord, y_coord):
        print("Processing Coordinates...")
        global img_x, img_y
        sl_x = (x_coord - img_x)
        sl_y = (y_coord - img_y)
        multiplier = self.main_window.scaleFactor
        for p in [sl_x, sl_y]:
            il_x = int(sl_x * (1 / multiplier))
            il_y = int(sl_y * (1 / multiplier))
        x_coord = il_x
        y_coord = il_y
        self.row = self.main_window.photo_row
        return x_coord, y_coord


class PolyLineImageEditor(RegularImageArea):
    """
    Edit and display area within the main interface.
    """

    def __init__(self, current_image, main_window):
        """
        Initialize current image for display.
        Args:
            current_image: Path to target image.
            main_window: ImageLabeler instance.
        """
        super().__init__(current_image, main_window)
        self.main_window = main_window
        global current_session
        self.main_window.session_data = self.main_window.session_data[:current_session]

        self.point = QPoint()
        self.right_click = False
        self.pointlist = []
        self.setFocusPolicy(Qt.StrongFocus)
    '''
    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        pen = QPen(Qt.black)
        pen.setWidth(2)
        qp.setPen(pen)
        qp.drawEllipse(self.point, 2, 1)
        if len(self.pointlist) > 1:
            points = QPolygon(self.pointlist)
            qp.drawPolygon(points)
    '''
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.check_label_selection()
            self.point = QPoint()
            self.point = event.pos()
            self.pointlist.append(self.point)
            self.update()
            # print(self.point)
        elif event.button() == Qt.RightButton:
            self.right_click = True
            point = event.pos()
            self.update()

            global img_x, img_y, img_w, img_h
            multiplier = self.main_window.scaleFactor
            x = int((point.x() - img_x) * (1 / multiplier))
            y = int((point.y() - img_y) * (1 / multiplier))

            rectangle, polygon, polyline, cube, rectangle_contours, polygon_contours, polyline_contours, cube_contours = \
                self.labels_to_annotation('Delete')

            self.menu_right = QMenu()
            self.actionDelete = QAction()
            self.actionDelete.setText("Delete")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("../Icons/New/delete_.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionDelete.setIcon(icon)
            self.actionDelete.triggered.connect(self.main_window.delete_selected_contour)
            self.menu_right.addAction(self.actionDelete)

            global polyline_highlight
            if polyline_contours and polyline:
                for contour in polyline_contours:
                    check = cv2.pointPolygonTest(np.array([contour]), (x, y), False)
                    print("dist", check)
                    print("x")
                    if check == 1:  # if the point is on the line
                        print("polyline detected")
                        # find contour coordinates in image label List
                        ann_id = self.main_window.find_matching_coordinates(contour, " 'Polyline'")
                        # append the annotation id in delete_highlight
                        if ann_id > -1:
                            polyline_highlight = polyline_contours.index(contour)
                            edit_highlight.append(ann_id)
                        print("line highlight", polyline_highlight)
                        print("contour", contour)
                        self.menu_right.popup(QtGui.QCursor.pos())  # delete menu shown

            self.show_annotations(self.main_window.photo_row)
            polyline_highlight = -1

    # overriding methods
    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    def keyPressEvent(self, event):
        if event.key() == (Qt.Key_Control and Qt.Key_Z):
            if len(self.pointlist) > 0:
                self.pointlist.pop()
                # show pop()
        elif event.key() == Qt.Key_Escape:
            self.check_label_selection()
            if len(self.pointlist) < 2:
                return
            self.store_coords()
            self.pointlist.clear()

    def store_coords(self):
        global current_session
        self.row = self.main_window.photo_row
        current_label_index = self.main_window.get_current_selection('selected_label')
        object_name = (self.main_window.right_widgets['Labels'].item(current_label_index).text())
        if current_label_index is None or current_label_index < 0:
            return
        frame_name = self.main_window.images[self.row]
        self.main_window.annotation_id = current_session
        id = self.main_window.annotation_id
        finalarray = [id, frame_name, object_name, current_label_index]

        print("Storing polyLine Coordinates...")
        y = len(self.pointlist)
        z = 0
        coordarray = []
        while z < y:
            point = self.pointlist[z]
            x_coord = point.x()
            y_coord = point.y()
            x_coord, y_coord = self.processcoords(x_coord, y_coord)
            coordarray.append([x_coord, y_coord])
            z += 1

        finalarray.append("Polyline")
        finalarray.append(coordarray)
        data = [finalarray]
        print("Adding Coordinates...")
        to_add = pd.DataFrame(data, columns=self.main_window.session_data.columns)
        self.main_window.session_data = pd.concat([self.main_window.session_data.iloc[:current_session], to_add], ignore_index=True)
        current_session += 1
        self.main_window.add_to_list(f'{data}', self.main_window.right_widgets['Image Label List'])

        self.show_annotations(self.row)

    def processcoords(self, x_coord, y_coord):
        print("Processing Coordinates...")
        global img_x, img_y
        sl_x = (x_coord - img_x)
        sl_y = (y_coord - img_y)
        multiplier = self.main_window.scaleFactor
        for p in [sl_x, sl_y]:
            il_x = int(sl_x * (1 / multiplier))
            il_y = int(sl_y * (1 / multiplier))
        x_coord = il_x
        y_coord = il_y
        self.row = self.main_window.photo_row
        return x_coord, y_coord

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon("../Icons/tata-elxsi-icon.jpg"))
        self.setWindowTitle("TESAAR")
        self.setFixedSize(400, 250)
        self.layout_ = QVBoxLayout(self)
        self.label = QLabel("<font size = 6>Open Project with</font>", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout_.addWidget(self.label)

        self.image1 = QLabel()
        self.image1.setPixmap(QPixmap("../Icons/New/new_task.png").scaledToWidth(128))
        self.image1.setAlignment(Qt.AlignCenter)
        self.image1.mousePressEvent = self.fresh_start
        self.label1 = QLabel("Fresh Start", self)
        self.label1.setFont(QFont('Arial', 12))
        self.label1.setAlignment(Qt.AlignCenter)

        self.layout1 = QVBoxLayout(self)
        self.layout1.addWidget(self.image1)
        self.layout1.addWidget(self.label1)

        self.image2 = QLabel(self)
        self.image2.setPixmap(QPixmap("../Icons/New/saved_task.png").scaledToWidth(128))
        self.image2.setAlignment(Qt.AlignCenter)
        self.image2.mousePressEvent = self.saved_start
        self.label2 = QLabel("Saved Work", self)
        self.label2.setFont(QFont('Arial', 12))
        self.label2.setAlignment(Qt.AlignCenter)

        self.layout2 = QVBoxLayout(self)
        self.layout2.addWidget(self.image2)
        self.layout2.addWidget(self.label2)

        self.hbox = QHBoxLayout()
        self.hbox.addLayout(self.layout1)
        self.hbox.addLayout(self.layout2)
        self.layout_.addLayout(self.hbox)
        # Create a widget and set the layout
        self.widget = QWidget()
        self.widget.setLayout(self.layout_)

        # Set the central widget of the main window
        self.setCentralWidget(self.widget)

    def readFrmCheckpointFile(self):
        if not os.path.exists('checkpoint/saved.yaml'):
            return
        with open("checkpoint/saved.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                print(data)
                if data is None:
                    print("the file is empty")
                    return
                isSaved = data['save']
                if isSaved:
                    global loadDataFromFile
                    loadDataFromFile = True
                else:
                    print("no saved file")
            except yaml.YAMLError as exc:
                print(exc)

    def fresh_start(self, event):
        ImageLabeler()
        self.close()

    def saved_start(self, event):
        self.readFrmCheckpointFile()
        ImageLabeler()
        self.close()

class ImageLabeler(QMainWindow):
    """
    Image labeling main interface.
    """

    def __init__(self, window_title="Tata Elxsi Smart AnonymizeR (TESAR)", current_image_area=RegularImageArea):
        """
        Initialize main interface and display.
        Args:
            window_title: Title of the window.
            current_image_area: RegularImageArea or ImageEditorArea object.
        """
        super().__init__()
        self.annotation_id = 0
        self.annotation_file_dir = None
        self.annotation_file_name = None
        self.zoomSliderVal = 100
        self.current_image = None
        self.main_dir = None
        self.current_image_area = current_image_area
        self.savearea = None
        self.images = []
        self.image_paths = {}
        self.video_flag = False
        self.photo_flag = False
        self.frame_dir = None
        self.photo_res = []
        self.image_pixels = []
        self.video_frame_pixels = []
        self.scaleFactor = 1.0
        self.frame = 0
        self.slidershift = 0
        self.frametimer = None
        self.frameplayer = False

        self.segmentation_flag = "manual"
        self.show_edit_window()

        # self.progresstracker = VideoBlurrer("1080p_medium_mosaic", None)
        # self.progresstracker.updateProgress.connect(self.progbar)

        # add annotation ID here_
    def show_edit_window(self):
        self.session_data = pd.DataFrame(columns=['ID', 'Image', 'Object Name', 'Object Index', 'Type', 'Coordinates'])
        self.window_title = " (TESAAR)"
        self.setWindowIcon(QIcon("../Icons/tata-elxsi-icon.jpg"))
        self.setWindowTitle(self.window_title)

        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.showMaximized()
        # self.setStyleSheet('QPushButton:!hover {color: orange} QLineEdit:!hover {color: orange}')
        # col = '#F2F2F2 '
        # self.setStyleSheet(f'QWidget {{background-color: {col};}}')

        self.toolBar = QToolBar()
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.tool_items = setup_toolbar(self)
        self.toolBar.setMovable(False)
        self.addToolBarBreak()

        self.toolBar2 = QToolBar()
        self.addToolBar(QtCore.Qt.BottomToolBarArea, self.toolBar2)
        self.toolBar2.setMovable(False)

        # self.toolBar3 = self.addToolBar('Tools')
        # self.toolBar3.setMovable(False)
        self.toolbarnew = QToolBar()
        self.addToolBar(self.toolbarnew)
        self.toolbarnew.setMovable(False)

        self.top_right_widgets = {'Add Label': (QLineEdit(), self.add_session_label)}
        self.right_widgets = {
            'Labels': QListWidget(),
            'Image Label List': QListWidget(),
            'Image List': QListWidget(),
            'Statics Display' : QListWidget(),
            'Navigation': QDockWidget("Zoom")
        }
        self.left_widgets = {'Image': self.current_image_area(None, self)}

        self.adjust_tool_bar1()
        # self.adjust_tool_bar3()
        self.setStatusBar(QStatusBar(self))
        self.central_widget = QWidget(self)

        self.main_layout = QHBoxLayout()
        # virtical box layout
        self.left_layout = QVBoxLayout()
        self.adjust_widgets()
        self.adjust_layouts()
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()

    def adjust_tool_bar1(self):
        """
        Adjust the left toolbar and setup buttons/icons.

        Return:
            None
        """

        self.toolBar.setStyleSheet("QToolButton{font: 12px;}");
        if sys.platform == 'darwin':
            self.setUnifiedTitleAndToolBarOnMac(True)

        self.spacer3 = QWidget()
        self.spacer3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolBar.addWidget(self.spacer3)
        for (label, icon_file, widget_method, status_tip, key, check) in self.tool_items.values():
            action = QAction(self)
            action.setText(f'&{label}')
            action.setIcon(QIcon(rf'..\Icons\New\{icon_file}'))
            self.toolBar.setIconSize(QSize(50, 40))
            action.setStatusTip(status_tip)
            action.setShortcut(key)

            if check:
                action.setCheckable(True)

            if label == 'Delete':
                action.setShortcut('Backspace')

            action.triggered.connect(widget_method)
            self.toolBar.addAction(action)

            if label == 'Auto Mode':
                self.toolBar.widgetForAction(action).setPopupMode(QToolButton.MenuButtonPopup)

                self.actionModel1 = QAction()
                self.actionModel1.setObjectName("actionModel1")
                self.actionModel1.setText("1080p medium mosaic")
                self.actionModel1.triggered.connect(self.go_to_Model1)

                self.actionModel2 = QAction()
                self.actionModel2.setObjectName("actionModel2")
                self.actionModel2.setText("1080p medium rect")
                self.actionModel2.triggered.connect(self.go_to_Model2)

                self.actionModel3 = QAction()
                self.actionModel3.setObjectName("action3")
                self.actionModel3.setText("1080p small mosaic")
                self.actionModel3.triggered.connect(self.go_to_Model3)

                self.actionModel4 = QAction()
                self.actionModel4.setObjectName("action4")
                self.actionModel4.setText("1080p small rect")
                self.actionModel4.triggered.connect(self.go_to_Model4)

                self.actionModel5 = QAction()
                self.actionModel5.setObjectName("action4")
                self.actionModel5.setText("720p medium mosaic")
                self.actionModel5.triggered.connect(self.go_to_Model5)

                self.actionModel6 = QAction()
                self.actionModel6.setObjectName("action4")
                self.actionModel6.setText("720p medium rect")
                self.actionModel6.triggered.connect(self.go_to_Model6)

                self.actionModel7 = QAction()
                self.actionModel7.setObjectName("action4")
                self.actionModel7.setText("720p small mosaic")
                self.actionModel7.triggered.connect(self.go_to_Model7)

                self.actionModel8 = QAction()
                self.actionModel8.setObjectName("action4")
                self.actionModel8.setText("720p small rect")
                self.actionModel8.triggered.connect(self.go_to_Model8)

                self.Automenu = QMenu()
                self.Automenu.addActions([self.actionModel1, self.actionModel2, self.actionModel3,
                                          self.actionModel4, self.actionModel5, self.actionModel6,
                                          self.actionModel7, self.actionModel8])
                action.setMenu(self.Automenu)
            if label == 'Segmentation Mode':
                self.toolBar.widgetForAction(action).setPopupMode(QToolButton.MenuButtonPopup)

                self.actionSegModel1 = QAction()
                self.actionSegModel1.setObjectName("manualSegModel")
                self.actionSegModel1.setText("Manual")
                self.actionSegModel1.triggered.connect(self.go_to_SegManual)

                self.actionSegModel2 = QAction()
                self.actionSegModel2.setObjectName("autoSegPolygonModel")
                self.actionSegModel2.setText("Auto (Polygon Output)")
                self.actionSegModel2.triggered.connect(self.go_to_SegAutoPolygon)

                self.actionSegModel3 = QAction()
                self.actionSegModel3.setObjectName("autoSegBinaryMaskModel")
                self.actionSegModel3.setText("Auto (Binary Mask Output)")
                self.actionSegModel3.triggered.connect(self.go_to_SegAutoBinaryMask)

                self.actionSegModel4 = QAction()
                self.actionSegModel4.setObjectName("autoSegBinaryMaskModelInstance")
                self.actionSegModel4.setText("Auto Instance (Binary Mask Output)")
                self.actionSegModel4.triggered.connect(self.go_to_SegAutoBinaryMaskInstance)

                self.actionSegModel5 = QAction()
                self.actionSegModel5.setObjectName("autoSegSemantic")
                self.actionSegModel5.setText("Semantic")
                self.actionSegModel5.triggered.connect(self.go_to_SegSemantic)

                self.Segmenu = QMenu()
                self.Segmenu.addActions([self.actionSegModel1, self.actionSegModel2, self.actionSegModel3, self.actionSegModel4, self.actionSegModel5])
                action.setMenu(self.Segmenu)

            if label == 'Blur Filter':
                self.toolBar.widgetForAction(action).setPopupMode(QToolButton.MenuButtonPopup)
                # self.toolBar.widgetForAction(action).setPopupMode(QToolButton.InstantPopup)

                self.actionGaussian = QAction()
                self.actionGaussian.setObjectName("actionGaussian")
                self.actionGaussian.setText("Gaussian Blur")
                self.actionGaussian.triggered.connect(self.go_to_Gaussian)

                self.actionMedian = QAction()
                self.actionMedian.setObjectName("actionMedian")
                self.actionMedian.setText("Median Blur")
                self.actionMedian.triggered.connect(self.go_to_Median)

                self.actionBilateral = QAction()
                self.actionBilateral.setObjectName("actionBilateral")
                self.actionBilateral.setText("Bilateral Blur")
                self.actionBilateral.triggered.connect(self.go_to_Bilateral)

                self.Blurmenu = QMenu()
                self.Blurmenu.addActions([self.actionGaussian, self.actionMedian, self.actionBilateral])
                action.setMenu(self.Blurmenu)

            if label == "Save Annotation":
                self.toolBar.widgetForAction(action).setPopupMode(QToolButton.MenuButtonPopup)
                self.json_format = QAction()
                self.json_format.setObjectName("json_format")
                self.json_format.setText("Saving file in Json Format")
                self.json_format.triggered.connect(self.save_changes)

                self.Csv_format = QAction()
                self.Csv_format.setObjectName("Csv_format")
                self.Csv_format.setText("Saving file in Csv Format")
                self.Csv_format.triggered.connect(self.save_changes)

                self.Savemenu = QMenu()
                self.Savemenu.addActions([self.Csv_format, self.json_format])
                action.setMenu(self.Savemenu)

        self.spacer4 = QWidget()
        self.spacer4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolBar.addWidget(self.spacer4)

        self.logo = QLabel()
        self.logo.setGeometry(QtCore.QRect(0, 0, 1, 3))
        self.logo.setPixmap(QtGui.QPixmap("../Icons/New/logo.png"))
        self.logo.setScaledContents(True)
        self.logo.setMinimumWidth(200)

        self.toolBar.addWidget(self.logo)

    # for video-play icons
    def adjust_tool_bar2(self):
        """
        Adjust the bottom toolbar and setup buttons/icons.

        Return:
            None
        """
        self.actionPrevious = QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../Icons/New/prev.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPrevious.setIcon(icon1)
        self.actionPrevious.setObjectName("actionforward")
        self.actionPrevious.triggered.connect(self.go_to_previous)

        self.actionPlay_Pause = QAction()
        self.actionPlay_Pause.setCheckable(True)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../Icons/New/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap("../Icons/New/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionPlay_Pause.setIcon(icon2)
        self.actionPlay_Pause.triggered.connect(self.play)

        self.actionNext = QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../Icons/New/next.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNext.setIcon(icon3)
        self.actionNext.setObjectName("actionNext")
        self.actionNext.triggered.connect(self.go_to_next)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.sliderMoved.connect(self.set_range)
        self.slider2.valueChanged.connect(self.set_position)
        self.slider2.setStyleSheet(
            "QSlider::groove:horizontal {height: 1px solid #bbb; margin: 0 0; background: solid gray; height: 17px; border-radius: 6px; width: 1200px}"
            "QSlider::sub-page:horizontal {background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #66e, stop: 1 #bbf);background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,stop: 0 #bbf, stop: 1 #55f);border: 1px solid #777;height: 10px;border-radius: 4px;}"
            "QSlider::add-page:horizontal {background: #fff;border: 1px solid #777;height: 10px;border-radius: 4px;}"
            "QSlider::handle:horizontal {background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,stop:0 #eee, stop:1 #ccc); border: 1px solid #777; width: 13px;margin-top: -2px; margin-bottom: -2px; border-radius: 4px;}"
            "QSlider::handle:horizontal:hover {background: qlineargradient(x1:0, y1:0, x2:1, y2:1,stop:0 #fff, stop:1 #ddd);border: 1px solid #444; border-radius: 6px;}"
            "QSlider::sub-page:horizontal:disabled {background: #bbb; border-color: #999;}"
            "QSlider::add-page:horizontal:disabled {background: #eee;border-color: #999;}"
            "QSlider::handle:horizontal:disabled {background: #eee;border: 1px solid #aaa;border-radius: 6px;}")
        self.slider2.setMinimumWidth(1200)

        self.spacer5 = QWidget()
        self.spacer5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.slider2.sliderPressed.connect(self.set_frame)

        self.toolBar2.setIconSize(QSize(40, 40))

        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionPrevious)
        self.toolBar2.addAction(self.actionPlay_Pause)
        # self.toolBar2.addAction(self.actionPause)
        self.toolBar2.addAction(self.actionNext)
        self.toolBar2.addWidget(self.slider2)
        self.toolBar2.addWidget(self.spacer5)

    def adjust_tool_bar3(self):
        """
        Adjust the top toolbar and setup buttons/icons.

        Return:
            None
        """
        self.spacer1 = QWidget()
        self.spacer1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.actionzoomIn = QAction("Zoom &In (10%)", self, shortcut="Ctrl++", triggered=self.zoomIn)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../Icons/ZoomIn.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionzoomIn.setIcon(icon6)

        self.actionzoomOut = QAction("Zoom &Out (10%)", self, shortcut="Ctrl+-", triggered=self.zoomOut)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("../Icons/ZoomOut.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionzoomOut.setIcon(icon7)

        self.actionFit = QAction("Fit image to screen size", self, shortcut="Ctrl+z", triggered=self.fit_image)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("../Icons/reset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFit.setIcon(icon8)

        self.spacer = QWidget()
        self.spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.logo = QLabel()
        self.logo.setGeometry(QtCore.QRect(0, 0, 1, 3))
        self.logo.setPixmap(QtGui.QPixmap("../Icons/logo.png"))
        self.logo.setScaledContents(True)

        self.toolBar3.setIconSize(QSize(36, 36))

        self.toolBar3.addWidget(self.spacer1)
        self.toolBar3.addAction(self.actionzoomIn)
        self.toolBar3.addAction(self.actionzoomOut)
        self.toolBar3.addAction(self.actionFit)
        self.toolBar3.addWidget(self.spacer)
        self.toolBar3.addWidget(self.logo)

    def adjust_layouts(self):
        """
        Adjust window layouts.

        Return:
            None
        """
        self.main_layout.addLayout(self.left_layout)
        # main window layout
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    def adjust_widgets(self):
        """
        Adjust window widgets.

        Return:
            None
        """
        self.left_layout.addWidget(self.left_widgets['Image'])
        for text, (widget, widget_method) in self.top_right_widgets.items():
            dock_widget = QDockWidget(text)
            dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
            dock_widget.setWidget(widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
            if widget_method:
                widget.editingFinished.connect(widget_method)
        self.top_right_widgets['Add Label'][0].setPlaceholderText('Add Label')
        self.right_widgets['Image List'].selectionModel().currentChanged.connect(self.display_selection)
        for text, widget in self.right_widgets.items():
            dock_widget = QDockWidget(text)
            dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
            dock_widget.setWidget(widget)
            self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)

        self.customize_labels_widget()
        self.customize_navigation_widget()

    def customize_navigation_widget(self):
        print("customize_navigation_widget")
        self.right_widgets['Navigation'].setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.nav_hbox = QHBoxLayout(self)
        self.nav_widg = QWidget(self)

        # Zoom In
        self.zoomin = QPushButton("+")
        self.zoomin.setMaximumWidth(25)
        self.zoomin.clicked.connect(self.zoom_in)

        # Zoom Out
        self.zoomout = QPushButton("-")
        self.zoomout.setMaximumWidth(25)
        self.zoomout.clicked.connect(self.zoom_out)

        # Zoom slider
        self.zoomslider = QSlider(Qt.Horizontal)

        self.zoomslider.setMinimum(0)
        self.zoomslider.setMaximum(200)
        # initial slider pos value i.e. zoomSliderVal = 100
        self.zoomslider.setValue(self.zoomSliderVal)
        # initial slider pos
        self.zoomslider.setTickPosition(QSlider.TicksBelow)
        # tick intervals
        self.zoomslider.setTickInterval(50)
        self.zoomslider.setMinimumWidth(250)

        # self.showme = QPushButton("$")
        # self.showme.setMaximumWidth(20)
        # self.showme.clicked.connect(self.show_rectangle_boxes)

        self.reset = QPushButton("Fit to screen")
        self.reset.setMaximumWidth(100)
        self.reset.clicked.connect(self.fit_image)

        # Adding Widgets to the navigation dock
        self.nav_hbox.addWidget(self.zoomout)
        self.nav_hbox.addWidget(self.zoomslider)
        self.nav_hbox.addWidget(self.zoomin)
        self.nav_hbox.addWidget(self.reset)
        # self.nav_hbox.addWidget(self.showme)
        self.nav_widg.setLayout(self.nav_hbox)

        self.right_widgets['Navigation'].setWidget(self.nav_widg)

    def zoom_in(self):
        if self.video_flag or self.photo_flag:
            self.zoomSliderVal = self.zoomSliderVal + 5
            self.zoomslider.setValue(self.zoomSliderVal)
            # print("zoom-in, v:", self.zoomSliderVal)
            self.zoom_image(1.05)
        else:
            self.warn_user_to_upload_file()

    def zoom_out(self):
        if self.video_flag or self.photo_flag:
            self.zoomSliderVal = self.zoomSliderVal - 5
            self.zoomslider.setValue(self.zoomSliderVal)
            # print("zoom-out, v:", self.zoomSliderVal)
            self.zoom_image(0.95)
        else:
            self.warn_user_to_upload_file()

    def customize_labels_widget(self):
        """
        Add delete button and sample labels to Labels widget.

        Return:
            None
        """
        self.delete_label = QPushButton("Delete label")
        self.delete_label.clicked.connect(self.delete_selected_labels)
        myQListWidgetItem = QListWidgetItem(self.right_widgets['Labels'])
        myQListWidgetItem.setSizeHint(self.delete_label.sizeHint())
        self.right_widgets['Labels'].addItem(myQListWidgetItem)
        self.right_widgets['Labels'].setItemWidget(myQListWidgetItem, self.delete_label)

        # label list
        for label in categories:
            self.add_to_list(label, self.right_widgets['Labels'])

    def delete_selected_labels(self):
        """
        Delete the labels selected in label list

        Return:
            None
        """
        widget_list = self.right_widgets['Labels']
        items = [widget_list.item(i) for i in range(widget_list.count())]
        checked_indexes = [checked_index for checked_index, item in enumerate(items) if item.checkState() == Qt.Checked]
        for q_list_index in reversed(checked_indexes):
            widget_list.takeItem(q_list_index)

    @staticmethod
    def add_to_list(item, widget_list):
        """
        Add item to one of the right QWidgetList(s).
        Args:
            item: str : Item to add.
            widget_list: One of the right QWidgetList(s).

        Return:
            None
        """

        item = QListWidgetItem(item)
        item.setFlags(
            item.flags()
            | Qt.ItemIsSelectable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
        )

        item.setCheckState(Qt.Unchecked)
        widget_list.addItem(item)
        widget_list.selectionModel().clear()

    @staticmethod
    def delete_from_list(widget_list, i):
        """
        Add item to one of the right QWidgetList(s).
        Args:
            item: str : Item to add.
            widget_list: One of the right QWidgetList(s).

        Return:
            None
        """

        widget_list.takeItem(i)
        widget_list.selectionModel().clear()

    @staticmethod
    def change_item_index(item, widget_list, i):
        """
        Add item to one of the right QWidgetList(s).
        Args:
            item: str : Item to add.
            widget_list: One of the right QWidgetList(s).

        Return:
            None
        """

        item = QListWidgetItem(item)
        item.setFlags(
            item.flags()
            | Qt.ItemIsSelectable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
        )

        item.setCheckState(Qt.Unchecked)
        # remove the old item from the list
        widget_list.takeItem(i)

        # add the updated item back into the list at index 1
        widget_list.insertItem(i, item)

    @staticmethod
    def get_sub_set(widget_list, n):
        """
        Add item to one of the right QWidgetList(s).
        Args:
            item: str : Item to add.
            widget_list: One of the right QWidgetList(s).

        Return:
            None
        """

        '''item = QListWidgetItem(item)
        item.setFlags(
            item.flags()
            | Qt.ItemIsSelectable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
        )

        item.setCheckState(Qt.Unchecked)'''

        # remove the last n items
        for i in range(n, widget_list.count()):
            item_temp = widget_list.takeItem(n)
        widget_list.selectionModel().clear()

    def display_selection(self):
        """
        Display image that is selected in the right Photo list.

        Return:
            None
        """
        self.right_widgets['Image Label List'].clear()

        self.frame_dir, self.current_image = self.get_current_selection('photo')
        if not self.frame_dir:
            return

        frame_name = self.images[self.photo_row].split('/')[-1].replace('temp-', '')
        for item in self.session_data.loc[self.session_data['Image'] == frame_name].values:
            self.add_to_list(f'{[[x for x in item]]}', self.right_widgets['Image Label List'])
        self.choose_display(self.photo_row)
        #self.fit_image()

    def get_current_selection(self, display_list):
        """
        Get current selected item data.
        Args:
            display_list: One of the right QWidgetList(s).

        Return:
            Image path or current row.
        """
        if display_list == 'photo':
            print("get_current_selection photo")
            current_selection = self.right_widgets['Image List'].currentRow()
            self.photo_row = current_selection
            self.left_widgets['Image'].row = self.photo_row
            if current_selection >= 0:
                if not self.video_flag:
                    # global res_width, res_height, scale_width, scale_ht, photo_res
                    global res_width, res_height, scale_width, scale_ht
                    global img_w, img_h
                    if img_w == 0 or img_h == 0:
                        img_w, img_h = self.photo_res[current_selection]
                    #scale_width, scale_ht = res_width, res_height

                    #self.zoom_image(1.0)
                    return self.images[current_selection], self.image_pixels[current_selection]
                else:
                    return self.images[current_selection], self.video_frame_pixels[current_selection]
            else:
                return None, None
            self.right_widgets['Image List'].selectionModel().clear()

        if display_list == 'selected_label':
            current_selection = self.right_widgets['Labels'].currentRow()
            if current_selection >= 0:
                return current_selection

    def choose_display(self, row):
        """
        Choose if image to be displayed contains annotations.
        Args:
            row: value: Row number of Photo list.

        Return:
            None.
        """
        global rectangle_highlight, polygon_highlight, polyline_highlight, cube_highlight

        if self.right_widgets['Image Label List'].count() > 0:
            rectangle_highlight = -1
            polyline_highlight = -1
            polygon_highlight = -1
            cube_highlight = -1
            edit_highlight.clear()
            # self.left_widgets['Image'].highlight.clear()
            self.left_widgets['Image'].show_annotations(row)
        else:
            self.left_widgets['Image'].switch_frame(self.current_image)

    def initial_display(self):
        """
        Display first image in Image List fit to screen with default label list
        Args:
            row: value: Row number of Image list.

        Return:
            None.
        """
        self.right_widgets['Image List'].setCurrentRow(0)
        global scale_width, scale_ht, win_width, win_ht, img_w, img_h
        '''
        if scale_width < win_width and scale_ht < win_ht:
            while scale_width < win_width and scale_ht < win_ht:
                self.zoomInSmall()
        elif scale_width > win_width or scale_ht > win_ht:
            while scale_width > win_width or scale_ht > win_ht:
                self.zoomOutSmall()
        '''
        #if (img_w > win_width or img_h > win_ht):
        #    while (img_w > win_width and img_h > win_ht):
        #        self.zoomOutSmall()
        #elif (img_w < win_width or img_h < win_ht):
        #    while (img_w < win_width and img_h < win_ht):
        #        self.zoomInSmall()
        #self.annotation_id = 0
        print("initial_display")
        # if self.right_widgets['Labels'].count() <= 0:
        #     self.customize_labels_widget()
        if self.photo_flag:
            self.left_widgets['Image'].setImageFileName(self.curr_frame_dir+"/"+self.curr_frame)
        elif self.video_flag:
            self.left_widgets['Image'].setImageFrame(self.video_frame_pixels[0])


    def upload_photos(self):
        """
        Add image(s) to the right photo list.
        Return:
            None
        """
        if self.photo_flag or self.video_flag:
            self.show_prompt()
            if self.answer == QMessageBox.Cancel:
                return
        # if self.toolBar2:
        #     self.toolBar2.deleteLater()
        file_dialog = QFileDialog()
        img_dirs, _ = file_dialog.getOpenFileNames(self, 'Upload Photos',
                                                   r"..\\..\\tesar\\sample files",
                                                   "Image files(*.jpg *.jpeg *.png)")

        if len(img_dirs) > 0:
            self.main_dir = img_dirs[0]
            split = len(self.main_dir.split('/')[-1])
            self.dir_folder_name = self.main_dir[0:-(split)]

            for img_dir in img_dirs:
                image_folder_name, image_name = ('/'.join(img_dir.split('/')[:-1]), img_dir.split('/')[-1],)
                self.curr_frame_dir = image_folder_name
                self.curr_frame = image_name
                self.add_to_list(image_name, self.right_widgets['Image List'])
                self.images.append(image_name)
                self.image_paths[image_name] = image_folder_name
                self.image_pixels.append(cv2.imread(img_dir))
                self.photo = im.open(img_dir)
                self.photo_res.append(self.photo.size)
            self.photo_flag = True
            self.video_flag = False
            self.initial_display()

    def upload_video(self):
        """
        Add video and split them into frames to add to the right photo list.

        Return:
            None
        """
        if self.photo_flag or self.video_flag:
            self.show_prompt()
            if self.answer == QMessageBox.Cancel:
                return

        self.video_dir = False
        self.video_dir, _ = QFileDialog.getOpenFileName(self, "Open Video",
                                                        r"..\\..\\tesar\\sample files",
                                                        "Video (*.mp4 *.avi)")

        if self.video_dir:
            self.main_dir = self.video_dir
            split = len(self.video_dir.split('/')[-1])
            self.dir_folder_name = self.video_dir[0:-(split)]
            vid = cv2.VideoCapture(self.video_dir)

            self.video_fps = round(vid.get(cv2.CAP_PROP_FPS))
            self.video_codec = int(vid.get(cv2.CAP_PROP_FOURCC))
            self.frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.video_framecount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

            currentframe = 0

            while (True):
                ret, frame = vid.read()

                if ret:
                    self.video_frame_pixels.append(frame)
                    frame_name = "frame" + str(currentframe)
                    self.add_to_list(frame_name, self.right_widgets['Image List'])
                    self.images.append(f'{frame_name}')
                    self.image_paths[frame_name] = 'video_frame_pixels' + '[' + str(currentframe) + ']'

                    currentframe += 1
                else:
                    break
            vid.release()

            global res_width, res_height, scale_width, scale_ht
            global img_w, img_h
            img_w = (self.video_size[0])
            img_h = (self.video_size[1])
            #scale_width, scale_ht = res_width, res_height
            self.video_flag = True
            self.photo_flag = False
            if self.video_flag:
                self.adjust_tool_bar2()
            self.initial_display()

    def get_image_label_list_from_index(self, idx, attr="id"):
        session_name = str((self.right_widgets['Image Label List'].item(idx).text()))
        obj_name = session_name.split(',')
        if attr == "id":
            id = str(obj_name[0])
            id = id.replace("[", "")
            id = id.replace("]", "")
            id = id.replace("'", "")
            id = int(id)
            return id 
        elif attr == "coords":
            coords = obj_name[5:obj_len]
            coords = str(coords)
            coords = coords.replace("[", "")
            coords = coords.replace("]", "")
            coords = coords.replace("'", "")
            coords = coords.replace("\"", "")
            coords = coords.replace(" ", "")
            coords = coords.split(",")
            return coords
        elif attr == "label id":
            label_id = int(obj_name[3])
            return label_id 
        elif attr == "shape type":
            typeofshape = obj_name[4]
            return typeofshape 
        else:
            return None

    def find_matching_coordinates(self, contour, contour_type):
        for i in range(self.right_widgets['Image Label List'].count()):
            session_name = str((self.right_widgets['Image Label List'].item(i).text()))
            obj_name = session_name.split(',')
            obj_len = len(obj_name)
            label_id = int(obj_name[3])
            typeofshape = obj_name[4]

            coords = obj_name[5:obj_len]
            coords = str(coords)
            coords = coords.replace("[", "")
            coords = coords.replace("]", "")
            coords = coords.replace("'", "")
            coords = coords.replace("\"", "")
            coords = coords.replace(" ", "")
            coords = coords.split(",")

            id = str(obj_name[0])
            id = id.replace("[", "")
            id = id.replace("]", "")
            id = id.replace("'", "")
            id = int(id)

            if typeofshape == contour_type == " 'Rectangle'":
                global selected_rectangle
                print("rectangle coords in match", coords)
                print("contour-matching", contour)
                pts = []
                x = int(coords[0])
                y = int(coords[1])
                w = int(coords[2])
                h = int(coords[3])

                i_tl = (x, y)
                i_tr = (x + w, y)
                i_br = (x + w, y + h)
                i_bl = (x, y + h)
                pts.clear()
                for item in [i_tl, i_tr, i_br, i_bl]:
                    pts.append(item)
                print("points", pts)
                if pts == contour:
                    selected_rectangle = pts.copy()
                    return id, label_id

            elif typeofshape == contour_type == " 'Polygon'":
                print("coords poly", coords)
                print("polygon contour values-matching", contour)

                length = len(coords)
                finalcoords = []
                loop = 0
                while loop < length:
                    x_coord = int(coords[loop])
                    y_coord = int(coords[loop + 1])
                    a = [x_coord, y_coord]
                    finalcoords.append(a)
                    loop += 2
                if finalcoords == contour:
                    print("polygon contour matched")
                    return id

            elif typeofshape == contour_type == " 'Cube'":
                global selected_cube
                pts = []
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                w = int(coords[4])
                h = int(coords[5])

                # all four coordinates for the rectangle
                f_tl, f_tr = (x1, y1), (x1 + w, y1)
                f_br, f_bl = (x1 + w, y1 + h), (x1, y1 + h)

                b_tl, b_tr = (x2, y2), (x2 + w, y2)
                b_br, b_bl = (x2 + w, y2 + h), (x2, y2 + h)

                for item in [b_bl, b_tl, b_tr, f_tr, f_br, f_bl]:
                    pts.append(item)
                if pts == contour:
                    selected_cube = [f_tl, f_tr, f_br, f_bl, b_tl, b_tr, b_br, b_bl]
                    print("cube contour matched")
                    return id, label_id

            elif typeofshape == contour_type == " 'Polyline'":
                length = len(coords)
                finalcoords = []
                loop = 0

                while loop < length:
                    x_coord = int(coords[loop])
                    y_coord = int(coords[loop + 1])
                    a = [x_coord, y_coord]
                    finalcoords.append(a)
                    loop += 2

                if finalcoords == contour:
                    print("polyline contour matched")
                    return id, label_id
        return -1

    def save_blur(self):
        if not self.video_flag:
            count = 0
            for i in range(len(self.images)):
                labels = self.session_data.loc[self.session_data['Image'] == self.images[i]].values
                if len(labels) != 0:
                    count += 1
                    frame_read = self.image_pixels[i]
                    self.to_blur = frame_read.copy()
                    blurred = self.labels_to_blur(labels)
                    cv2.imwrite(str(self.dir_folder_name) + 'blur-' + str(self.images[i]), blurred)

            if count != 0:
                info = QMessageBox()
                info.setIcon(QMessageBox.Information)
                info.setText("Work saved to " + str(self.dir_folder_name))
                info.setWindowTitle("TESAR")
                info.exec_()

        else:
            self.export_work()
        self.restore_variables()
        return

    def show_prompt(self):
        msgBox = QMessageBox();
        msgBox.setWindowTitle("TESAR")
        msgBox.setText("Current work will be lost, do you want to save it?")
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Save)
        self.answer = msgBox.exec()

        if self.answer == QMessageBox.Save:
            # self.save_blur()
            self.save_changes()

        elif self.answer == QMessageBox.Discard:
            self.restore_variables()
            return
        else:
            return

    def restore_variables(self):
        # self.right_widgets['Labels'].clear()
        self.right_widgets['Image Label List'].clear()
        self.right_widgets['Image List'].clear()

        self.images.clear()
        self.image_paths.clear()
        self.image_pixels.clear()
        self.photo_res.clear()
        self.video_frame_pixels.clear()

        self.session_data = self.session_data[0:0]
        # session_data has columns ['Image', 'Object Name', 'Object Index', 'Type', 'Coordinates']

        # self.photo_flag = False
        # self.video_flag = False
        self.scaleFactor = 1.0

        global res_width, res_height, scale_width, scale_ht
        global delta_x, delta_y, offset_x, offset_y
        global img_x, img_y, img_w, img_h

        res_width, res_height, scale_width, scale_ht = 0, 0, 0, 0
        delta_x, delta_y, offset_x, offset_y = 0, 0, 0, 0
        img_x, img_y, img_w, img_h = 0, 0, 0, 0

    def show_annotations_from_file(self, csv_file):
        # print(csv_file)

        frame_name = self.images[self.photo_row].split('/')[-1]
        for item in csv_file.loc[csv_file['Image'] == frame_name].values:
            self.add_to_list(f'{[[x for x in item]]}', self.right_widgets['Image Label List'])
            # to_add = pd.DataFrame(item, columns=self.session_data.columns)
            # self.session_data = pd.concat([self.session_data, to_add], ignore_index=True)
        self.left_widgets['Image'].show_annotations(self.photo_row)

    def upload_annotation(self):
        file_dialog = QFileDialog()
        ann_dir, _ = file_dialog.getOpenFileNames(self, 'Add Annotation file',
                                                  r"..\\..\\tesar\\saved_annotations",
                                                  "file(*.csv)")
        print("ann", ann_dir)
        if len(ann_dir) > 0:
            self.annotation_file_dir = '/'.join(ann_dir[0].split('/')[:-1])
            self.annotation_file_name = ann_dir[0].split('/')[-1]

            print("annotation dir", self.annotation_file_dir)
            print("annotation file", self.annotation_file_name)
            file = str(self.annotation_file_dir + "\\" + self.annotation_file_name)
            csv_file = pd.read_csv(file)
            self.show_annotations_from_file(csv_file)
        else:
            return

    def auto_mode(self):
        if not self.video_flag and not self.photo_flag:
            self.warn_user_to_upload_file()
        else:
            MODEL_VARS = '../yolov3/'#'../models/yolo/'
            yolo = cv2.dnn.readNet(MODEL_VARS + "yolov3.weights", MODEL_VARS + "yolov3.cfg")
            global img_h, img_w
            multiplier = self.scaleFactor
            width, height = img_w * (1 / multiplier), img_h * (1 / multiplier)
            with open(MODEL_VARS + "coco.names", "r") as file:
                classes = [line.strip() for line in file.readlines()]
            layer_names = yolo.getLayerNames()
            output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]


            self.right_widgets['Labels'].clear()


            if self.photo_flag:
                image = self.left_widgets['Image'].current_image
                self.yolo_helper(image, yolo, width, height, classes, output_layers, True)
            elif self.video_flag:
                for i in range(0, self.right_widgets['Image List'].count()):
                    image = self.left_widgets['Image'].current_image
                    self.yolo_helper(image, yolo, width, height, classes, output_layers, False)
                    self.choose_display_frame(i)
                    self.frametimer = QTimer()
                    self.videoSlider.setValue(i)

    def fix_polygon_with_hole(self,polygon):
        # Find the outer boundary of the polygon
        outer_boundary = []
        inner_boundary = []
        for i in range(len(polygon)):
            if i < len(polygon) - 1:
                dx1 = polygon[i+1][0] - polygon[i][0]
                dy1 = polygon[i+1][1] - polygon[i][1]
            else:
                dx1 = polygon[0][0] - polygon[i][0]
                dy1 = polygon[0][1] - polygon[i][1]
            if i > 0:
                dx2 = polygon[i][0] - polygon[i-1][0]
                dy2 = polygon[i][1] - polygon[i-1][1]
            else:
                dx2 = polygon[i][0] - polygon[len(polygon)-1][0]
                dy2 = polygon[i][1] - polygon[len(polygon)-1][1]
            cross_product = dx1 * dy2 - dy1 * dx2
            if cross_product < 0:
                outer_boundary.append(polygon[i])
            else:
                inner_boundary.append(polygon[i])
        # Reverse the order of the points in the hole
        inner_boundary.reverse()
        # Add the hole as a new polygon to the original polygon
        polygon_with_hole = [outer_boundary, inner_boundary]
        # Remove the original hole from the polygon
        polygon_with_hole.remove([])
        return polygon_with_hole

    def segmentation_mode(self):
        if self.segmentation_flag == "manual":
            if self.video_flag or self.photo_flag:
                if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
                    self.setWindowTitle('TESAR(Editor Mode Segmentation)')
                    #self.switch_editor(PolygonImageEditor)
                    self.left_widgets['Image'].setMode('segmentation')
                else:
                    self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
                    #self.switch_editor(RegularImageArea)
                    self.left_widgets['Image'].setMode('none')
                self.display_selection()
            else:
                self.warn_user_to_upload_file()
                return 
        elif self.segmentation_flag == "semantic": # auto

            
            if not self.video_flag and not self.photo_flag:
                self.warn_user_to_upload_file()
            else:
                # load MaskFormer fine-tuned on COCO panoptic segmentation
                feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")

                pix = self.left_widgets['Image']._image.pixmap().toImage() #im.open(self.images[self.photo_row])
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pix.save(buffer, "PNG")
                image = im.open(io.BytesIO(buffer.data()))
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                result = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                print(result)
                #predicted_mask = result.detach().cpu().numpy()
                predicted_mask = np.array(result)
                print(predicted_mask)
                print(predicted_mask.shape)
                w,h = predicted_mask.shape
                unique_ids = list(set(predicted_mask.reshape((1,w*h))[0]))
                binary_masks = []
                for segment in unique_ids:
                    binary_mask = predicted_mask == segment
                    binary_masks.append(binary_mask)

                self.right_widgets['Labels'].clear()
                
                classes = set()
                self.left_widgets['Image'].mode = "segmentation"
                new_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                for i,binary_mask in enumerate(binary_masks):
                    class_ = model.config.id2label[unique_ids[i]]
                    print(class_)
                    if class_ not in classes:
                        self.add_to_list(class_, self.right_widgets['Labels'])
                        classes.add(class_)

                    matching_items = self.right_widgets['Labels'].findItems(class_, QtCore.Qt.MatchContains)

                    item = matching_items[0] #self.right_widgets['Labels'].item(result['segments_info'][i]['label_id'])
                    index = self.right_widgets['Labels'].row(item)
                    # select the item
                    self.right_widgets['Labels'].setCurrentItem(item)
                    self.left_widgets['Image'].binary_mask = binary_mask.astype(int)
                    self.left_widgets['Image'].store_coords_binary_mask()

                    clr = categories_color[index]
                    new_mask[binary_mask] = (clr.red(), clr.green(), clr.blue())
                predicted_image = im.fromarray(new_mask)
                overlay = im.blend(image.convert("RGBA"), predicted_image.convert("RGBA"), alpha=0.5)

                # Convert the PIL image to a numpy array
                np_image = np.array(overlay)

                # Convert the numpy array to a cv2 image
                cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                self.left_widgets['Image'].switch_frame(cv2_image)
        elif self.segmentation_flag == "auto binary mask": # auto
            
            if not self.video_flag and not self.photo_flag:
                self.warn_user_to_upload_file()
            else:
                # load MaskFormer fine-tuned on COCO panoptic segmentation
                feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")

                pix = self.left_widgets['Image']._image.pixmap().toImage() #im.open(self.images[self.photo_row])
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pix.save(buffer, "PNG")
                image = im.open(io.BytesIO(buffer.data()))
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                predicted_mask = result["segmentation"].detach().cpu().numpy()
                binary_masks = []
                for segment in result['segments_info']:
                    binary_mask = predicted_mask == segment['id']
                    binary_masks.append(binary_mask)

                self.right_widgets['Labels'].clear()
                
                classes = set()
                self.left_widgets['Image'].mode = "segmentation"
                new_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                for i,binary_mask in enumerate(binary_masks):
                    class_ = model.config.id2label[result['segments_info'][i]['label_id']]
                    print(class_)
                    if class_ not in classes:
                        self.add_to_list(class_, self.right_widgets['Labels'])
                        classes.add(class_)

                    matching_items = self.right_widgets['Labels'].findItems(class_, QtCore.Qt.MatchContains)

                    item = matching_items[0] #self.right_widgets['Labels'].item(result['segments_info'][i]['label_id'])
                    index = self.right_widgets['Labels'].row(item)
                    # select the item
                    self.right_widgets['Labels'].setCurrentItem(item)
                    self.left_widgets['Image'].binary_mask = binary_mask.astype(int)
                    self.left_widgets['Image'].store_coords_binary_mask()

                    clr = categories_color[index]
                    new_mask[binary_mask] = (clr.red(), clr.green(), clr.blue())
                predicted_image = im.fromarray(new_mask)
                overlay = im.blend(image.convert("RGBA"), predicted_image.convert("RGBA"), alpha=0.5)

                # Convert the PIL image to a numpy array
                np_image = np.array(overlay)

                # Convert the numpy array to a cv2 image
                cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                self.left_widgets['Image'].switch_frame(cv2_image)
        elif self.segmentation_flag == "auto binary mask instance": # auto
            
            if not self.video_flag and not self.photo_flag:
                self.warn_user_to_upload_file()
            else:
                # load MaskFormer fine-tuned on COCO panoptic segmentation
                feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")

                pix = self.left_widgets['Image']._image.pixmap().toImage() #im.open(self.images[self.photo_row])
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pix.save(buffer, "PNG")
                image = im.open(io.BytesIO(buffer.data()))
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                predicted_mask = result["segmentation"].detach().cpu().numpy()
                binary_masks = []
                for segment in result['segments_info']:
                    binary_mask = predicted_mask == segment['id']
                    binary_masks.append(binary_mask)

                self.right_widgets['Labels'].clear()
                
                classes = set()
                self.left_widgets['Image'].mode = "segmentation"
                new_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                categories_color_custom = list_color.get_colors(transparent,len(binary_masks))

                for i,binary_mask in enumerate(binary_masks):
                    class_ = model.config.id2label[result['segments_info'][i]['label_id']]
                    print(class_)
                    if class_ not in classes:
                        self.add_to_list(class_, self.right_widgets['Labels'])
                        classes.add(class_)

                    matching_items = self.right_widgets['Labels'].findItems(class_, QtCore.Qt.MatchContains)

                    item = matching_items[0] #self.right_widgets['Labels'].item(result['segments_info'][i]['label_id'])
                    index = self.right_widgets['Labels'].row(item)
                    # select the item
                    self.right_widgets['Labels'].setCurrentItem(item)
                    self.left_widgets['Image'].binary_mask = binary_mask.astype(int)
                    self.left_widgets['Image'].store_coords_binary_mask()

                    #clr = categories_color[index]
                    clr = categories_color_custom[i]
                    new_mask[binary_mask] = (clr.red(), clr.green(), clr.blue())
                predicted_image = im.fromarray(new_mask)
                overlay = im.blend(image.convert("RGBA"), predicted_image.convert("RGBA"), alpha=0.5)

                # Convert the PIL image to a numpy array
                np_image = np.array(overlay)

                # Convert the numpy array to a cv2 image
                cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                self.left_widgets['Image'].switch_frame(cv2_image)

        elif self.segmentation_flag == "auto polygon": # auto
            
            if not self.video_flag and not self.photo_flag:
                self.warn_user_to_upload_file()
            else:
                # load MaskFormer fine-tuned on COCO panoptic segmentation
                feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
                model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")

                #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
                #image = Image.open(requests.get(url, stream=True).raw)
                #print(self.images[self.photo_row])
                #print(type(self.images[self.photo_row]))
                pix = self.left_widgets['Image']._image.pixmap().toImage() #im.open(self.images[self.photo_row])
                #image = im.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pix.save(buffer, "PNG")
                image = im.open(io.BytesIO(buffer.data()))
                #buffer = QBuffer()
                #buffer.open(QBuffer.ReadWrite)
                #image = Image.open(io.BytesIO(buffer.data()))
                #image = pix.getImageData("RGB")
                inputs = feature_extractor(images=image, return_tensors="pt")
                #self.fitInView(self.zoomStack[-1], self.aspectRatioMode)  # Show zoomed rect.
                #self.left_widgets['Image'].fitInView(self.left_widgets['Image'].zoomStack[-1], self.left_widgets['Image'].aspectRatioMode)
                outputs = model(**inputs)
                result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                predicted_mask = result["segmentation"].detach().cpu().numpy()
                polygons = []
                #print("SHAPEEEEEEEEEEEEEEEEEE",image.size)
                #print("SHAPEEEEEEEEEEEEEEEEEE",predicted_mask.shape)
                #print(self.left_widgets['Image'].width(),self.left_widgets['Image'].height())
                # iterate through each segment in the predicted mask
                for segment in result['segments_info']:
                    # get the binary mask for the segment
                    binary_mask = predicted_mask == segment['id']

                    # convert the binary mask to a polygon
                    #contour = find_contours(binary_mask, 0.5)[0][:, ::-1]
                    #contour = find_boundaries(binary_mask, mode='inner')[:, ::-1]

                    #vertices, faces, _, _ = marching_cubes_lewiner(binary_mask, 0)
                    #contour = vertices[faces[:,::-1]]
                    
                    #contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #contour = contours[0][:,::-1]

                    #contour = cv2.approxPolyDP(contours[0][:,::-1], epsilon=1.0, closed=True)[:,::-1]
                    contour, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_SIMPLE)

                    contours = []
                    for c in contour[0]:
                        contours.append(c[0])
                    print(len(contours))
                    while len(contours) < 3:
                        contours.append(contours[0])

                    polygon = Polygon(contours)
                    # add the polygon to the list
                    tolerance = 0.9
                    polygon = polygon.simplify(tolerance)
                    ps = []    
                    for p in list(polygon.exterior.coords):
                        ps.append([(p[0]), (p[1])])
                    #print(ps)
                    # add the polygon to the list
                    polygons.append(ps)
                    #polygons.append(polygon.simplify(tolerance))

                self.right_widgets['Labels'].clear()
                
                #for label_id, text in model.config.id2label.items():
                #    #label_text = f'{label}: {count}'
                #    #self.add_to_list(label_text, self.right_widgets['Statics Display'])
                #    self.add_to_list(text, self.right_widgets['Labels'])

                classes = set()
                self.left_widgets['Image'].mode = "segmentation"
                for i,polygon in enumerate(polygons):
                    class_ = model.config.id2label[result['segments_info'][i]['label_id']]
                    print(class_)
                    #print(polygon)
                    if class_ not in classes:
                        self.add_to_list(class_, self.right_widgets['Labels'])
                        classes.add(class_)

                    matching_items = self.right_widgets['Labels'].findItems(class_, QtCore.Qt.MatchContains)

                    item = matching_items[0] #self.right_widgets['Labels'].item(result['segments_info'][i]['label_id'])
                    index = self.right_widgets['Labels'].row(item)
                    # select the item
                    self.right_widgets['Labels'].setCurrentItem(item)
                    '''
                    try:
                        # Create a buffer around the polygon
                        buffered_polygon = polygon.buffer(0)

                        # Get the exterior of the buffered polygon
                        exterior = buffered_polygon.exterior

                        # Get the interior of the original polygon
                        interior = polygon.interiors[0]

                        # Use the difference method to remove the hole
                        points = exterior.difference(interior)
                    except:
                        # create a new QPolygonF object from the coordinates of the polygon
                        points = polygon.exterior.coords
                    qpoints = [(x,y) for x,y in points]
                    '''
                    qpoints = [(x,y) for x,y in polygon]
                    print(qpoints)
                    #qpoints = self.fix_polygon_with_hole(qpoints)
                    #qpoints = [self.left_widgets['Image'].mapFromScene(QPoint(p[0], p[1])) for p in qpoints]
                    qpoints = [QPoint(p[0], p[1]) for p in qpoints]
                    #qpoints = fix_polygon_with_hole(qpoints)
                    self.left_widgets['Image'].polygon = QPolygonF(qpoints)
                    self.polygon_item = QGraphicsPolygonItem(self.left_widgets['Image'].polygon)
                    #self.polygon_item.setPen(self.pen_red)
                    self.polygon_item.setZValue(1)
                    self.left_widgets['Image'].scene.addItem(self.polygon_item)
                    self.polygon_item.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
                    #self.polygon_item.setAcceptedMouseButtons(Qt.RightButton)
                    #self.polygon_item.mousePressEvent = self.on_polygon_clicked
                    if self.left_widgets['Image'].mode == "segmentation":
                        self.pen_none = QPen(Qt.NoPen)
                        self.polygon_item.setPen(self.pen_none)
                        #i = self.main_window.get_current_selection('selected_label')
                        #self.polygon_item.setBrush(model.config.id2label[result['segments_info']['label_id']])
                        #print(result['segments_info'])
                        #print(model.config.id2label[result['segments_info'][i]['label_id']])
                        #print(categories_color[index].red())
                        #print(categories_color[index].green())
                        #print(categories_color[index].blue())
                        self.polygon_item.setBrush(categories_color[index])
                    if current_session == len(self.left_widgets['Image'].shape_items):
                        self.left_widgets['Image'].shape_items.append(self.polygon_item)
                    else:
                        self.left_widgets['Image'].shape_items[current_session] = self.polygon_item
                    self.polygon_item.hide()

                    self.left_widgets['Image'].store_coords_polygon()
                    #self.polygon_point_list.clear()
                    #self.polygon = None
                

    def yolo_helper(self, image, yolo, width, height, classes, output_layers, frame):

        self.session_data_auto = pd.DataFrame(columns=['label','confidence','x','y','w','h'])

        frame_name = self.images[self.photo_row].split('/')[-1].replace('temp-', '')
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blob)

        start_time_detection = time.time()  # timer starts for detection

        outputs = yolo.forward(output_layers)

        end_time_detection = time.time()
        # timer stops after detecting the objects
        detection_time_taken = end_time_detection - start_time_detection
        print("Time taken for detections: ", f"{detection_time_taken :.5f}", "ms")

        class_ids, boxes, confidences = [], [], []
        labels = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    labels.append(classes[class_id])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)



        num_objects_detected = len(indexes)

        print("Number of objects detected:", num_objects_detected)
        objects_detected = f'Number of objects detected: {num_objects_detected}'
        self.add_to_list(objects_detected,self.right_widgets['Statics Display'])

        if frame:
            frame_read = self.image_pixels[self.photo_row]
        else:
            frame_read = self.video_frame_pixels[self.photo_row]
        print(type(frame_read))
        to_draw = frame_read

        label_counts = {}

        for i in range(len(boxes)):
            if i in indexes:
                box = boxes[i]
                x, y = box[0], box[1]
                w, h = box[2], box[3]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Update label count
                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1

                cv2.rectangle(to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(to_draw, (x, y), (x + 15, y + 15), (0, 255, 0), -1)

                cv2.putText(to_draw, label[0], (x, y + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2,cv2.LINE_AA)



                rectangle_coordinates = [x, y, w, h]


                print(label,f"{confidence:.5f}" ,x,y,w,h)

                self.annotation_id += 1
                a_id = self.annotation_id

                if '"' in label:
                    label = label.replace('"', '')

                data = [[label,f"{confidence:.5f}" ,x,y,w,h]]
                to_add = pd.DataFrame(data, columns=self.session_data_auto.columns)
                self.session_data_auto = pd.concat([self.session_data_auto, to_add], ignore_index=True)
                # self.session_data_auto['label'] = self.session_data_auto['label'].apply(lambda x: x.replace('"', ''))
                self.add_to_list(f'{data}', self.right_widgets['Image Label List'])

                # directory_path = r"..\\..\\tesar\\Object-Detection-Metrics-master\\samples\\sample_2\\detections\\"
                # file_name = frame_name.split('.')[0]  # remove the file extension
                # location = os.path.join(directory_path, file_name + '.txt')
                # self.session_data_auto.to_csv(location, header=False, sep=' ', index=False)
        #self.left_widgets['Image'].
        # subprocess.run(['python', r"..\\..\\tesar\\Object-Detection-Metrics-master\\samples\\sample_2\\sample_2.py"])


        # dialog = QFileDialog()
        # directory_path = r"..\\..\\tesar\\saved_annotations\\"
        # file_name = frame_name
        # location, _ = dialog.getSaveFileName(self, "Save as", directory_path + file_name,
        #                                      "All Files (*);;Json Files (*.json);;Csv Files (*.csv);;Txt Files (*.txt)")
        # # self.session_data_auto.to_csv(location, header=False, index=False)
        # self.session_data_auto.to_csv(location, header=False, sep=' ', index=False)

        # Print label counts
        for label, count in label_counts.items():
            label_text = f'{label}: {count}'
            self.add_to_list(label_text, self.right_widgets['Statics Display'])
            self.add_to_list(label, self.right_widgets['Labels'])

        self.left_widgets['Image'].switch_frame(to_draw)

    def switch_editor(self, image_area):
        """
        Switch between the display/edit interfaces.
        Args:
            image_area: RegularImageArea or ImageEditorArea object.

        Return:
            None
        """

        temp_zoom = self.left_widgets['Image'].zoomStack
        temp_rois = self.left_widgets['Image'].ROIs 
        temp_image = self.left_widgets['Image']._image
        temp_image_file_name = self.left_widgets['Image'].image_file_name
        temp_scene = self.left_widgets['Image'].scene
        temp_zoom2 = self.left_widgets['Image'].transform().m11()
        self.left_layout.removeWidget(self.left_widgets['Image'])
        self.left_widgets['Image'] = image_area(temp_image, self, temp_zoom, temp_rois, temp_image_file_name,temp_scene,temp_zoom2)
        self.left_layout.addWidget(self.left_widgets['Image'])
        self.left_widgets['Image']._image = None
        self.left_widgets['Image'].setImageFileName(temp_image_file_name)
        self.left_widgets['Image'].updateViewer()

        '''
        # set up a...
        temp_widget = self.left_widgets['Image']
        new_widget = image_area(temp_widget._image, self)
        for attr, value in vars(temp_widget).items():
            setattr(new_widget, attr, value)
        self.left_layout.removeWidget(self.left_widgets['Image'])
        self.left_layout.addWidget(new_widget)
        self.left_widgets['Image'] = new_widget
        temp_widget.deleteLater()
        '''

    def edit_mode(self):
        """
        Switch between the display/edit interfaces.

        Return:
            None
        """
        # if nothing prompt the user to add file
        if self.photo_flag or self.video_flag:
            if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
                self.setWindowTitle('TESAR(Editor Mode Rectangle)')
                #self.switch_editor(ImageEditorArea)
                self.left_widgets['Image'].setMode('rectangle')
            else:
                self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
                self.left_widgets['Image'].setMode('none')
                #self.switch_editor(RegularImageArea)
            self.display_selection()
            #self.initial_display()
        else:
            self.warn_user_to_upload_file()

    def draw_rectangle(self):
        self.edit_mode()

    def draw_polygon(self):
        if self.video_flag or self.photo_flag:
            if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
                self.setWindowTitle('TESAR(Editor Mode Polygons)')
                #self.switch_editor(PolygonImageEditor)
                self.left_widgets['Image'].setMode('polygon')
            else:
                self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
                #self.switch_editor(RegularImageArea)
                self.left_widgets['Image'].setMode('none')
            self.display_selection()
        else:
            self.warn_user_to_upload_file()
            return

    def draw_cube(self):
        if self.video_flag or self.photo_flag:
            if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
                self.setWindowTitle('TESAR(Editor Mode Cube)')
                #self.switch_editor(CubeEditorArea)
                self.left_widgets['Image'].setMode('cube')
            else:
                self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
                #self.switch_editor(RegularImageArea)
                self.left_widgets['Image'].setMode('none')
            self.display_selection()
        else:
            self.warn_user_to_upload_file()
            return

    def draw_polyline(self):
        if self.video_flag or self.photo_flag:
            if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
                self.setWindowTitle('TESAR(Editor Mode Polyline)')
                #self.switch_editor(PolyLineImageEditor)
                self.left_widgets['Image'].setMode('polyline')
            else:
                self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
                #self.switch_editor(RegularImageArea)
                self.left_widgets['Image'].setMode('none')
            self.display_selection()
        else:
            self.warn_user_to_upload_file()
            return
    def pixmap2cv(self, pixmap):
        '''  Converts a QImage into an opencv MAT format  '''
        pixmap= pixmap.pixmap()
        incomingImage = pixmap.toImage()
        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr

    def apply_blur(self):
        print("====================")
        print(self.photo_flag,self.video_flag)
        #if not self.photo_flag or not self.video_flag:
        if not self.left_widgets['Image'].hasImage():
            self.warn_user_to_upload_file()
            return
        print(type(self.left_widgets['Image']._image))
        for i in range(self.right_widgets['Image List'].count()):
            img_name = self.images[i]
            labels = self.session_data.loc[self.session_data['Image'] == img_name].values
            # print("length",len(labels))
            if len(labels) != 0:
                if self.photo_flag:
                    #frame_read = self.image_pixels[i]
                    frame_read = self.pixmap2cv(self.left_widgets['Image']._image)
                    self.to_blur = frame_read.copy()
                    #blurred = self.labels_to_blur(labels)
                    blurred = self.labels_to_blur2(labels, frame_read)
                    type(blurred)
                    # cv2.imwrite(str(self.dir_folder_name) + '/temp-blur-'+ str(img_name),blurred)
                elif self.video_flag:
                    #frame_read = self.video_frame_pixels[i]
                    frame_read = self.pixmap2cv(self.left_widgets['Image']._image)
                    self.to_blur = frame_read.copy()
                    #blurred = self.labels_to_blur(labels)
                    blurred = self.labels_to_blur2(labels, frame_read)
                    type(blurred)
                    # cv2.imwrite(str(self.dir_folder_name) + '/temp-blur-'+ str(img_name) + '.png',blurred)
                if i == self.photo_row:
                    #blurred = cv2.convertScaleAbs(blurred, alpha=(255.0/65535.0))
                    self.left_widgets['Image'].switch_frame(blurred)
            else:
                pass


    def labels_to_blur2(self, labels, img):
        # Create a mask with the same dimensions as the image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)*255

        for label in labels:
            print("label:", label)
            coords = label[5]
            object_type = label[4]
            print("coords", coords)
            print("obj", object_type)

            if object_type == "Rectangle":
                print("Blur Rectangle")
                x = int(coords[0][0])
                y = int(coords[0][1])
                w = int(coords[0][2])
                h = int(coords[0][3])
                '''
                print("1",x,y,w,h)
                point = self.left_widgets['Image'].mapFromScene(QPoint(x,y))
                size = self.left_widgets['Image'].mapFromScene(QPoint(w,h))
                transform = self.left_widgets['Image'].viewportTransform()
                point_in_image = transform.map(point)
                print("1.5",point_in_image)
                x,y,w,h = int(point.x()), int(point.y()), int(size.x()), int(size.y())
                print("2",x,y,w,h)
                scene_rect = self.left_widgets['Image'].sceneRect()
                p = scene_rect.x()
                q = scene_rect.y()
                width = scene_rect.width()
                height = scene_rect.height()
                print("3",p,q,width,height)
                print("4",img.shape)
                '''
                try:
                    ROI = img[y:y+h, x:x+w]
                    # apply guassian blur on src image
                    dst = cv2.GaussianBlur(ROI,(51,51),cv2.BORDER_DEFAULT)
                    img[y:y+h, x:x+w] = dst
                except:
                    pass
            elif object_type == "Polygon":
                print("Blur Polygon")
                coords_polygon = coords
                polygon_pt_len = len(coords_polygon)
                roi_corners = np.array([coords],dtype = np.int32)
                blurred_image = cv2.GaussianBlur(img,(51, 51), cv2.BORDER_DEFAULT)
                mask = np.zeros(img.shape, dtype=np.uint8)
                channel_count = img.shape[2]
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
                img = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(img, mask_inverse)
        return img


    def labels_to_blur(self, labels):
        print("Blurring...")
        for label in labels:
            print("label:", label)
            coords = label[5]
            object_type = label[4]
            print("coords", coords)
            print("obj", object_type)

            if object_type == "Polygon":
                print("Blur Polygon")
                coords_polygon = coords
                polygon_pt_len = len(coords_polygon)
                print(polygon_pt_len)
                src = self.to_blur
                blur_src = cv2.GaussianBlur(src, (19, 19), 19)
                roi_array = []
                loopvar_2 = 0
                print(coords_polygon)
                while loopvar_2 < polygon_pt_len:
                    x = coords_polygon[loopvar_2][0]
                    y = coords_polygon[loopvar_2][1]
                    array_coords = (x, y)
                    roi_array.append(array_coords)
                    loopvar_2 += 1
                print(roi_array)
                roi_array = np.array([roi_array], dtype=np.int32)
                count = src.shape[2]
                print("c", count)
                mask = np.zeros(blur_src.shape, dtype=np.uint8)
                ignore_mask_color = (255,) * count
                cv2.fillPoly(mask, roi_array, ignore_mask_color)
                mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                self.to_blur = cv2.bitwise_and(blur_src, mask) + cv2.bitwise_and(src, mask_inverse)

            if object_type == "Rectangle":
                print("Blur Rectangle")
                x = int(coords[0][0])
                y = int(coords[0][1])
                w = int(coords[0][2])
                h = int(coords[0][3])
                print(x, y, w, h)
                i_tl = (x, y)
                i_tr = (x + w, y)
                i_br = (x + w, y + h)
                i_bl = (x, y + h)
                roi = [i_tl, i_tr, i_br, i_bl]
                print(roi)
                self.to_blur = self.BlurContours(self.to_blur, [[np.array(roi)]], 19, 19)
            else:
                print("None")
        return self.to_blur

    def BlurContours(self, image, contours, ksize, sigmaX, *args):
        sigmaY = args[0] if len(args) > 0 else sigmaX
        mask = np.zeros(image.shape[:2])
        for i, contour in enumerate(contours):
            cv2.drawContours(mask, contour, i, 255, -1)
        blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX, None, sigmaY)
        result = np.copy(image)
        alpha = mask / 255.
        result = alpha[:, :, None] * blurred_image + (1 - alpha)[:, :, None] * result
        return result

    def export_work(self):
        if not self.video_flag:
            return
        dialog = QFileDialog()
        pathOut, _ = dialog.getSaveFileName(self, 'Export Video')

        fps = 30.0
        out = cv2.VideoWriter(pathOut, self.video_codec, self.video_fps, self.video_size)
        # out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, self.video_size)
        # out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'IYUV'), self.video_fps, self.video_size)
        final_array = [0] * self.video_framecount

        for i in range(self.right_widgets['Image List'].count()):
            img_name = self.images[i]
            labels = self.session_data.loc[self.session_data['Image'] == img_name].values
            if len(labels) != 0:
                frame_read = self.video_frame_pixels[i]
                self.to_blur = frame_read.copy()
                blurred = self.labels_to_blur(labels)
                final_array[i] = blurred
            else:
                final_array[i] = self.video_frame_pixels[i]

        for i in range(len(final_array)):
            out.write(np.uint8(final_array[i]))
            # out.write(img_as_ubyte(final_array[i]))
        out.release()

    def delete_selected_contour(self, idx = None):
        """
        Deleting the selected bounding box.

        Return:
            None
        """
        global rectangle_highlight, polygon_highlight, polyline_highlight, cube_highlight
        global current_session
        self.session_data.sort_values(by=['ID', 'Image'], inplace=True)
        self.session_data.reset_index(drop=True, inplace=True)

        img_name = self.images[self.photo_row]

        # ind = index of all the annotation done on the current image
        # ind = self.session_data.loc[self.session_data['Image'] == img_name].index

        # print("ind ",self.session_data['ID'][0])
        if idx == None:
            idx = edit_highlight[-1]
        print('index', idx)
        for i in self.session_data['ID']:
            if i == idx:
                # if self.session_data['ID']
                # drop = self.session_data['ID'][highlight[0]-1]

                # self.right_widgets['Image Label List'].takeItem(index)
                # self.session_data.drop(index=drop, inplace=True)
                df = self.session_data
                delete_index = df.loc[df['ID'] == idx].index
                if self.session_data['Image'][delete_index].item() == img_name:
                    self.right_widgets['Image Label List'].takeItem(delete_index.values.astype(int)[0])
                    print("delete index", delete_index.values.astype(int))
                    df.drop(delete_index, inplace=True)
                    self.session_data['ID'] = np.arange(len(self.session_data['ID']))
                    self.session_data.reset_index()
                    for j in range(delete_index.values.astype(int)[0],self.right_widgets['Image Label List'].count()):
                        session_name = str((self.right_widgets['Image Label List'].item(j).text()))
                        print(session_name)
                        session_name = ast.literal_eval(session_name)
                        session_name[0][0] = j
                        new_session_name = str(session_name[:])
                        print(new_session_name)
                        self.change_item_index(new_session_name,self.right_widgets['Image Label List'],j)

                    current_session -= 1
                    #self.left_widgets['Image'].show_annotations(self.left_widgets['Image'].row)
                    #self.left_widgets['Image'].main_window.get_sub_set(self.left_widgets['Image'].main_window.right_widgets['Image Label List'],current_session)
                    #self.left_widgets['Image'].main_window.takeItem(delete_index.values.astype(int)[0])
                    box = self.left_widgets['Image'].shape_items[idx]
                    if type(box) is tuple:
                        for i in range(4):
                            box[i].hide()
                    else:
                        box.hide()
                        print("hide box index",idx)
                    del self.left_widgets['Image'].shape_items[idx]
        polygon_highlight = -1
        rectangle_highlight = -1
        polyline_highlight = -1
        cube_highlight = -1
        edit_highlight.clear()
        self.left_widgets['Image'].show_annotations(self.photo_row)
        print("session data after contour deletion\n", self.session_data)

    def save_changes(self):
        """
        Save the data in self.session_data to new/existing csv or json format in tesar/saved_annotations folder.

        Return:
            None
        """

        if self.right_widgets['Image Label List'].count() > 0:
            dialog = QFileDialog()
            location, _ = dialog.getSaveFileName(self, "Save as", r"..\\..\\tesar\\saved_annotations\\","All Files (*);;Json Files (*.json);;Csv Files (*.csv)")
            self.label_file = location
            self.session_data['Image'] = pd.Categorical(self.session_data['Image'], categories=self.images,
                                                                 ordered=True)
            self.session_data = self.session_data.sort_values('Image')
            if location.endswith('.json'):
                result = self.session_data.to_json(location, orient="records")
                parsed = json.loads(result)
                json.dumps(parsed, indent=4)

            elif location.endswith('.csv'):
                self.session_data.to_csv(location, index=False)

            self.statusBar().showMessage(f'Labels Saved to {location}')

        elif self.right_widgets['Image List'].count() == 0:
            msg = QMessageBox()
            msg.setStyleSheet("QLabel{min-height:40px; font-size: 13px;} QPushButton{ width:17px; font-size: 12px; }");
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No Image detected !! \nPlease Upload Image and perform annotation.")
            msg.setWindowTitle("No Image found")
            msg.exec_()

        elif self.right_widgets['Image Label List'].count() == 0:
            msg = QMessageBox()
            msg.setStyleSheet("QLabel{min-height:40px; font-size: 12px;} QPushButton{ width:15px; font-size: 11px; }");
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No annotation detected !! \nPlease perform annotation and Save.")
            msg.setWindowTitle("No annotation found")
            msg.exec_()

    def delete_element(self):
        """
        To delete from Image Label List

        Return:
            None
        """

        widget_list = self.right_widgets['Image Label List']
        items = [widget_list.item(i) for i in range(widget_list.count())]
        checked_indexes = [checked_index for checked_index, item in enumerate(items)
                           if item.checkState() == Qt.Checked]
        self.delete_selected_element(checked_indexes, widget_list)
        if len(checked_indexes) > 0:
            self.left_widgets['Image'].show_annotations(self.photo_row)

    def undo_state(self):
        global current_session
        print("UNDO",current_session)
        if current_session != 0:
            current_session -= 1
            self.left_widgets['Image'].show_annotations(self.left_widgets['Image'].row)
            self.left_widgets['Image'].main_window.get_sub_set(self.left_widgets['Image'].main_window.right_widgets['Image Label List'],current_session)
            box = self.left_widgets['Image'].shape_items[current_session]
            if type(box) is tuple:
                for i in range(4):
                    box[i].hide()
            else:
                box.hide()
            
        print("BECOME",current_session)

    def redo_state(self):
        global current_session
        print("REDO")
        if current_session != self.left_widgets['Image'].main_window.session_data.shape[0]:
            current_session += 1
            data = [self.left_widgets['Image'].main_window.session_data.iloc[current_session-1].tolist()]
            self.left_widgets['Image'].main_window.add_to_list(f'{data}',self.left_widgets['Image'].main_window.right_widgets['Image Label List'])
            self.left_widgets['Image'].show_annotations(self.left_widgets['Image'].row)
            box = self.left_widgets['Image'].shape_items[current_session-1]
            if type(box) is tuple:
                for i in range(4):
                    box[i].show()
            else:
                box.show()
    

    def delete_selected_element(self, checked_indexes, widget_list):
        if widget_list == self.right_widgets['Image Label List']:
            """
                Deletes form widget_list as well as from session data
            """
            self.session_data.sort_values('Image', inplace=True)
            self.session_data.reset_index(drop=True, inplace=True)

            img_name = self.images[self.photo_row]
            ind = self.session_data.loc[self.session_data['Image'] == img_name].index

            drop = []
            for q_list_index in reversed(checked_indexes):
                drop.append(ind[0] + q_list_index)
                widget_list.takeItem(q_list_index)
            self.session_data.drop(index=drop, inplace=True)
        else:
            pass

    def go_to_previous(self):
        """
        Switch to previous image in photo list.

        Return:
            None
        """

        previous_image = self.photo_row - 1
        if previous_image >= 0:
            self.choose_display_frame(previous_image)
        x = self.slider2.width()
        print(x)
        print(self.video_framecount)
        percentage = 1 / self.video_framecount
        movement = x * percentage
        self.slidershift = self.slidershift - movement
        self.slider2.setValue(self.slidershift)

    def go_to_next(self):
        """
        Switch to next image in photo list.

        Return:
            None
        """

        next_image = self.photo_row + 1
        if next_image >= 0:
            if next_image > (self.right_widgets['Image List'].count() - 1):
                return
            self.choose_display_frame(next_image)
        x = self.slider2.width()
        y = self.video_framecount
        percentage = 1 / y
        movement = x * percentage
        self.slidershift = self.slidershift + movement
        self.slider2.setValue(self.slidershift)

    def play(self):
        self.frameplayer = False
        if self.photo_row >= self.video_framecount:
            self.photo_row = 0
            self.slider2.setValue(0)
            self.frametimer.stop()
            self.frameplayer = False
            return
        if not self.actionPlay_Pause.isChecked():
            print("pause")
            self.frametimer.stop()
            self.frameplayer = False
            return
        self.frametimer = QTimer(self)
        self.frametimer.setTimerType(Qt.PreciseTimer)
        self.frametimer.timeout.connect(self.go_to_next)
        self.frametimer.start()

    def play_pause(self):
        self.frametimer.stop()

    def set_range(self):
        """
        Set the value range of the slider.

        Return:
            None
        """
        if self.right_widgets['Image List'].count() > 0:
            self.slider2.setMinimum(0)
            self.slider2.setMaximum(len(self.images) - 1)
        else:
            self.slider2.setRange(0, 100)

    def set_position(self, value):
        """
        Change display image based on slider value.
        Args:
            value: Slider position value.

        Return:
            None
        """
        if self.right_widgets['Image List'].count() >= 0:
            self.choose_display_frame(value)
            print(value)
            self.photo_row = value
    def zoom_image(self, factor):
        """
        Change display image based on zoom factor.
        Args:
            factor: Zoom factor.

        Return:
            None
        """
        self.value = factor
        self.scaleFactor = factor
        global res_width, res_height, scale_width, scale_ht
        global img_w, img_h, win_width, win_ht
        print("zoom image",win_width, win_ht, img_w, img_h,factor)
        #scale_width = int(self.scaleFactor * (res_width))
        #scale_ht = int(self.scaleFactor * (res_height))
        img_w = int(self.scaleFactor*img_w)
        img_h = int(self.scaleFactor*img_h)

        self.choose_display_frame(self.photo_row)
    '''
    def zoom_image_modified(self, factor):
        """
        Change display image based on zoom factor.
        Args:
            factor: Zoom factor.

        Return:
            None
        """
        self.value = factor
        self.scaleFactor = factor
        global res_width, res_height, scale_width, scale_ht
        global img_w, img_h, win_width, win_ht
        print("zoom image",win_width, win_ht, img_w, img_h,factor)
        #scale_width = int(self.scaleFactor * (res_width))
        #scale_ht = int(self.scaleFactor * (res_height))
        img_w = int(self.scaleFactor*img_w)
        img_h = int(self.scaleFactor*img_h)

        global delta_x, delta_y, offset_x, offset_y
        print(win_width, img_w, offset_x, delta_x)
        img_x_pred = int(win_width / 2 - img_w / 2) + offset_x + delta_x  # Top left scaled image coordinates
        img_y_pred = int(win_ht / 2 - img_h / 2) + offset_y + delta_y
        print("img_pred",img_x_pred,img_x_pred)

        x_max = img_w+img_x_pred
        y_max = img_h+img_y_pred
        if x_max < win_width:
            offset_x += win_width-x_max
        if y_max < win_ht:
            offset_y += win_ht-y_max
        x_min = img_x_pred
        y_min = img_y_pred
        if x_min > 0:
            offset_x += -x_min
        if y_min > 0:
            offset_y += -y_min
        print("OFFSET",offset_x,offset_y)
        self.choose_display_frame(self.photo_row)
    '''

    def zoom_image_modified_roi(self, factor, center=None):
        """
        Change display image based on zoom factor.
        Args:
            factor: Zoom factor.

        Return:
            None
        """
        self.value = factor
        self.scaleFactor = factor
        global res_width, res_height, scale_width, scale_ht
        global img_w, img_h, win_width, win_ht
        img_w = int(self.scaleFactor*img_w)
        img_h = int(self.scaleFactor*img_h)

        global delta_x, delta_y, offset_x, offset_y
        print("CCEENNTTEERR",center)
        if center != None:
            d_x = int((self.scaleFactor-1)*((center.x())-img_x))
            d_y = int((self.scaleFactor-1)*((center.y())-img_y))
            img_x_temp = img_x - d_x
            img_y_temp = img_y - d_y

            offset_x = img_x_temp-int(win_width/2-img_w/2)
            offset_y = img_y_temp-int(win_ht/2-img_h/2)

        img_x_pred = int(win_width / 2 - img_w / 2) + offset_x + delta_x  # Top left scaled image coordinates
        img_y_pred = int(win_ht / 2 - img_h / 2) + offset_y + delta_y

        x_max = img_w+img_x_pred
        y_max = img_h+img_y_pred
        #if x_max < win_width:
        #    offset_x += win_width-x_max
        #if y_max < win_ht:
        #    offset_y += win_ht-y_max
        x_min = img_x_pred
        y_min = img_y_pred
        #if x_min > 0:
        #    offset_x += -x_min
        #if y_min > 0:
        #    offset_y += -y_min
        print("OFFSET",offset_x,offset_y)
        self.choose_display_frame(self.photo_row)

    def zoom_image_modified(self, factor, center=None):
        """
        Change display image based on zoom factor.
        Args:
            factor: Zoom factor.

        Return:
            None
        """
        self.value = factor
        self.scaleFactor = factor
        global res_width, res_height, scale_width, scale_ht
        global img_w, img_h, win_width, win_ht
        img_w = int(self.scaleFactor*img_w)
        img_h = int(self.scaleFactor*img_h)

        global delta_x, delta_y, offset_x, offset_y
        print("CCEENNTTEERR",center)
        if center != None:
            d_x = int((self.scaleFactor-1)*((center.x())-img_x))
            d_y = int((self.scaleFactor-1)*((center.y())-img_y))
            img_x_temp = img_x - d_x
            img_y_temp = img_y - d_y

            offset_x = img_x_temp-int(win_width/2-img_w/2)
            offset_y = img_y_temp-int(win_ht/2-img_h/2)

        img_x_pred = int(win_width / 2 - img_w / 2) + offset_x + delta_x  # Top left scaled image coordinates
        img_y_pred = int(win_ht / 2 - img_h / 2) + offset_y + delta_y

        x_max = img_w+img_x_pred
        y_max = img_h+img_y_pred
        if x_max < win_width:
            offset_x += win_width-x_max
        if y_max < win_ht:
            offset_y += win_ht-y_max
        x_min = img_x_pred
        y_min = img_y_pred
        if x_min > 0:
            offset_x += -x_min
        if y_min > 0:
            offset_y += -y_min
        print("OFFSET",offset_x,offset_y)
        self.choose_display_frame(self.photo_row)

    def zoomIn(self,center = None):
        """
        Zoom in on image by 10%.

        Return:
            None
        """
        print("Zoom-IN")
        global img_w, img_h, win_width, win_ht
        if img_w != 0:
            if img_w >= win_width*10 or img_h >= win_ht*10:
                return
        self.zoom_image_modified(1.05, center)

    def zoomOut(self,center = None):
        """
        Zoom out of image by 10%.

        Return:
            None
        """
        print("Zoom-OUT")
        global img_w, img_h, win_width, win_ht
        if img_w != 0:
            if img_w <= win_width or img_h <= win_ht:
                return
        self.zoom_image_modified(0.95, center)

    def zoomInSmall(self):
        """
        Zoom in on image by 10%.

        Return:
            None
        """
        self.zoom_image(1.01)

    def zoomOutSmall(self):
        """
        Zoom out of image by 10%.

        Return:
            None
        """
        self.zoom_image(0.99)

    def warn_user_to_upload_file(self):
        msg = QMessageBox()
        msg.setStyleSheet("QLabel{min-height:25px; font-size: 12px;} QPushButton{ width:15px; font-size: 11px; }");
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Kindly upload a file !")
        msg.setWindowTitle("No file found")
        msg.exec_()

    def fit_image(self):
        """
        Fit the image to current window size

        Return:
            None
        """
        if self.video_flag or self.photo_flag:
            global delta_x, delta_y, offset_x, offset_y
            delta_x, delta_y, offset_x, offset_y = 0, 0, 0, 0

            global scale_width, scale_ht, win_width, win_ht
            global img_w, img_h
            if (img_w > win_width or img_h > win_ht):
                while (img_w > win_width and img_h > win_ht):
                    self.zoomOutSmall()
            elif (img_w < win_width or img_h < win_ht):
                while (img_w < win_width and img_h < win_ht):
                    self.zoomInSmall()
            # back to 100% view
            self.zoomSliderVal = 100
            print('original image -> fit to screen')
            self.zoomslider.setValue(self.zoomSliderVal)
        else:
            self.warn_user_to_upload_file()

    def reset_frame(self):
        """
        Restore original frame by removing all its annotations.

        Return:
            None
        """
        global delta_x, delta_y, offset_x, offset_y
        delta_x, delta_y, offset_x, offset_y = 0, 0, 0, 0
        img_name = self.images[self.photo_row].split('/')[-1].replace('temp-', '')

        ind = self.session_data.loc[self.session_data['Image'] == img_name].index
        self.session_data = self.session_data.drop(index=ind)

        self.right_widgets['Image Label List'].clear()
        for item in self.session_data.loc[self.session_data['Image'] == img_name].values:
            self.add_to_list(f'{[[x for x in item]]}', self.right_widgets['Image Label List'])

        self.choose_display_frame(self.photo_row)

    def choose_display_frame(self, value):
        """
        Change image/video frame to be displayed.
        Args:
            value: Row number of Photo list to be chosen.

        Return:
            None
        """
        self.right_widgets['Image List'].setCurrentRow(value)
        if not self.video_flag:
            self.current_image = self.image_pixels[value]
        else:
            self.current_image = self.video_frame_pixels[value]
        self.choose_display(value)

    def go_to_Model1(self):
        print("auto mode")
        text = "1080p_medium_mosaic"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model2(self):
        print("auto mode")
        text = "1080p_medium_rect"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model5(self):
        print("auto mode")
        text = "720p_medium_mosaic"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model6(self):
        print("auto mode")
        text = "720p_medium_rect"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model7(self):
        print("auto mode")
        text = "720p_small_mosaic"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model8(self):
        print("auto mode")
        text = "720p_small_rect"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def dashcamcleaner(self):
        global video_dir
        global save_area
        global blursize
        global detectionthresh
        global framememory
        global ROIEnlargement
        global outputquality
        global inferencesize

        self.filename_dir = os.path.split(self.main_dir)
        self.filename_dir = self.filename_dir[1]
        self.savearea = "..\\Results\\" + str(self.filename_dir)

        print("dashcamcleaner")

        video_dir = str(self.main_dir)

        print(video_dir)
        print(self.video_framecount)

        save_area = self.savearea
        detectionthresh = self.detection_thresh.value()
        blursize = self.blur_size.value()
        framememory = self.frame_memory.value()
        ROIEnlargement = self.ROI_enlargement.value()
        outputquality = self.output_quality.value()
        inferencesize = int(self.inference_size.currentText()[:-1]) * 16 / 9

        parameters = [video_dir, save_area, blursize, framememory, detectionthresh, ROIEnlargement, inferencesize,
                      outputquality]

        if self.blurrer:
            self.blurrer.parameters = parameters
            self.blurrer.run()
            print("Blurring started")
            self.displayChanges()
        else:
            print("Error")

    def setProgress(self, value: int):
        self.progress_bar.setValue(value)

    def go_to_Model3(self):
        print("auto mode")
        text = "720p_medium_mosaic"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Model4(self):
        print("auto mode")
        text = "1080p_small_rect"
        if self.windowTitle() == "Tata Elxsi Smart AnonymizeR (TESAR)":
            self.blurrer = VideoBlurrer(text, None)
            # self.blurrer.setMaximum.connect(self.setMaximumValue)
            self.blurrer.updateProgress.connect(self.progbar)
            self.blurrer.finished.connect(self.blurrer_finished)
            self.blurrer.alert.connect(self.alertuser)
            msg_box = QMessageBox()
            msg_box.setText("Successfully loaded: " + str(text) + ".pt")
            msg_box.exec_()
            self.toolbarconstructor()
        else:
            print("Switching off Auto Mode...")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()

    def go_to_Gaussian(self):
        pass

    def go_to_Median(self):
        pass

    def go_to_Bilateral(self):
        pass

    def go_to_SegManual(self):
        print("Segmentation Manual") 
        self.segmentation_flag = "manual"
        self.segmentation_mode()

    def go_to_SegAutoPolygon(self):
        print("Segmentation Auto Polygon") 
        self.segmentation_flag = "auto polygon"
        self.segmentation_mode()

    def go_to_SegAutoBinaryMask(self):
        print("Segmentation Auto Binary Mask") 
        self.segmentation_flag = "auto binary mask"
        self.segmentation_mode()

    def go_to_SegAutoBinaryMaskInstance(self):
        print("Instance Segmentation Auto Binary Mask") 
        self.segmentation_flag = "auto binary mask instance"
        self.segmentation_mode()

    def go_to_SegSemantic(self):
        print("Semantic Segmentation") 
        self.segmentation_flag = "semantic"
        self.segmentation_mode()

    def add_session_label(self, label=None):
        """
        Add label entered to the session labels list.

        Return:
            None
        """
        labels = self.right_widgets['Labels']
        new_label = label or self.top_right_widgets['Add Label'][0].text()
        session_labels = [str(labels.item(i).text()) for i in range(labels.count())]
        if new_label and new_label not in session_labels:
            self.add_to_list(new_label, labels)
            self.top_right_widgets['Add Label'][0].clear()

    def remove_temps(self):
        """
        Remove temporary image files from working directories.

        Return:
            None
        """
        if self.main_dir:
            working_dirs = set(['/'.join(self.main_dir.split('/')[:-1])])
            for working_dir in working_dirs:
                for file_name in os.listdir(working_dir):
                    if 'temp-' in file_name:
                        os.remove(f'{working_dir}/{file_name}')
            print("removed temporary files")

    def closeEvent(self, event):
        """
            Handles application closing event
        """
        if self.photo_flag or self.video_flag:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("TESAR")
            msgBox.setText("Current work will be lost, do you want to save it?")
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Close)
            msgBox.setDefaultButton(QMessageBox.Save)
            self.answer = msgBox.exec()

            if self.answer == QMessageBox.Close:
                self.remove_temps()
                event.accept()
                return

            elif self.answer == QMessageBox.Save:
                self.save_changes()
                event.ignore()

            elif self.answer == QMessageBox.Discard:
                event.ignore()

        else:
            print("No work to be saved! Exiting")

    def alertuser(self, message: str):
        print("alert user")
        msg_box = QMessageBox()
        msg_box.setText(message)
        msg_box.exec_()

    def abort_process(self):
        if self.savearea == None:
            print("...Turning off Automode")
            self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR)")
            self.toolbarnew.clear()
        else:
            if self.blurrer.isRunning():
                self.blurrer.terminate()
                self.blurrer.wait()
            self.progressbar.setValue(0)

    def setMaximumValue(self, value: int):
        print("setMaximum")
        self.progressbar.setMaximum(value)

    def progbar(self, value: int):
        print(value)
        percentchange = (value / (self.video_framecount))
        percentchange = percentchange * 100
        percentchange = int(percentchange)
        self.progressbar.setValue(percentchange)

    def blurrer_finished(self):
        """
        Create a new blurrer, setup UI and notify the user
        """
        msg_box = QMessageBox()
        if self.blurrer and self.blurrer.result["success"]:
            minutes = int(self.blurrer.result["elapsed_time"] // 60)
            seconds = round(self.blurrer.result["elapsed_time"] % 60)
            msg_box.setText(f"Video blurred successfully in {minutes} minutes and {seconds} seconds.")
        else:
            msg_box.setText("Blurring resulted in errors.")
        msg_box.exec_()
        if not self.blurrer:
            self.setup_blurrer()
        self.progressbar.setValue(0)

    def displayChanges(self):
        self.restore_variables()

        print(self.savearea)
        self.vid_change_dir = self.savearea

        split = len(self.vid_change_dir.split('/')[-1])
        self.dir_folder_name = self.vid_change_dir[0:-(split)]
        vid = cv2.VideoCapture(self.vid_change_dir)

        self.video_fps = round(vid.get(cv2.CAP_PROP_FPS))
        self.video_codec = int(vid.get(cv2.CAP_PROP_FOURCC))
        self.frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.video_framecount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        currentframe = 0

        while (True):
            ret, frame = vid.read()

            if ret:
                self.video_frame_pixels.append(frame)
                frame_name = "frame" + str(currentframe)
                self.add_to_list(frame_name, self.right_widgets['Image List'])
                self.images.append(f'{frame_name}')
                self.image_paths[frame_name] = 'video_frame_pixels' + '[' + str(currentframe) + ']'

                currentframe += 1
            else:
                break
        vid.release()

        global res_width, res_height, scale_width, scale_ht
        res_width = (self.video_size[0])
        res_height = (self.video_size[1])
        scale_width, scale_ht = res_width, res_height
        self.video_flag = True
        self.initial_display()
        msg_box = QMessageBox()
        msg_box.setText("Video Successfully Updated")
        msg_box.exec_()


    def keyPressEvent(self, event):
        print("PRESS")
        if event.key() == Qt.Key_Shift:
            self.setCursor(QCursor(Qt.CrossCursor))

    def keyReleaseEvent(self, event):
        print("RELEASE")
        if event.key() == Qt.Key_Shift:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def toolbarconstructor(self):
        print("Turning on Auto Mode...")
        self.setWindowTitle("Tata Elxsi Smart AnonymizeR (TESAR) - AUTO MODE")
        self.toolbarnew.clear()

        # Mid Spacer 7
        self.spacer7 = QWidget()
        self.spacer7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Label 1
        self.detection_thresh_label = QLabel("Detection Threshold: ", self)

        # Set Spinbox1
        self.detection_thresh = QDoubleSpinBox()
        self.detection_thresh.setMaximum(1.00000)
        self.detection_thresh.setSingleStep(0.05)
        self.detection_thresh.setValue(0.30000)

        # Label 2
        self.ROI_enlargement_label = QLabel("    ROI Enlargement: ", self)

        # Set Spinbox2
        self.ROI_enlargement = QDoubleSpinBox()
        self.ROI_enlargement.setMinimum(0.80000)
        self.ROI_enlargement.setMaximum(10.00000)
        self.ROI_enlargement.setSingleStep(0.05000)
        self.ROI_enlargement.setValue(1.00000)

        # Label 3
        self.blur_size_label = QLabel("    Blur Size: ", self)

        # Set Spinbox 3
        self.blur_size = QSpinBox()
        self.blur_size.setValue(9)

        # Label 4
        self.frame_memory_label = QLabel("    Frame Memory: ", self)

        # Set Spinbox 4
        self.frame_memory = QSpinBox()
        self.frame_memory.setValue(0)

        # Label 5
        self.output_quality_label = QLabel("    Output Quality: ", self)

        # Set Spinbox 5
        self.output_quality = QSpinBox()
        self.output_quality.setMaximum(10)
        self.output_quality.setValue(4)
        self.output_quality.setMinimum(1)

        # Label 6
        self.inference_size_label = QLabel("    Inference Size: ", self)

        # Combobox
        self.inference_size = QComboBox()
        self.inference_size.addItem("360p")
        self.inference_size.addItem("720p")
        self.inference_size.addItem("1080p")
        self.inference_size.setCurrentText("360p")

        # Mid Spacer 8
        self.spacelabel = QLabel("    ", self)

        # Anonymize
        self.submit = QPushButton("Execute")
        self.submit.clicked.connect(self.dashcamcleaner)

        self.spacelabel2 = QLabel("    Progress: ", self)

        # Force Quit
        self.quit = QPushButton("Abort")
        self.quit.clicked.connect(self.abort_process)

        self.spacelabel3 = QLabel("    ", self)
        self.spacelabel4 = QLabel("    ", self)
        self.spacelabel5 = QLabel("    Framecount: " + str(self.video_framecount), self)

        # Progress Bar
        self.progressbar = QProgressBar()
        self.progressbar.setEnabled(True)
        self.progressbar.setValue(0)
        # self.progressbar.setMaximum(self.video_framecount)

        # Mid Spacer 9
        self.spacer8 = QWidget()
        self.spacer8.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Adding Widgets/Actions on the toolbar
        self.toolbarnew.addWidget(self.spacer7)
        self.toolbarnew.addWidget(self.detection_thresh_label)
        self.toolbarnew.addWidget(self.detection_thresh)
        self.toolbarnew.addWidget(self.ROI_enlargement_label)
        self.toolbarnew.addWidget(self.ROI_enlargement)
        self.toolbarnew.addWidget(self.blur_size_label)
        self.toolbarnew.addWidget(self.blur_size)
        self.toolbarnew.addWidget(self.frame_memory_label)
        self.toolbarnew.addWidget(self.frame_memory)
        self.toolbarnew.addWidget(self.output_quality_label)
        self.toolbarnew.addWidget(self.output_quality)
        self.toolbarnew.addWidget(self.inference_size_label)
        self.toolbarnew.addWidget(self.inference_size)
        self.toolbarnew.addWidget(self.spacelabel)
        self.toolbarnew.addWidget(self.spacelabel2)
        self.toolbarnew.addWidget(self.progressbar)
        self.toolbarnew.addWidget(self.spacelabel3)
        self.toolbarnew.addWidget(self.submit)
        self.toolbarnew.addWidget(self.spacelabel4)
        self.toolbarnew.addWidget(self.quit)
        self.toolbarnew.addWidget(self.spacelabel5)
        self.toolbarnew.addWidget(self.spacer8)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
