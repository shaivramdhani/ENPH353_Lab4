#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()

        # Load GUI created in Qt Designer
        loadUi("./SIFT_app.ui", self)

        # ---------------- CAMERA SETUP ----------------

        self._cam_id = 0
        self._cam_fps = 10

        self._is_cam_enabled = False
        self._is_template_loaded = False

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # ---------------- GUI SIGNALS ----------------

        self.browse_button.clicked.connect(
            self.SLOT_browse_button
        )

        self.toggle_cam_button.clicked.connect(
            self.SLOT_toggle_camera
        )

        # ---------------- CAMERA TIMER ----------------

        self._timer = QtCore.QTimer(self)

        self._timer.timeout.connect(
            self.SLOT_query_camera
        )

        self._timer.setInterval(
            int(1000 / self._cam_fps)
        )

        # ---------------- SIFT SETUP ----------------

        self.sift = cv2.SIFT_create()

        # FLANN matcher configuration
        FLANN_INDEX_KDTREE = 1

        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5
        )

        search_params = dict(
            checks=50
        )

        self.flann = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

        # Template information
        self.template_img = None
        self.template_keypoints = None
        self.template_descriptors = None


    # =================================================
    # TEMPLATE BUTTON
    # =================================================

    def SLOT_browse_button(self):

        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():

            self.template_path = dlg.selectedFiles()[0]

            # Display template in Qt
            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)

            # Load grayscale version for SIFT
            self.template_img = cv2.imread(
                self.template_path,
                cv2.IMREAD_GRAYSCALE
            )

            # Calculate template SIFT features
            (
                self.template_keypoints,
                self.template_descriptors
            ) = self.sift.detectAndCompute(
                self.template_img,
                None
            )

            self._is_template_loaded = True

            print(
                "Loaded template with",
                len(self.template_keypoints),
                "keypoints"
            )


    # =================================================
    # SIFT TRACKER
    # =================================================

    def track_template(self, frame):

        # Convert camera frame to grayscale
        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Find SIFT features in camera frame
        frame_keypoints, frame_descriptors = \
            self.sift.detectAndCompute(
                gray_frame,
                None
            )

        # No features found
        if frame_descriptors is None:
            return frame

        # Find two nearest matches for each template feature
        matches = self.flann.knnMatch(
            self.template_descriptors,
            frame_descriptors,
            k=2
        )

        # Lowe ratio test
        good_matches = []

        for m, n in matches:

            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Need enough matches for reliable homography
        if len(good_matches) < 10:
            return frame

        # Coordinates in template
        src_pts = np.float32([
            self.template_keypoints[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        # Corresponding coordinates in camera frame
        dst_pts = np.float32([
            frame_keypoints[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        # Estimate homography while rejecting outliers
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if H is None:
            return frame

        # Get template dimensions
        h, w = self.template_img.shape

        # Four corners of original template
        template_corners = np.float32([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ]).reshape(-1, 1, 2)

        # Determine where those corners appear in camera
        projected_corners = cv2.perspectiveTransform(
            template_corners,
            H
        )

        # Draw detected object
        cv2.polylines(
            frame,
            [np.int32(projected_corners)],
            True,
            (0, 255, 0),
            3
        )

        return frame


    # =================================================
    # CAMERA CALLBACK
    # =================================================

    def SLOT_query_camera(self):

        ret, frame = self._camera_device.read()

        if not ret:
            return

        # Run SIFT tracker if template exists
        if self._is_template_loaded:
            frame = self.track_template(frame)

        pixmap = self.convert_cv_to_pixmap(frame)

        self.live_image_label.setPixmap(pixmap)


    # =================================================
    # OPENCV -> QT IMAGE
    # =================================================

    def convert_cv_to_pixmap(self, cv_img):

        cv_img = cv2.cvtColor(
            cv_img,
            cv2.COLOR_BGR2RGB
        )

        height, width, channel = cv_img.shape

        bytesPerLine = channel * width

        q_img = QtGui.QImage(
            cv_img.data,
            width,
            height,
            bytesPerLine,
            QtGui.QImage.Format_RGB888
        )

        return QtGui.QPixmap.fromImage(q_img)


    # =================================================
    # CAMERA BUTTON
    # =================================================

    def SLOT_toggle_camera(self):

        if self._is_cam_enabled:

            self._timer.stop()

            self._is_cam_enabled = False

            self.toggle_cam_button.setText(
                "&Enable camera"
            )

        else:

            self._timer.start()

            self._is_cam_enabled = True

            self.toggle_cam_button.setText(
                "&Disable camera"
            )


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    myApp = My_App()

    myApp.show()

    sys.exit(app.exec_())