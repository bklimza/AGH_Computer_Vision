import numpy as np
from dataclasses import dataclass
from typing import Any
import cv2
import imutils


@dataclass
class Camera:
    rotation_vector: Any
    translation_vector: Any
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    @staticmethod
    def estimate_pose(image_size,
                      points_2d: np.ndarray,
                      points_3d: np.ndarray):
        camera_matrix = Camera._compute_camera_matrix(image_size)
        distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion

        print(points_3d)
        print(points_2d)

        _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            distortion_coefficients,
            iterationsCount=500,
            reprojectionError=2.0,
            confidence=0.95
        )

        return Camera(rotation_vector,
                      translation_vector,
                      camera_matrix,
                      distortion_coefficients)

    @staticmethod
    def _compute_camera_matrix(image_size):
        focal_length = min(image_size[1], image_size[0])
        center = (image_size[1] / 2.0, image_size[0] / 2.0)
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ]).astype(float)

    def project_points(self, points_3d):
        projected, _ = cv2.projectPoints(points_3d,
                                         self.rotation_vector,
                                         self.translation_vector,
                                         self.camera_matrix,
                                         self.distortion_coefficients)

        return projected[:, 0, :]


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


def make_3d_plane(size):
    return np.array([
        [0, 0, 0],
        [size[1], 0, 0],
        [size[1], size[0], 0],
        [0, size[0], 0],
    ]).astype(float)


def draw_plane(image, plane):
    cv2.line(image, tuple(plane[0].astype(int)), tuple(plane[1].astype(int)), RED, 2)
    cv2.line(image, tuple(plane[1].astype(int)), tuple(plane[2].astype(int)), RED, 2)
    cv2.line(image, tuple(plane[2].astype(int)), tuple(plane[3].astype(int)), RED, 2)
    cv2.line(image, tuple(plane[3].astype(int)), tuple(plane[0].astype(int)), RED, 2)


def make_3d_axis(size):
    return np.array([
        [0, 0, 0],
        [size, 0, 0],
        [0, size, 0],
        [0, 0, size],
    ]).astype(float)


def draw_axis(image, axis):
    cv2.line(image, tuple(axis[0].astype(int)), tuple(axis[1].astype(int)), BLUE, 4)
    cv2.line(image, tuple(axis[0].astype(int)), tuple(axis[2].astype(int)), GREEN, 4)
    cv2.line(image, tuple(axis[0].astype(int)), tuple(axis[3].astype(int)), RED, 4)


def draw_points(image, points):
    for point in points:
        # Center coordinates
        center_coordinates = (int(point[0]), int(point[1]))

        # Radius of circle
        radius = 10

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)


img1 = cv2.imread("./lab_3_image_1.png")
img2 = cv2.imread("./lab_3_image_2.png")
img1 = imutils.resize(img1, width=500)
img2 = imutils.resize(img2, width=500)
img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ========== BF + ORB ==========

orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)
cv2.imwrite('pose_estimation.png', img3)


# ========== BF + SIFT ==========

# sift = cv2.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
#
# # Apply ratio test
# good = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
#
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# cv2.imwrite('pose_estimation.png', img3)


# ========== SIFT+FLANN ==========

# sift = cv2.SIFT_create()
#
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
#
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=200)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#
# good_matches = []
# matches_mask = []
#
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.8 * n.distance:
#         good_matches.append(m)
#         matches_mask.append([1, 0])
#     else:
#         matches_mask.append([0, 0])
#
# matches_visualization = cv2.drawMatchesKnn(img1, keypoints1,
#                                            img2, keypoints2,
#                                            matches,
#                                            outImg=None,
#                                            matchColor=GREEN,
#                                            singlePointColor=BLUE,
#                                            matchesMask=matches_mask)
#
# cv2.imshow('Matches', matches_visualization)
# cv2.imwrite('matches.png', matches_visualization)

# ===SIFT pose===
# points_3d = np.float32([
#     (keypoints1[match.queryIdx].pt[0], keypoints1[match.queryIdx].pt[1], 0.0)
#     for match in good_matches]
# ).reshape(-1, 3)
# points_2d = np.float32([
#     keypoints2[match.trainIdx].pt
#     for match in good_matches
# ]).reshape(-1, 2)

# ===ORB pose===
points_3d = np.float32([
    (kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 0.0)
    for match in matches]
).reshape(-1, 3)
points_2d = np.float32([
    kp2[match.trainIdx].pt
    for match in matches
]).reshape(-1, 2)

camera = Camera.estimate_pose(img2_grey.shape, points_2d, points_3d)

print(f"Camera Matrix:\n {camera.camera_matrix}")
print(f"Rotation Vector:\n {camera.rotation_vector}")
print(f"Translation Vector:\n {camera.translation_vector}")

draw_points(img2, points_2d)
draw_plane(img2, camera.project_points(make_3d_plane(img1_grey.shape)))
draw_axis(img2, camera.project_points(make_3d_axis(100)))
cv2.imshow('Pose estimation', img2)
cv2.imwrite('pose_estimation.png', img2)
cv2.waitKey(0)
