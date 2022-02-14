import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path = "lab_3_image_1.png"
image = cv2.imread(f"./{image_path}")
image_size = image.shape

points_3d = np.array([
    (1200.0, 700.0, 0.0),
    (1130.0, 90.0, 0.0),
    (1350.0, 1700.0, 0.0),
    (85.0, 700.0, 0.0)
])

points_2d = np.array([
    (1200.0, 700.0),
    (1130.0, 90.0),
    (1350.0, 1700.0),
    (85.0, 700.0)
])

for p in points_2d:
    cv2.circle(image, (int(p[0]), int(p[1])), 30, (255, 0, 0), -1)

focal_length = image_size[1]
center = (image_size[1] / 2, image_size[0] / 2)
camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]]).astype(float)

distorsion_coefficients = np.zeros((4, 1))
_, rotation_vector, translation_vector = cv2.solvePnP(points_3d, points_2d, camera_matrix, distorsion_coefficients,
                                                                                        flags=cv2.SOLVEPNP_EPNP)


def draw_axis(rotation_vector, translation_vector, camera_matrix, distorsion_coefficients, image_points):
    end, _ = cv2.projectPoints(np.array([(0.0, 0.0, 200.0)]), rotation_vector, translation_vector, camera_matrix,
                                                                                            distorsion_coefficients)

    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255), 1)
    point1, point2 = (int(image_points[0][0]), int(image_points[0][1])), (int(end[0][0][0]), int(end[0][0][1]))
    cv2.line(image, point1, point2, (255, 0, 0), 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


draw_axis(rotation_vector, translation_vector, camera_matrix, distorsion_coefficients, points_2d)
plt.show()
