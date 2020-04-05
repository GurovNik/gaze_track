import cv2
import dlib
import numpy as np
import scipy
from keras.models import load_model
import operator
import math
from functools import reduce
import matplotlib.pyplot as plt
import imutils
from scipy import optimize

# from ref.ellipse_fitting import compute_guaranteedellipse_estimates

kernel = np.ones((3, 3), np.uint8)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def find(ret):
	cnts = cv2.findContours(ret.copy(), cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	c = cnts[0]
	((x, y), radius) = cv2.minEnclosingCircle(c)
	return int(x), int(y)


def chaikins_corner_cutting(coords, refinements=3):
	coords = np.array(coords)
	for _ in range(refinements):
		L = coords.repeat(2, axis=0)
		R = np.empty_like(L)
		R[0] = L[0]
		R[2::2] = L[1:-1:2]
		R[1:-1:2] = L[2::2]
		R[-1] = L[-1]
		coords = L * 0.75 + R * 0.25
	return coords


def order_points(coords):
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	return (sorted(coords, key=lambda coord: (-135 - math.degrees(
		math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))


class Eye:
	def __init__(self, landmarks, frame):
		self.frame = frame
		self.eye_points = self._extract_eye_points(landmarks)
		self.bbox = cv2.boundingRect(self.eye_points)
		self.eye_region, self.eye_origin = self._extract_eye_region(frame_padding=2)
		self.mask = self._make_mask()
		left, right = self._generate_masked_gradient()
		self.iris = iris(left, right, self.eye_region, self.mask)
		self.eye_corners = self._extract_eye_corners()
		self.center = self._calculate_center()
		# print(R)

	def _calculate_center(self):
		candidate_points = []
		if len(self.iris.left_gradient) > 0:
			candidate_points.extend(self.iris.left_gradient.transpose())
		if len(self.iris.right_gradient) > 0:
			candidate_points.extend(self.iris.right_gradient.transpose())
		candidate_points = np.array(candidate_points)

		def calc_R(xc, yc):
			x = candidate_points[:, 1]
			y = candidate_points[:, 0]
			""" calculate the distance of each 2D points from the center (xc, yc) """
			x = x - xc
			y = y - yc
			return np.power((np.power(x, 2) + np.power(y, 2)), 0.5)

		def f_2(c):
			global R
			""" calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
			Ri = calc_R(*c)
			R = np.mean(Ri)
			return Ri - R

		circle = optimize.least_squares(f_2, self.iris.rough_center, ftol=1e-4)
		center = circle['x'].astype(int)
		return center

	def _extract_eye_corners(self):
		left_corner = self.eye_points[np.where(self.eye_points == min(self.eye_points[:, 0]))]
		right_corner = self.eye_points[np.where(self.eye_points == max(self.eye_points[:, 0]))]
		return (left_corner, right_corner)

	def _extract_eye_points(self, landmarks):
		eye_points = chaikins_corner_cutting(landmarks).astype(int)
		eye_points = np.array(order_points(eye_points[np.unique(eye_points[:, 0], return_index=True, axis=0)[1]]))
		return eye_points

	def _make_mask(self):
		mask = np.ones((self.eye_region.shape[0], self.eye_region.shape[1]), np.uint8) * 255
		cv2.fillConvexPoly(mask, self.eye_points, 0)
		mask = np.array(mask, dtype=int).astype('uint8')
		mask = 255 - mask
		return mask

	def _extract_eye_region(self, frame_padding):
		x, y, w, h = self.bbox
		croped = self.frame[y - frame_padding:y + h + frame_padding, x:x + w].copy()
		eye_origin = (x, y - frame_padding)
		self.eye_points = self.eye_points - eye_origin
		croped = cv2.bilateralFilter(croped, 15, 30, 30)

		return croped, eye_origin

	def _generate_masked_gradient(self):
		sobelx = cv2.Sobel(self.eye_region, cv2.CV_64F, 1, 0, ksize=3)
		sobely = cv2.Sobel(self.eye_region, cv2.CV_64F, 1, 1, ksize=3)
		sobelx = sobelx + sobely
		sobelx = sobelx / np.max(sobelx) * 128
		grad_right = sobelx.copy()
		grad_right[np.where(grad_right < 0)] = 0
		grad_left = sobelx.copy()
		grad_left[np.where(grad_left > 0)] = 0
		grad_left = abs(grad_left)
		grad_left = np.array(grad_left, dtype='uint8')
		grad_right = np.array(grad_right, dtype='uint8')
		# grad_left = cv2.dilate(grad_left, kernel, iterations=1)  # todo JUST ADDED
		# grad_left = cv2.erode(grad_left, kernel, iterations=1)
		# grad_right = cv2.bilateralFilter(grad_right, 15, 30, 30)
		# grad_left = cv2.bilateralFilter(grad_left, 15, 30, 30)
		grad_right = cv2.bitwise_and(grad_right, self.mask)
		grad_left = cv2.bitwise_and(grad_left, self.mask)
		return grad_left, grad_right


class Face:
	def __init__(self, face, frame):
		self.face = face
		self.frame = frame
		self.landmarks = predictor(frame, face)
		self.face_position = self.posit_predict()
		self.extract_eyes()

	def extract_eyes(self):
		eye1 = np.array([[self.landmarks.part(mark).x, self.landmarks.part(mark).y] for mark in range(36, 42)])
		eye2 = np.array([[self.landmarks.part(mark).x, self.landmarks.part(mark).y] for mark in range(42, 48)])
		self.left_eye, self.right_eye = (Eye(eye1, self.frame), Eye(eye2, self.frame))

	def posit_predict(self):
		landmarks = self.landmarks
		# global head_land_enertia
		# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
		# preds = fa.get_landmarks(gray)

		size = self.frame.shape
		chin = (landmarks.part(8).x, landmarks.part(8).y)
		tip = (landmarks.part(30).x, landmarks.part(30).y)
		lipL = (landmarks.part(54).x, landmarks.part(54).y)
		lipR = (landmarks.part(48).x, landmarks.part(48).y)
		eyeL = (landmarks.part(45).x, landmarks.part(45).y)
		eyeR = (landmarks.part(36).x, landmarks.part(36).y)
		image_points = np.array([tip, chin, eyeL, eyeR, lipL, lipR], dtype="double")
		# if counter == 0:
		# 	head_land_enertia = image_points
		# else:
		# 	image_points = 0.7 * image_points + 0.3 * head_land_enertia
		# 	head_land_enertia = image_points
		model_points = np.array([
			(0.0, 0.0, 0.0),  # Nose tip
			(0.0, -330.0, -65.0),  # Chin
			(-225.0, 170.0, -135.0),  # Left eye left corner
			(225.0, 170.0, -135.0),  # Right eye right corner
			(-150.0, -150.0, -125.0),  # Left Mouth corner
			(150.0, -150.0, -125.0)  # Right mouth corner

		])
		focal_length = size[1]
		center = (size[1] / 2, size[0] / 2)
		camera_matrix = np.array(
			[[focal_length, 0, center[0]],
			 [0, focal_length, center[1]],
			 [0, 0, 1]], dtype="double"
		)
		dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
		                                                              dist_coeffs)
		# print(rotation_vector)
		# Project a 3D point (0, 0, 1000.0) onto the image plane.
		# We use this to draw a line sticking out of the nose

		(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
		                                                 translation_vector,
		                                                 camera_matrix, dist_coeffs)

		for p in image_points:
			cv2.circle(self.frame, (int(p[0]), int(p[1])), 3, (255), -1)

		p1 = (int(image_points[0][0]), int(image_points[0][1]))
		p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

		cv2.line(self.frame, p1, p2, (255), 2)
		return rotation_vector, translation_vector
	# return chin, tip, lipL, lipR, eyeL, eyeR
class iris:
	def __init__(self, left, right, eye_frame, mask):
		self.mask = mask
		self.rough_center = self._compute_rough_center(eye_frame)
		self.shape = left.shape
		self.r = self.shape[1] / 4
		left_border = self.rough_center[0] > self.r
		right_border = self.rough_center[0] < self.shape[1] - 0.6*self.r   #todo empiric coef
		self.right_gradient = []
		self.left_gradient = []
		approx_r, approx_l = self._approximate_iris_border(10, left_border=left_border,
		                                                   right_border=right_border)

		if approx_l:
			candidate_region_l = self._iris_border_dilation(approx_l)

			if DEBUG == 3:
				plt.imshow(left)
				plt.show()
			left = self._squeeze_gradient(left)
			left = left*candidate_region_l
			if DEBUG == 3:
				plt.imshow(left)
				plt.show()
			self.left_gradient = np.array(np.where(left > 0))
		if approx_r:
			candidate_region_r = self._iris_border_dilation(approx_r)
			if DEBUG == 3:
				plt.imshow(right)
				plt.show()
			right = self._squeeze_gradient(right)
			right = right*candidate_region_r

			if DEBUG == 3:
				plt.imshow(right)
				plt.show()
			self.right_gradient = np.array(np.where(right > 0))

	def _compute_rough_center(self, eye_frame):
		croped = cv2.equalizeHist(eye_frame)
		croped = cv2.medianBlur(croped, 7)

		croped = croped + self.mask
		croped[np.where(croped > 255)] = 255

		croped = np.array(croped, dtype=float)
		croped = np.power(croped, 3)  # TODO
		croped = np.array(croped / croped.min(), dtype=int)
		croped[np.where(croped > 255)] = 255
		croped = 255 - croped
		croped = croped.astype('uint8')
		pupil = find(croped)
		return pupil

	def _squeeze_gradient(self, img):
		img_copy = img.copy()
		non_zero = img_copy[np.where(img_copy > 0)]
		d = np.percentile(non_zero, 60)
		img_copy[np.where(img_copy < d)] = 0
		img_copy[np.where(img_copy > 0)] = 1
		img = img * img_copy
		xyw = []
		[[xyw.append([x, y, img[y, x] ** 1 / 3]) for x in range(img.shape[1]) if img[y, x] > 0] for y in
		 range(img.shape[0])]

		# cluster experiment
		points_for_clustering = []
		img_copy = np.zeros_like(img).astype(float)

		for i, line in enumerate(img):
			non_zero_ind = np.where(line > 0)[0]
			non_zero = line[non_zero_ind]

			if len(non_zero) > 2:
				target = np.vstack([non_zero_ind, non_zero]).transpose()
				target = target[target[:, 1].argsort()]
				target = target[-3:, :]
				for t in target[:, 0]:
					points_for_clustering.append([i, t])
					img_copy[i, int(t)] = 1
		kernel = np.ones((2, 2))
		img_copy = cv2.dilate(img_copy, kernel, iterations=1)
		clustered = scipy.ndimage.label(img_copy)[0]
		c = clustered.flatten()
		c = c[np.where(c > 0)]
		counts = np.bincount(c)
		element = np.argmax(counts)
		img_copy = np.zeros_like(img).astype(float)
		img_copy[np.where(clustered == element)] = 1
		img_copy = cv2.erode(img_copy, kernel, iterations=1)

		return img_copy

	def _approximate_iris_border(self, n, left_border, right_border):
		x0, y0 = self.rough_center
		points_l = []
		points_r = []
		delta_fi = math.pi / (2 * n)

		fi0 = math.pi / 12
		for i in range(n):
			xi = int(x0 + self.r * math.cos(fi0 - delta_fi * i))
			xi_2 = int(2 * x0 - xi)
			yi = int(y0 - self.r * math.sin(fi0 - delta_fi * i))
			if left_border: points_l.append([xi, yi])
			if right_border: points_r.append([xi_2, yi])
		return points_l, points_r

	def _iris_border_dilation(self, coords):
		polygon = np.zeros(self.shape, dtype='uint8')
		for point in coords:
			if point[0] < self.shape[1] and point[1] < self.shape[0]:
				polygon[point[1], point[0]] = 255
		dilated_polygon = cv2.dilate(polygon, kernel, iterations=4)/255
		# plt.imshow(dilated_polygon)
		# plt.show()
		# extended_points = np.array(np.where(dilated_polygon == 255)).transpose()
		return dilated_polygon


def extract_faces(frame):
	return detector(frame)


import time
def video_demo():
	global R

	cap = cv2.VideoCapture(0)
	while True:
		try:
			_, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			frame = cv2.resize(frame, (800, 450))

			R = 10
			t = time.time()
			faces = extract_faces(frame)

			face1 = Face(faces[0], frame)
			print(time.time()-t)
			frame = face1.frame
			eye_center = face1.left_eye.center + face1.left_eye.eye_origin
			frame[eye_center[1], eye_center[0]] = 255
			eye_center = face1.right_eye.center + face1.right_eye.eye_origin
			frame[eye_center[1], eye_center[0]] = 255
			cv2.imshow(f"Face", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		except:
			cv2.imshow(f"Face", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

def photo_demo():
	global R
	R = 14

	frame = cv2.imread("leo.jpg")
	frame = cv2.imread("ivan.jpg")
	# frame = cv2.imread("ya.jpg")
	# frame = cv2.imread("taya.jpeg")
	# frame = cv2.imread("flora.png")
	# frame = cv2.resize(frame, (800, 450))
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = extract_faces(frame)
	face1 = Face(faces[0], frame)

	eye_frame = face1.left_eye.eye_region
	eye_center = face1.left_eye.center
	eye_frame[eye_center[1], eye_center[0]] = np.max(eye_frame)
	plt.imshow(eye_frame)
	plt.show()

	eye_frame = face1.right_eye.eye_region
	eye_center = face1.right_eye.center
	eye_frame[eye_center[1], eye_center[0]] = np.max(eye_frame)
	plt.imshow(eye_frame)
	plt.show()


DEBUG = 1
if __name__ == '__main__':
	video_demo()
	# photo_demo()
	print()
