import numpy as np
import cv2
from collections import defaultdict
from .utils import get_center, calculate_angle, calculate_average_movement_vector, descale_contour, bbox_area
import time

timestamp_format = "%Y-%m-%d:%H:%M:%S"
class Product():
	def __init__(self, prod_id, prod_bbox, class_id, frame_id, cam_id, max_hist_len = 100,
				interact_length = 15, max_class_hist_len = 5):
		self._bbox = prod_bbox
		self._cam_id = cam_id
		self._id = prod_id
		self._prod_hist = list()                 # history of last tracked prod bboxes, [prod_bbox]
		self._hand_hist = list()                 # history of last matched hand_bbox, [hand_bbox], None if not matched
		self._class_id = class_id                # class of the product 
		self._current_frame = frame_id
		self._max_hist_len = max_hist_len        # max frames to store 
		self._interact_length = interact_length  # min length of new track to run model or for old track to be removed
		self._interact = False                   # flag if we need to interact 
		self._is_added = False                   # flag if track was already interacted after addition 
		self._max_class_hist_len = max_class_hist_len
		self._class_hist = list()
		self._need_cut = None                    # flag that shows if we need to empty hist after first add interaction
		self._last_hand_idx = None
		self._last_hand_idxs = []
		self._last_hand_before_hist = []
		self._delay = 0
		self._min_cent = 600
		self._active_zone = None
		self._zone_set = set([])
		
	def update_state(self, update_bbox, class_id, frame_id):
		"""Update product data.
		
		Args:
			update_bbox (list): new bounding box in Pascal VOC format
			class_id (int): predicted class of an object
			frame_id (int): number of a frame
		"""
		self._min_cent = min(update_bbox[0], self._min_cent)
		self._bbox = update_bbox
		self._class_id = class_id
		self._prod_hist.append(self._bbox)
		self._class_hist.append(self._class_id)
		self._current_frame = frame_id

		if not self._is_added:
			if len(self._prod_hist) == self._interact_length:
				self._interact = True

		if len(self._prod_hist) > self._max_hist_len:
			self._prod_hist.pop(0)
		if len(self._class_hist) > self._max_class_hist_len:
			self._class_hist.pop(0)

	def avg_movement_vector(self, hand=False):
		"""Calculates average movement vector from product or hand history

		Args:
			hand (bool): If True then hand history would be used

		Returns:
			np.ndarray: average movement vector
		"""
		if hand:
			hist = self.hand_hist
		else:
			hist = self._prod_hist
		return calculate_average_movement_vector(hist)


	def _most_frequent_class(self):
		"""Finding class that was predicted the most often for this product

			Returns:
				int: class of the product
		"""
		return max(set(self._class_hist), key = self._class_hist.count)

	def __repr__(self):
		return f'Product {self._id} of class {self._class_id} [{self._bbox}] on {self.cam_id}'

	@property
	def hist(self):
		return self._prod_hist

	@property
	def hand_hist(self):
		return self._hand_hist

	@property
	def class_hist(self):
		return self._class_hist

	@property
	def interact(self):
		return self._interact

	@property
	def class_id(self):
		return self._most_frequent_class()

	@property 
	def bbox(self):
		return self._bbox

	@property
	def cam_id(self):
		return self._cam_id


	@property
	def is_detected(self):
		return self._delay <= 1

	@property
	def is_processing(self):
		if not self.is_detected or not self.is_added:
			return True
		return False

	@property
	def is_added(self):
		return self._is_added

	@is_added.setter
	def is_added(self, flag):
		self._is_added = flag


    
class FrontCam:
	def __init__(self, cam_id, zones_contours_path):
		self.cart = defaultdict(int)
		self._tracks = {}
		self._max_delay = 30
		self._interact_length = 15
		self._cam_id = cam_id
		
		self._zones = np.load(zones_contours_path)
		self._shelves_priority = ['all_shelves']
			
	def update_zone(self, product, cur_frame):
		intersect_dict = {}
		shape = cur_frame.shape[:2]

		for contour_name in self._zones.files:
			contour = descale_contour(self._zones[contour_name], shape)
			mask = np.zeros(shape, dtype=np.uint8)
			mask = cv2.drawContours(mask, [contour], -1, (1), cv2.FILLED)
			
			last_bbox = product.bbox
			x1, y1, x2, y2 = last_bbox
			
			cropped_mask = mask[y1:y2, x1:x2]
			intersect_dict[contour_name] = cropped_mask.sum() / bbox_area(last_bbox)

		zone, max_intersection = max(intersect_dict.items(), key=lambda x: x[1])

		if max_intersection > 0.15:
			product._active_zone = zone
		else:
			product._active_zone = None
		product._zone_set.add(product._active_zone)
			
	def update_tracks(self, prod_clss, prod_bbox, product_track_id, frame_id, frame):
		if product_track_id not in self._tracks.keys():
			self._tracks[product_track_id] = Product(product_track_id, prod_bbox, prod_clss, frame_id, self._cam_id,
										max_hist_len = 100, interact_length = self._interact_length,
										max_class_hist_len = 12)

		if product_track_id in self._tracks.keys():
			self._tracks[product_track_id].update_state(prod_bbox, prod_clss, frame_id)
			self.update_zone(self._tracks[product_track_id], frame)
			self._tracks[product_track_id]._delay = 0
			
			
	def interact_tracks(self, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle = False):
		keys_to_remove = []
		res_actions = []
		for key, product in self._tracks.items():
			# product, _ = value
			product._delay += 1 # update delay

			if product._delay > self._max_delay and (len(product.hist) > 9 or product._is_added):
				actions = self.perform_inference(logger, key, product.class_id, cv_activities, cv_pick_cam, cv_ret_cam, type = "remove", idle = idle)
				if actions:
					res_actions.append(actions)
				keys_to_remove.append(key)
			elif product._delay > self._max_delay:
				actions = self.perform_inference(logger, key, product.class_id, cv_activities, cv_pick_cam, cv_ret_cam, idle = idle)
				if actions:
					res_actions.append(actions)
				keys_to_remove.append(key)

			if not product.is_added:
				if product.interact:
					actions = self.perform_inference(logger, key, product.class_id, cv_activities, cv_pick_cam, cv_ret_cam, type = "add", idle = idle)
					if actions:
						res_actions.append(actions)
					product.is_added = True
		
		self._tracks = {key: self._tracks[key] for key in self._tracks if key not in keys_to_remove}

		return res_actions
			
		
	def perform_inference(self, logger, obj_id, class_id, cv_activities, cv_pick_cam, cv_ret_cam, type = None, idle = False):
		product = self._tracks[obj_id]
		#print(product.is_added, obj_id, product._min_cent)
		#print(len(product._prod_hist))
		if len(product._prod_hist) < 5 or product._min_cent > 240:
			return
			
		movement_vector = calculate_average_movement_vector(product._prod_hist)

		if np.dot(movement_vector, np.array([0, 1])) > 0:
			action = 'RETURN'
		else:
			action = 'PICK'
		
		if action == 'PICK' and type == 'remove':
			action = 'NO ACTION'
		elif action == 'RETURN' and type == 'add':
			action = 'NO ACTION'
			
		if action == 'PICK':
			v_flag = 0
			for _z in product._zone_set:
				if _z in self._shelves_priority:
					v_flag = 1
			if v_flag == 0:
				action = 'NO ACTION'
		
		if action == 'RETURN':
			if product._active_zone not in self._shelves_priority:
				action = 'NO ACTION'
				
		if not action == 'NO ACTION':
			product._hand_hist = []
			product._prod_hist = []
			product._last_hand_idx = None
			product._last_hand_idxs = []
			product._last_hand_before_hist = []
		if action != 'NO ACTION' and not idle:
			logger.info('      {} - {} - {} {}'.format(self._cam_id, product._active_zone, action, int(class_id)))
			cv_activities.append({"class_id":int(class_id), "action":action, "timestamp": time.strftime(timestamp_format), "active_zone": product._active_zone})
			if action == 'PICK':
				self.cart[int(class_id)] += 1
				cv_pick_cam.append((int(class_id), 'PICK', cv_activities[-1]['timestamp'], cv_activities[-1]['active_zone']))
			else:
				self.cart[int(class_id)] -= 1
				cv_ret_cam.append((int(class_id), 'RETURN', cv_activities[-1]['timestamp'], cv_activities[-1]['active_zone']))
			
