import numpy as np
import cv2
import time
from collections import defaultdict
from .utils import get_roi_bbox, bbox_area, get_center, descale_contour, find_zone

timestamp_format = "%Y-%m-%d:%H:%M:%S"

class Product:
	_inference_delay = 2
	_zone_names = {'high_left': 'top_shelf', 'high_right': 'top_shelf', 'mid_left': 'second_shelf', 
				   'mid_right': 'second_shelf', 'low_left': 'lower_shelfs', 'low_right': 'lower_shelfs'}
	def __init__(self, cam_id): #, machine_contours, still_frame):
		self._image_size = (416, 416)
		self.contour_path = "utils_lite/contours_v2.npz"
		self.staging_cnt, self.roi_cnt = get_roi_bbox(self._image_size, self.contour_path)
		#self.roi_cnt[:,:,-1] -= 10
		
		self._action_rel_iou = 0
		self._action = False
		self._action_started_frame = None        
		self._action_ended = False
		self._movement_label = None
		self._action_ended_frame = None
		self._data = list()
		self._post_inference_couter = -1
		self._prods_bboxes_to_remove = set()
		self._side_action = False
		self._hand_detected = False
		self._zone_name = None
		self.is_processing = False
		self._perform_inference = False
		self._pick_action_ended = False
		self._return_action_ended = False
		self._action_type = None
		self._cam_id = cam_id
		self._active_zone = None
		#self._machine_contours = machine_contours
		#self._still_frame = still_frame
	
		
	def update(self, frame_id, hand, cls, frame):
		self._perform_inference = False
		hand_in_roi = False
		
		self._avg_movement_vector()
		#if hand is not None:
			#self._active_zone = find_zone(hand, frame, self._machine_contours, self._still_frame)
			
		#if hand is not None and (self._check_cnt_intersection(self.staging_cnt, hand) > 0.1): # hand is in roi
		if hand is not None and (self._check_cnt_intersection(self.staging_cnt, hand) > 0.0):
			hand_in_roi = True

		# check if hand is in roi for interactions with the highest shelf       
		if hand is not None and (hand[0] < 0.1*self._image_size[0]) or \
			   (hand[2] > 0.95*self._image_size[0]):
			hand_in_roi = True
			
		if hand is not None:
			self._update_action(self.roi_cnt, hand, frame_id)
		
		if len(self._data) > 0:
			#print(self._action, self._data[-1]['action'])
			prev_action = self._data[-1]['action']
			if (not self._action and self._data[-1]['action']):
				self._action_ended = True
				self._pick_action_ended = True
				self._action_ended_frame = frame_id
				
			if (self._action and not self._data[-1]['action']):
				self._return_action_ended = True
				self._action_ended_frame = frame_id
				
			# wait self._inference_delay frames for inference
			#if self._action_ended and not hand_in_roi:
			
			if (self._pick_action_ended and not hand_in_roi) or (self._return_action_ended):
				self._post_inference_couter = self._inference_delay
				self._action_ended = False
				if self._pick_action_ended:
					self._action_type = 'PICK'
					self._pick_action_ended = False
				if self._return_action_ended:
					self._action_type = 'RETURN'
					self._return_action_ended = False
				self.is_processing = True

			# check if hand is outside of RoI 
			
			#if not self._action and self._post_inference_couter > 0 and not hand_in_roi:
			if (not self._action and self._post_inference_couter > 0 and not hand_in_roi) or (self._action and self._post_inference_couter > 0):
				self.is_processing = True
				self._post_inference_couter -= 1

			if self._post_inference_couter == 0:
				self.is_processing = True
				self._perform_inference = True
				
				
		self._data.append({'frame_id': frame_id, 'hand': hand, 'zone': self._zone_name, 'hand_in_roi': hand_in_roi, 'action_zone_iou': self._action_rel_iou,
						   'action': self._action, 'op': self._movement_label, 'prod': cls})
						   
						   
	def _update_action(self, roi_cnt, hand, frame_id):
		self._action_rel_iou = self._check_cnt_intersection(roi_cnt, hand)
		main_action = True if self._action_rel_iou > 0 else False
		
		# check side of the hand
		side_left = True if (hand[0] + hand[2]) // 2 < 0.5*self._image_size[0] else False
		
		side_action = False
		# check if side action is performed using hand and wrist positions
		if (side_left and hand[0] < 0.05*self._image_size[0]) or \
				(not side_left and hand[2] > 0.95*self._image_size[0]) or \
					(self._side_action and self._data[-1]['hand_in_roi']):
			side_action = True
			self._action_rel_iou = 0.2


		if self._side_action and \
			   (side_left and hand[0] > 0.1*self._image_size[0] or not side_left and hand[2] < 0.9*self._image_size[0]):
			main_action = False
			side_action = False
			
		self._action = True if main_action or side_action else False
		self._side_action = side_action
		if self._action or self._side_action:
			self.is_processing = True



	def _check_cnt_intersection(self, cnt, hand):
		hand_area = bbox_area(hand)
		if hand_area == 0:
			return 0
		
		if len(cnt.shape) > 1:
			cnt = [cnt]
		
		blank = np.zeros(self._image_size[:2]).astype('int8')
		cnt_img = cv2.drawContours(blank.copy(), cnt, 0, color=255, thickness=-1)
		hand_img = cv2.rectangle(blank, (hand[0], hand[1]), (hand[2], hand[3]), color=255, thickness=-1)
		
		inter = cv2.bitwise_and(hand_img, cnt_img)
		inter_area = np.count_nonzero(inter)
		
		rel_iou = inter_area / hand_area
		return rel_iou
		

					
	def _avg_movement_vector(self):
		"""Estimates movement vector in current frame based on previous movements
		"""
		if len(self._data) < 2:
			vec = [0, 0]
		else:
			n = min(len(self._data) - 1, 4)
			starts = np.array([get_center(p['hand']) for p in self._data[-n-1:-1]])
			ends = np.array([get_center(p['hand']) for p in self._data[-n:]])
			vec =  np.mean(ends - starts, axis=0)
		
		if np.dot(vec, np.array([0, 1])) > 0:
			self._movement_label = 'in'
		elif np.dot(vec, np.array([0, 1])) < 0:
			self._movement_label = 'out'
		
		#print(self._movement_label)
		
		
	def _clear_state(self):
		self._data = self._data[-1:]
		self._post_inference_couter = -100
		self._movement_label = None
		self._prods_bboxes_to_remove = set()
		
	def infer(self, cart, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle):
		#print(self._post_inference_couter)
		if self._perform_inference:
			self._inference(cart, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle)
			self._clear_state()

	def _get_prods_cls(self, type):
		filtered_op = list(filter(lambda x: x['op'] == type, self._data))
		#print([x for x in filtered_op if x['prod']])

		prods_cls = {}
		for frame_data in filtered_op:
			frame_id = frame_data['frame_id']
			classes = frame_data['prod']
			prods_cls[frame_id] = [classes]
				
		prod_num = [len(x) for x in prods_cls.values()]
		
		if not prod_num:
			return prods_cls, []
		
		# prevent cases of FP prods detections in case there is one FP detection
		n_prods = max(prod_num) if prod_num.count(max(prod_num)) > 1 else max(prod_num) - 1 
		classes = {}

		for i in range(n_prods):
			classes[i + 1] = [x[i] for x in prods_cls.values() if len(x) > i]
			
		classes_final = []
		for values in classes.values():
			if values:
				classes_final.append(max(values, key=values.count))
		
		return prods_cls, classes_final
		
	def _inference(self, cart, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle):
		"""Performs action inference

		Returns:
			list[tuple(action:str, product class: int, alert: bool)]: list with detected actions
		"""
		in_prods_cls, in_classes = self._get_prods_cls('in')
		out_prods_cls, out_classes = self._get_prods_cls('out') 
		#print(f'Prods in out: {in_prods_cls, out_prods_cls}')
		
		if in_prods_cls and out_prods_cls:
			min_in_frame = min(in_prods_cls.keys()) if in_prods_cls else 0
			max_out_frame  = max(out_prods_cls.keys()) if out_prods_cls else 1
			if min_in_frame > max_out_frame:
				if len(out_prods_cls) > 2*len(in_prods_cls):
					in_prods_cls = []
				elif len(in_prods_cls) > 2*len(out_prods_cls):
					out_prods_cls = []

		frame_id = self._data[-1]['frame_id']
		# get action type based on macthed prods info
		if in_prods_cls and len(out_prods_cls) <= 1:
			if not idle:
				for cl in in_classes:
					logger.info('      {} - RETURN {}'.format(self._cam_id, int(cl)))
					cv_activities.append({"class_id":int(cl), "action":"RETURN", "timestamp": time.strftime(timestamp_format)})
					cv_ret_cam.append((int(cl), 'RETURN', cv_activities[-1]['timestamp']))
					cart[int(cl)] -= 1
		elif len(in_prods_cls) <= 1 and out_prods_cls:
			if not idle:
				for cl in out_classes:
					logger.info('      {} - PICK {}'.format(self._cam_id, int(cl)))
					cv_activities.append({"class_id":int(cl), "action":"PICK", "timestamp": time.strftime(timestamp_format)})
					cv_pick_cam.append((int(cl), 'PICK', cv_activities[-1]['timestamp']))
					cart[int(cl)] += 1
		elif len(in_prods_cls) > 1 and len(out_prods_cls) > 1:
			if self._action_type == 'RETURN':
				if not idle:
					for cl in in_classes:
						logger.info('      {} - RETURN {}'.format(self._cam_id, int(cl)))
						cv_activities.append({"class_id":int(cl), "action":"RETURN", "timestamp": time.strftime(timestamp_format)})
						cv_ret_cam.append((int(cl), 'RETURN', cv_activities[-1]['timestamp']))
						cart[int(cl)] -= 1
				self._action_type == None
			elif self._action_type == 'PICK':
				if not idle:
					for cl in out_classes:
						logger.info('      {} - PICK {}'.format(self._cam_id, int(cl)))
						cv_activities.append({"class_id":int(cl), "action":"PICK", "timestamp": time.strftime(timestamp_format)})
						cv_pick_cam.append((int(cl), 'PICK', cv_activities[-1]['timestamp']))
						cart[int(cl)] += 1
				self._action_type == None


class FrontCam:
	def __init__(self, cam_id, zones_contour_path):
		self.cart = defaultdict(int)
		self._tracks = {}
		#self._image_size = (640, 640)
		self._cam_id = cam_id
		#self._machine_cnts = self.get_machine_contours(zones_contour_path)
		#self.still_frame = cv2.resize(cv2.imread('utils_lite/still_frame.png'), self._image_size)[:,:,::-1]
		
	def get_machine_contours(self, zones_path):
		contours = {}
		with np.load(zones_path, allow_pickle=True) as data:
			for zone_name in ['high_left', 'high_right', 'mid_left', 'mid_right', 'low_left', 'low_right']:
				contours[zone_name] = data[zone_name]
		return contours
		
	def update_tracks(self, cls, bbox, objectID, frame_id, frame):
		if objectID not in self._tracks:
			self._tracks[objectID] = Product(self._cam_id) #, self._machine_cnts, self.still_frame)
		self._tracks[objectID].update(frame_id, bbox, cls, frame)
		
	def interact_tracks(self, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle = False):
		for id_, prod in self._tracks.items():
			prod.infer(self.cart, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle)
