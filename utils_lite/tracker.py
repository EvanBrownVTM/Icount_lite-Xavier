import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class AVT():
	def __init__(self, maxDisappeared=5):

		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, cents):
		disap = []
		tmp_flag = 0


		# check to see if the list of input bounding box rectangles
		# is empty
		if len(cents) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			for _ID in self.disappeared:
				if (not (_ID in self.objects)) and (self.disappeared[_ID] > 0):
					disap.append(_ID)
			return self.objects, disap

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(cents), 2), dtype="int")
		ind = 0
		#for cnt in cents:
		#	inputCentroids[ind] = (cnt[0], cnt[1])
		#	ind += 1
		# loop over the bounding box rectangles
		for (i, (cX, cY)) in enumerate(cents):
		# use the bounding box coordinates to derive the centroid
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):

				if D[row][col] > 100:
					skip_flag = 1
					continue

				if row in usedRows or col in usedCols:
					continue
				
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] > D.shape[1] or (D.shape[0] == D.shape[1] and tmp_flag != 1):
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
					
			else:
				tmp_flag = 0
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		for _ID in self.disappeared:
			if (not (_ID in self.objects)) and (self.disappeared[_ID] > 0):
				disap.append(_ID)
		return self.objects, disap


