##################################################
'''
Date: Nov 29, 2022
author: CV teamm

To run asynchronously run batch upload

Observe output in logs

'''
##################################################

import os
import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("/usr/local/cuda-10.2/bin")
sys.path.append("/usr/local/cuda-10.2/lib64")
import time
import logging
import pika
import copy
import numpy as np
import shutil
import cv2
import json
import random
import traceback
import pycuda.autoinit  # This is needed for initializing CUDA driver
import multiprocessing
import utils_lite.configSrc as cfg
import tensorflow as tf
import requests
import moviepy.video.io.ImageSequenceClip
from datetime import datetime


logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(filename='{}logs/Post.log'.format(cfg.log_path), level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
logging.disable(logging.DEBUG)
logger=logging.getLogger()
logger.info("")
sys.stderr.write=logger.error

vicki_app = "http://192.168.1.140:8085/tsv/flashapi"
archive_size = 200
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



def make_archive(source, destination, format='zip'):
	base, name = os.path.split(destination)
	archive_from = os.path.dirname(source)
	archive_to = os.path.basename(source.strip(os.sep))
	shutil.make_archive(name, format, archive_from, archive_to)
	shutil.move('%s.%s' % (name, format), destination)

#parser for tfrecords
def parse(serialized):
	features = \
	{
	'bytes': tf.FixedLenFeature([], tf.string),
	'timestamp': tf.FixedLenFeature([], tf.string),
	#'frame_cnt': tf.FixedLenFeature([], tf.string)
	}

	parsed_example = tf.parse_single_example(serialized=serialized,features=features)
	image = parsed_example['bytes']
	timestamp = parsed_example['timestamp']
	#frame_cnt = parsed_example['frame_cnt']
	image = tf.io.decode_image(image)

	return {'image':image, 'timestamp':timestamp} #, 'frame_cnt': frame_cnt}



#parse tfrecords to jpg's
def readTfRecords(transid, cam_id):
	dataset = tf.data.TFRecordDataset(["{}archive/{}/img_{}.tfrecords".format(cfg.base_path, transid, cam_id[-1])])
	dataset = dataset.map(parse)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	frame_cnt = 0
	while True:
		frame_cnt += 1
		try:
			img, timestr = sess.run([next_element['image'], next_element['timestamp']]) #, next_element['frame_cnt']])
			current_frame = img.reshape((archive_size, archive_size, 3))
			if not os.path.exists("{}archive/{}/cam{}/images".format(cfg.base_path, transid, cam_id[-1])):
				os.makedirs("{}archive/{}/cam{}/images".format(cfg.base_path, transid, cam_id[-1]))
			cv2.imwrite('%sarchive/%s/cam%s/images/%s_%05d.jpg'%(cfg.base_path, transid, cam_id[-1], timestr.decode('utf-8'), int(frame_cnt)), current_frame)

		except Exception as e:
			if frame_cnt == 1:
				logger.info("Something Wrong With TFRecords")
			logger.info("{} frame_cnt: {}".format(cam_id,frame_cnt))
			return -1
			break
	return 200

def combine_json(transid, cam, func):

	new_json = {}
	curr_path = os.path.join('archive', transid, cam,func)

	if not os.path.exists(curr_path):
		logger.info('      {} data not available'.format(cam))
		return

	files_cam = sorted(os.listdir(curr_path))

	for num, fil in enumerate(files_cam):
		fil_name = fil.strip('.json')

		with open(os.path.join(curr_path,fil), 'r') as file:
			pose = json.load(file)

		new_json[fil_name] = pose

	with open("post_archive/{}/{}/{}_{}.json".format(transid, cam,cam,func), "w") as outfile:
		json.dump(new_json, outfile)

	return

def exe_combine(transid):
	combine_json(transid, 'cam0', 'prod')
	combine_json(transid, 'cam1', 'prod')
	combine_json(transid, 'cam2', 'prod')

def gen_video(transid):

	if not os.path.exists("archive/{}/tmp/".format(transid)):
		os.mkdir("archive/{}/tmp/".format(transid))
	l_cam0 = sorted(os.listdir("archive/{}/cam0/images".format(transid)))
	l_cam1 = sorted(os.listdir("archive/{}/cam1/images".format(transid)))
	l_cam2 = sorted(os.listdir("archive/{}/cam2/images".format(transid)))
	l = min(min(len(l_cam0), len(l_cam2)), len(l_cam1))
	for i in range(l):
		img0 = cv2.imread("archive/{}/cam0/images/{}".format(transid, l_cam0[i]))
		img1 = cv2.imread("archive/{}/cam1/images/{}".format(transid, l_cam1[i]))
		img2 = cv2.imread("archive/{}/cam2/images/{}".format(transid, l_cam2[i]))
		img_hstack = img0
		img_hstack = np.hstack((img_hstack, img1))
		img_hstack = np.hstack((img_hstack, img2))
		cv2.putText(img_hstack, 'frame:' + str(i), (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imwrite("archive/{}/tmp/{}".format(transid, l_cam0[i]), img_hstack)

	cam_folder = 'archive/{}/tmp/'.format(transid)
	c0 = sorted(os.listdir(cam_folder))
	image_files = [os.path.join(cam_folder, img) for img in c0]
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=12)
	clip.write_videofile('post_archive/{}/media.mp4'.format(transid), verbose=False, logger = None)
	return


def distance(item1, item2):
    if item1[0] != item2['product_name']:
        return 20
    time1 = datetime.strptime(item1[-1], "%Y-%m-%d:%H:%M:%S")
    time2 = datetime.strptime(item2['activity_time'], "%Y-%m-%d:%H:%M:%S")
    return abs((time1 - time2).total_seconds())


def postprocess(transid, cv_activities, ls_activities):
	logger.info("      Extracting TFRecords")
	status = readTfRecords(transid, 'cam0')
	status = readTfRecords(transid, 'cam1')
	status =readTfRecords(transid, 'cam2')

	if not os.path.exists('post_archive/{}/cam0'.format(transid)):
		os.makedirs('post_archive/{}/cam0'.format(transid))
	if not os.path.exists('post_archive/{}/cam1'.format(transid)):
		os.makedirs('post_archive/{}/cam1'.format(transid))
	if not os.path.exists('post_archive/{}/cam2'.format(transid)):
		os.makedirs('post_archive/{}/cam2'.format(transid))
	logger.info("      Merging Detection Results")
	exe_combine(transid)
	logger.info("      Generating Video")
	gen_video(transid)

	logger.info("      Cleaned Transaction")


def main():
	try:

		Trans = ['Testtrans']
		cv_activities = []
		ls_activities = []


		for transid in  Trans:
			logger.info("")
			logger.info(" Processing trans: {}".format(transid))
			postprocess(transid, cv_activities, ls_activities)
			logger.info("   Finished Current Transaction: ")

		logger.info("\n --------- batch end -----------")

	except Exception as e:
		logger.info(traceback.format_exc())


if __name__ == "__main__":
	main()

