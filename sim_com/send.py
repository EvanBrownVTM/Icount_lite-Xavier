#!/usr/bin/env python
import pika
import json
import base64
import os
import zlib
import sys
import time
import random
import time
from PIL import Image

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvStream",durable = True)

"""
with open("test.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

data = zlib.compress(encoded_string, 9)
"""

i = 0
while True:
	data = {'tmstp':str(time.time()), 'transid':'trans700'}
	data = json.dumps(data)
	channel.basic_publish(exchange='',
		                routing_key="cvStream",
		                body=data)
	i += 1
	break
#print(" [x] Sent data %", data)
connection.close()
