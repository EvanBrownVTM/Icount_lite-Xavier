#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvDetect",durable = True)

with open('/home/cv001/Desktop/tf_trt_models/Status.json','r') as p:
	data =p.read()	#and
data = json.loads(data) 
data["Status"][4]["INTERNAL"]["Detection"] = 'runDetect'
data["Status"][4]["INTERNAL"]["Image"] = 'detect'

data = json.dumps(data,indent=4)
print(data)
channel.basic_publish(exchange='',
                        routing_key="cvDetect",
                        body=data)

print(" [x] Sent data %", data)
connection.close()
