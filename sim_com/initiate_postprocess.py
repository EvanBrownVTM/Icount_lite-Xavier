#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvPost",durable = True)

data = '{\n "transid": "trans702", \n "cv_activities":"TEST-cv_activities", \n "ls_activities":"TEST-ls_activities"}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvPost",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
