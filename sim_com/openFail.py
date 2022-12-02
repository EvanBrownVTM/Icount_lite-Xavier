#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('192.168.1.145',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvRequest",durable = True)

data = "DoorOpened"
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvRequest",
                        body=data)

print(" [x] Sent data %", data)
connection.close()
