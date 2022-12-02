#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvMedia",durable = True)

data = '{\n "src": "all", \n "parm1":"trans2"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvMedia",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
