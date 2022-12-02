#!/usr/bin/env python
import pika
#from getkey import getkey, keys

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue='cvRequest',durable =True)
#channel.queue_purge("cvRequest")

def callback(ch, method, properties, body):
    global channel
    print(" [x] Received %r" % body)
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume(queue = 'cvRequest', on_message_callback=callback) #, auto_ack=True)
channel.start_consuming()

def main(count):
    #connection = pika.BlockingConnection(pika.ConnectionParameters(host = 'localhost'))
    #channel = connection.channel()
    while(1):
        try:
            _,_,recv = channel.basic_get('cvRequest')
            if recv != None:
                print("received", recv)
            
        except Exception as e:
            #print(e)
            count += 1
            if count > 2:
                print("creating a new connection")
                count = 0
                break
            main(count)

if __name__ == "__main__":
    count = 0
    main(count)

    


