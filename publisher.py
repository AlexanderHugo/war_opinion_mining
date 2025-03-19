import pika
import json
from typing import Dict

class RabbitmqPublisher:
    def __init__(self) -> None:
        self.__host = "localhost"
        self.__port = 5672
        self.__username = "guest"
        self.__password = "guest"
        self.__exchange = 'test_exchange'
        self.__routing_key=''
        self.__channel = self.__create_channel()
    def __create_channel(self):
        conections_parameters = pika.ConnectionParameters(
            host = self.__host,
            port=self.__port,
            credentials=pika.PlainCredentials(
                username=self.__username,
                password=self.__password
            )
        )
        channel = pika.BlockingConnection(conections_parameters).channel()

        channel.queue_declare(queue='mensagens_3', durable=True, 
                         arguments={'x-queue-type': 'stream'})

        return channel
    def send_message(self, body: Dict):
        self.__channel.basic_publish(
            exchange=self.__exchange,
            routing_key=self.__routing_key,
            body=json.dumps(body),
            properties=pika.BasicProperties(
                delivery_mode=2
                )
            )
