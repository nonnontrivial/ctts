import json
import aio_pika


class RabbitMQ:
    def __init__(self, url: str, queue: str):
        self.url = url
        self.queue = queue
        self.connection = None
        self.channel = None

    async def connect(self):
        if not self.connection or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()

    async def publish(self, message: dict):
        await self.connect()
        if self.channel is not None:
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(), content_type="application/json"
                ),
                routing_key=self.queue,
            )

    async def close(self):
        pass
