import asyncio
import typing
import logging

import websockets
from websockets import WebSocketServerProtocol

from pc.config import ws_host, ws_port

log = logging.getLogger(__name__)


class WebsocketHandler:
    def __init__(self, host: str, port: int):
        self._clients: typing.Set[WebSocketServerProtocol] = set()

        self._host = host
        self._port = port

    async def start(self):
        log.info(f"starting websocket server on {self._host}:{self._port}")
        async with websockets.serve(self._handler, self._host, self._port):
            await asyncio.Future()

    async def broadcast(self, message: typing.Dict):
        """send message to all clients without blocking"""
        import asyncio

        if self._clients:
            log.info(f"sending {message} to {len(self._clients)} clients")
            await asyncio.wait([asyncio.create_task(c.send(message)) for c in self._clients])
        else:
            log.warning("no websocket clients to broadcast message to!")

    async def _handler(self, websocket: WebSocketServerProtocol):
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.remove(websocket)


websocket_handler = WebsocketHandler(host=ws_host, port=ws_port)
