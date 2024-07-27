import asyncio
import json
import logging
from dataclasses import asdict

from websockets import serve, broadcast

from pc.config import *
from pc.model import BrightnessMessage

log = logging.getLogger(__name__)


class WebSocketsHandler:
    clients = set()

    async def setup(self):
        async def register_client(websocket):
            log.info(f"registering {websocket}")
            self.clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)

        async with serve(register_client, WS_HOST, WS_PORT):
            await asyncio.Future()

    async def broadcast(self, message: BrightnessMessage):
        """send the message to all websockets"""
        log.info(f"broadcasting to {len(self.clients)} clients")
        message_json = json.dumps(asdict(message))
        broadcast(self.clients, message_json)
