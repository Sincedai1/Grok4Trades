"""
Telegram Notification Utility for QuantumSol Stack

This module provides a simple interface for sending notifications via Telegram
bot, used for alerts, trade signals, and system status updates.
"""
import os
import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from enum import Enum

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Priority levels for Telegram messages"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TelegramMessage:
    """Container for Telegram message data"""
    text: str
    chat_id: Optional[str] = None
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    reply_to_message_id: Optional[int] = None
    reply_markup: Optional[Dict[str, Any]] = None
    priority: MessagePriority = MessagePriority.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        data = {
            'chat_id': self.chat_id,
            'text': self.text,
            'parse_mode': self.parse_mode,
            'disable_web_page_preview': self.disable_web_page_preview,
            'disable_notification': self.disable_notification,
        }
        
        if self.reply_to_message_id:
            data['reply_to_message_id'] = self.reply_to_message_id
            
        if self.reply_markup:
            data['reply_markup'] = json.dumps(self.reply_markup)
            
        return data

class TelegramBot:
    """
    A simple Telegram bot client for sending notifications.
    
    Features:
    - Async message sending
    - Message queuing with priority
    - Rate limiting
    - Message formatting
    - Error handling and retries
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    MAX_MESSAGE_LENGTH = 4096
    
    def __init__(self, 
                token: Optional[str] = None, 
                default_chat_id: Optional[str] = None):
        """
        Initialize the Telegram bot.
        
        Args:
            token: Telegram bot token (default: from TELEGRAM_BOT_TOKEN env var)
            default_chat_id: Default chat ID for messages (default: from TELEGRAM_CHAT_ID env var)
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.default_chat_id = default_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.session: Optional[aiohttp.ClientSession] = None
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.is_processing = False
        self.rate_limit_semaphore = asyncio.Semaphore(20)  # 20 messages per minute
        
        if not self.token:
            logger.warning("No Telegram bot token provided. Notifications will be disabled.")
        
        if not self.default_chat_id:
            logger.warning("No default chat ID provided. Must specify chat_id for each message.")
    
    async def start(self):
        """Initialize the bot and start the message processing loop"""
        if not self.token:
            logger.warning("Cannot start Telegram bot: No token provided")
            return
            
        self.session = aiohttp.ClientSession()
        self.is_processing = True
        asyncio.create_task(self._process_queue())
        
        # Test the connection
        try:
            me = await self.get_me()
            logger.info(f"Telegram bot started: @{me.get('username')} ({me.get('id')})")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    async def stop(self):
        """Stop the bot and clean up resources"""
        self.is_processing = False
        
        # Wait for queue to empty with timeout
        try:
            await asyncio.wait_for(self.queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for message queue to empty")
        
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Telegram bot stopped")
    
    async def get_me(self) -> Dict[str, Any]:
        """Get information about the bot"""
        return await self._make_request("getMe")
    
    async def send_message(self, 
                         text: str, 
                         chat_id: Optional[str] = None,
                         parse_mode: str = "HTML",
                         priority: MessagePriority = MessagePriority.NORMAL,
                         **kwargs) -> bool:
        """
        Send a message via the Telegram bot.
        
        Args:
            text: Message text (supports HTML formatting if parse_mode="HTML")
            chat_id: Chat ID (default: from constructor or TELEGRAM_CHAT_ID)
            parse_mode: Text formatting mode ("HTML" or "MarkdownV2")
            priority: Message priority (higher = sent sooner)
            **kwargs: Additional arguments for TelegramMessage
            
        Returns:
            bool: True if message was queued successfully
        """
        if not self.token:
            logger.warning("Cannot send message: No Telegram bot token configured")
            return False
            
        chat_id = chat_id or self.default_chat_id
        if not chat_id:
            logger.error("Cannot send message: No chat_id provided")
            return False
        
        # Split long messages
        if len(text) > self.MAX_MESSAGE_LENGTH:
            parts = [text[i:i+self.MAX_MESSAGE_LENGTH] 
                    for i in range(0, len(text), self.MAX_MESSAGE_LENGTH)]
            
            results = []
            for part in parts:
                msg = TelegramMessage(
                    text=part,
                    chat_id=chat_id,
                    parse_mode=parse_mode,
                    priority=priority,
                    **kwargs
                )
                results.append(await self._enqueue_message(msg))
            
            return all(results)
        else:
            msg = TelegramMessage(
                text=text,
                chat_id=chat_id,
                parse_mode=parse_mode,
                priority=priority,
                **kwargs
            )
            return await self._enqueue_message(msg)
    
    async def _enqueue_message(self, message: TelegramMessage) -> bool:
        """Add a message to the send queue"""
        if not self.is_processing:
            logger.warning("Message queue not running. Call start() first.")
            return False
            
        # Priority is inverted because queue uses min-heap
        priority = -message.priority.value
        await self.queue.put((priority, datetime.now(timezone.utc), message))
        return True
    
    async def _process_queue(self):
        """Process messages from the queue with rate limiting"""
        while self.is_processing or not self.queue.empty():
            try:
                # Get the highest priority message
                _, timestamp, message = await self.queue.get()
                
                # Respect message timing (for rate limiting)
                now = datetime.now(timezone.utc)
                if timestamp > now:
                    await asyncio.sleep((timestamp - now).total_seconds())
                
                # Send the message with rate limiting
                async with self.rate_limit_semaphore:
                    success = await self._send_message_internal(message)
                    if not success:
                        # Requeue failed messages with backoff
                        message.priority = MessagePriority(min(message.priority.value + 1, 4))
                        await self._enqueue_message(message)
                
                # Mark task as done
                self.queue.task_done()
                
                # Small delay to prevent tight loops
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)  # Prevent tight error loops
    
    async def _send_message_internal(self, message: TelegramMessage) -> bool:
        """Internal method to send a message to the Telegram API"""
        if not self.session:
            logger.error("Cannot send message: Session not initialized")
            return False
            
        url = self.BASE_URL.format(token=self.token, method="sendMessage")
        
        try:
            async with self.session.post(url, json=message.to_dict()) as response:
                data = await response.json()
                
                if response.status == 200 and data.get('ok', False):
                    logger.debug(f"Message sent to chat {message.chat_id}")
                    return True
                else:
                    error_msg = data.get('description', 'Unknown error')
                    logger.error(f"Failed to send message: {error_msg}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error sending message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}", exc_info=True)
            return False
    
    async def _make_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a generic request to the Telegram Bot API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Call start() first.")
            
        url = self.BASE_URL.format(token=self.token, method=method)
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if response.status == 200 and data.get('ok', False):
                    return data.get('result', {})
                else:
                    error_msg = data.get('description', 'Unknown error')
                    raise Exception(f"API error: {error_msg}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {e}") from e

# Global instance
telegram_bot = TelegramBot()

# Example usage:
# # Initialize the bot
# await telegram_bot.start()
# 
# # Send a simple message
# await telegram_bot.send_message(
#     "<b>Alert:</b> Price of SOL has increased by 5% in the last hour!",
#     parse_mode="HTML"
# )
# 
# # Send a high-priority message
# await telegram_bot.send_message(
#     "🚨 <b>EMERGENCY:</b> Daily loss limit exceeded!",
#     priority=MessagePriority.CRITICAL
# )
# 
# # Clean up
# await telegram_bot.stop()
