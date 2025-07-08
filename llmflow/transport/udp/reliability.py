"""
UDP Reliability Layer

This module provides reliability mechanisms for UDP transport,
including acknowledgments, retransmission, and flow control.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class FlowControlState(Enum):
    """Flow control states."""
    OPEN = "open"
    CONGESTED = "congested"
    THROTTLED = "throttled"
    CLOSED = "closed"


@dataclass
class FlowControlMetrics:
    """Flow control metrics."""
    window_size: int = 64
    bytes_in_flight: int = 0
    rtt_ms: float = 0.0
    bandwidth_bps: int = 0
    packet_loss_rate: float = 0.0
    congestion_window: int = 1
    slow_start_threshold: int = 64
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'window_size': self.window_size,
            'bytes_in_flight': self.bytes_in_flight,
            'rtt_ms': self.rtt_ms,
            'bandwidth_bps': self.bandwidth_bps,
            'packet_loss_rate': self.packet_loss_rate,
            'congestion_window': self.congestion_window,
            'slow_start_threshold': self.slow_start_threshold
        }


@dataclass
class PendingMessage:
    """Pending message for retransmission."""
    sequence_number: int
    data: bytes
    timestamp: float
    retries: int = 0
    last_retry: float = 0.0
    ack_received: bool = False
    send_callback: Optional[Callable[[bytes], None]] = None



class FlowController:
    """Flow control for UDP transport."""
    
    def __init__(self, initial_window_size: int = 64, max_window_size: int = 1024):
        self.metrics = FlowControlMetrics(window_size=initial_window_size)
        self.max_window_size = max_window_size
        self.state = FlowControlState.OPEN
        self.rtt_samples: List[float] = []
        self.max_rtt_samples = 10
        self.last_congestion_time = 0.0
        self.congestion_recovery_time = 5.0  # seconds
        
        # Congestion control state
        self.in_slow_start = True
        self.duplicate_ack_count = 0
        self.last_ack_seq = -1
        
    def can_send(self, message_size: int) -> bool:
        """Check if we can send a message given current flow control state."""
        if self.state == FlowControlState.CLOSED:
            return False
        
        if self.metrics.bytes_in_flight + message_size > self.get_effective_window_size():
            return False
        
        return True
    
    def get_effective_window_size(self) -> int:
        """Get the effective window size considering congestion control."""
        return min(self.metrics.window_size, self.metrics.congestion_window * 1400)  # 1400 = typical MTU
    
    def on_message_sent(self, sequence_number: int, message_size: int) -> None:
        """Called when a message is sent."""
        self.metrics.bytes_in_flight += message_size
        
        if self.metrics.bytes_in_flight > self.get_effective_window_size():
            self.state = FlowControlState.CONGESTED
    
    def on_ack_received(self, sequence_number: int, rtt_ms: float) -> None:
        """Called when an acknowledgment is received."""
        # Update RTT
        self.rtt_samples.append(rtt_ms)
        if len(self.rtt_samples) > self.max_rtt_samples:
            self.rtt_samples.pop(0)
        
        self.metrics.rtt_ms = sum(self.rtt_samples) / len(self.rtt_samples)
        
        # Update congestion control
        if sequence_number == self.last_ack_seq:
            self.duplicate_ack_count += 1
            if self.duplicate_ack_count >= 3:
                # Fast retransmit/recovery
                self._handle_congestion()
        else:
            self.duplicate_ack_count = 0
            self.last_ack_seq = sequence_number
            self._handle_successful_ack()
        
        # Update flow control state
        self._update_flow_control_state()
    
    def on_timeout(self, sequence_number: int) -> None:
        """Called when a message times out."""
        # Timeout indicates congestion
        self._handle_congestion()
        self.metrics.packet_loss_rate = min(1.0, self.metrics.packet_loss_rate + 0.1)
    
    def _handle_successful_ack(self) -> None:
        """Handle successful acknowledgment for congestion control."""
        if self.in_slow_start:
            # Slow start: increase congestion window by 1
            self.metrics.congestion_window += 1
            
            # Exit slow start when congestion window reaches slow start threshold
            if self.metrics.congestion_window >= self.metrics.slow_start_threshold:
                self.in_slow_start = False
        else:
            # Congestion avoidance: increase congestion window by 1/cwnd
            self.metrics.congestion_window += 1.0 / self.metrics.congestion_window
        
        # Decrease packet loss rate
        self.metrics.packet_loss_rate = max(0.0, self.metrics.packet_loss_rate - 0.01)
    
    def _handle_congestion(self) -> None:
        """Handle congestion event."""
        # Set slow start threshold to half of current congestion window
        self.metrics.slow_start_threshold = max(2, self.metrics.congestion_window // 2)
        
        # Reset congestion window
        self.metrics.congestion_window = 1
        self.in_slow_start = True
        
        # Update state
        self.state = FlowControlState.CONGESTED
        self.last_congestion_time = time.time()
    
    def _update_flow_control_state(self) -> None:
        """Update flow control state based on current metrics."""
        current_time = time.time()
        
        # Check if we're recovering from congestion
        if (self.state == FlowControlState.CONGESTED and 
            current_time - self.last_congestion_time > self.congestion_recovery_time):
            self.state = FlowControlState.OPEN
        
        # Check if we need to throttle
        if self.metrics.packet_loss_rate > 0.1:
            self.state = FlowControlState.THROTTLED
        elif self.metrics.packet_loss_rate < 0.01:
            self.state = FlowControlState.OPEN
    
    def get_recommended_timeout(self) -> float:
        """Get recommended timeout based on current RTT."""
        if self.metrics.rtt_ms > 0:
            # Use 4 * RTT as timeout (RFC 6298)
            return (self.metrics.rtt_ms * 4) / 1000.0
        else:
            return 1.0  # Default 1 second timeout
    
    def get_metrics(self) -> Dict[str, any]:
        """Get current flow control metrics."""
        return {
            'state': self.state.value,
            'metrics': self.metrics.to_dict(),
            'in_slow_start': self.in_slow_start,
            'duplicate_ack_count': self.duplicate_ack_count,
            'effective_window_size': self.get_effective_window_size(),
            'recommended_timeout': self.get_recommended_timeout()
        }


class ReliabilityManager:
    """Manages reliability mechanisms for UDP transport."""
    
    def __init__(self, max_retries: int = 3, ack_timeout: float = 1.0, enable_flow_control: bool = True):
        self.max_retries = max_retries
        self.ack_timeout = ack_timeout
        self.enable_flow_control = enable_flow_control
        
        # Pending messages awaiting acknowledgment
        self.pending_messages: Dict[int, PendingMessage] = {}
        
        # Flow control
        self.flow_controller = FlowController() if enable_flow_control else None
        
        # Sequence number management
        self.next_sequence_number = 1
        self.max_sequence_number = 65535  # 16-bit sequence numbers
        
        # Receive tracking
        self.received_sequences: Set[int] = set()
        self.last_received_sequence = 0
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_retransmitted': 0,
            'acks_received': 0,
            'timeouts': 0,
            'duplicates_received': 0,
            'out_of_order_received': 0
        }
        
        # Background tasks
        self.retry_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the reliability manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self.retry_task = asyncio.create_task(self._retry_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Reliability manager started")
    
    async def stop(self) -> None:
        """Stop the reliability manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Reliability manager stopped")
    
    def get_next_sequence_number(self) -> int:
        """Get the next sequence number."""
        seq_num = self.next_sequence_number
        self.next_sequence_number = (self.next_sequence_number + 1) % self.max_sequence_number
        if self.next_sequence_number == 0:
            self.next_sequence_number = 1  # Skip 0
        return seq_num
    
    async def send_reliable(self, data: bytes, send_callback: Callable[[bytes], None]) -> bool:
        """Send data reliably."""
        sequence_number = self.get_next_sequence_number()
        
        # Check flow control
        if self.flow_controller and not self.flow_controller.can_send(len(data)):
            logger.warning(f"Flow control prevents sending message {sequence_number}")
            return False
        
        # Create pending message with send_callback for retransmission
        pending_msg = PendingMessage(
            sequence_number=sequence_number,
            data=data,
            timestamp=time.time(),
            send_callback=send_callback
        )
        
        self.pending_messages[sequence_number] = pending_msg
        
        # Send the message
        try:
            await send_callback(data)
            
            # Update flow control
            if self.flow_controller:
                self.flow_controller.on_message_sent(sequence_number, len(data))
            
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {sequence_number}: {e}")
            del self.pending_messages[sequence_number]
            return False

    
    async def handle_ack(self, sequence_number: int) -> None:
        """Handle acknowledgment for a sequence number."""
        if sequence_number in self.pending_messages:
            pending_msg = self.pending_messages[sequence_number]
            rtt_ms = (time.time() - pending_msg.timestamp) * 1000.0
            
            # Update flow control
            if self.flow_controller:
                self.flow_controller.on_ack_received(sequence_number, rtt_ms)
            
            # Remove from pending
            del self.pending_messages[sequence_number]
            
            self.stats['acks_received'] += 1
            logger.debug(f"Received ACK for sequence {sequence_number}, RTT: {rtt_ms:.2f}ms")
    
    async def handle_received_message(self, sequence_number: int, data: bytes) -> Tuple[bool, bool]:
        """
        Handle received message.
        
        Returns:
            Tuple of (is_duplicate, is_out_of_order)
        """
        is_duplicate = sequence_number in self.received_sequences
        is_out_of_order = sequence_number < self.last_received_sequence
        
        if is_duplicate:
            self.stats['duplicates_received'] += 1
            logger.debug(f"Duplicate message received: {sequence_number}")
        
        if is_out_of_order:
            self.stats['out_of_order_received'] += 1
            logger.debug(f"Out-of-order message received: {sequence_number}")
        
        # Update receive tracking
        self.received_sequences.add(sequence_number)
        if sequence_number > self.last_received_sequence:
            self.last_received_sequence = sequence_number
        
        return is_duplicate, is_out_of_order
    
    async def _retry_loop(self) -> None:
        """Background task for retrying timed-out messages."""
        while self._running:
            try:
                current_time = time.time()
                timeout = self.flow_controller.get_recommended_timeout() if self.flow_controller else self.ack_timeout
                
                # Check for timed-out messages
                timed_out_messages = []
                for seq_num, pending_msg in self.pending_messages.items():
                    if current_time - pending_msg.timestamp > timeout:
                        timed_out_messages.append(seq_num)
                
                # Handle timeouts
                for seq_num in timed_out_messages:
                    await self._handle_timeout(seq_num)
                
                # Sleep before next check
                await asyncio.sleep(min(timeout / 2, 1.0))
                
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old received sequences."""
        while self._running:
            try:
                # Clean up old received sequences to prevent memory growth
                if len(self.received_sequences) > 10000:
                    # Keep only recent sequences
                    cutoff = self.last_received_sequence - 5000
                    self.received_sequences = {seq for seq in self.received_sequences if seq > cutoff}
                
                # Sleep for 60 seconds before next cleanup
                await asyncio.sleep(60.0)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60.0)
    
    async def _handle_timeout(self, sequence_number: int) -> None:
        """Handle timeout for a specific sequence number."""
        if sequence_number not in self.pending_messages:
            return
        
        pending_msg = self.pending_messages[sequence_number]
        
        # Update flow control
        if self.flow_controller:
            self.flow_controller.on_timeout(sequence_number)
        
        # Check if we should retry
        if pending_msg.retries < self.max_retries:
            pending_msg.retries += 1
            pending_msg.last_retry = time.time()
            
            # FIXED: Implement actual retransmission using stored send_callback
            if pending_msg.send_callback:
                try:
                    await pending_msg.send_callback(pending_msg.data)
                    logger.debug(f"Retransmitted message {sequence_number}, attempt {pending_msg.retries}")
                except Exception as e:
                    logger.error(f"Retransmission failed for {sequence_number}: {e}")
            
            self.stats['messages_retransmitted'] += 1
        else:
            # Max retries reached, give up
            del self.pending_messages[sequence_number]
            self.stats['timeouts'] += 1
            logger.warning(f"Message {sequence_number} timed out after {self.max_retries} retries")

    
    def get_stats(self) -> Dict[str, any]:
        """Get reliability statistics."""
        stats = self.stats.copy()
        stats['pending_messages'] = len(self.pending_messages)
        stats['received_sequences'] = len(self.received_sequences)
        stats['last_received_sequence'] = self.last_received_sequence
        
        if self.flow_controller:
            stats['flow_control'] = self.flow_controller.get_metrics()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'messages_sent': 0,
            'messages_retransmitted': 0,
            'acks_received': 0,
            'timeouts': 0,
            'duplicates_received': 0,
            'out_of_order_received': 0
        }
