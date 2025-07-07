"""
UDP Flow Control

This module provides flow control mechanisms for UDP transport,
including congestion control and bandwidth management.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class CongestionState(Enum):
    """Congestion control states."""
    SLOW_START = "slow_start"
    CONGESTION_AVOIDANCE = "congestion_avoidance"
    FAST_RECOVERY = "fast_recovery"
    TIMEOUT_RECOVERY = "timeout_recovery"


@dataclass
class BandwidthSample:
    """Bandwidth measurement sample."""
    timestamp: float
    bytes_sent: int
    rtt_ms: float


class CongestionController:
    """TCP-like congestion control for UDP."""
    
    def __init__(self, initial_window: int = 1, max_window: int = 1024):
        self.initial_window = initial_window
        self.max_window = max_window
        
        # Congestion window (in packets)
        self.congestion_window = float(initial_window)
        self.slow_start_threshold = max_window // 2
        self.state = CongestionState.SLOW_START
        
        # RTT tracking
        self.rtt_samples: deque = deque(maxlen=100)
        self.smoothed_rtt = 0.0
        self.rtt_variance = 0.0
        
        # Loss detection
        self.duplicate_ack_count = 0
        self.last_ack_seq = -1
        self.fast_recovery_start_seq = -1
        
        # Bandwidth estimation
        self.bandwidth_samples: deque = deque(maxlen=10)
        self.estimated_bandwidth = 0.0
        
        # Statistics
        self.stats = {
            'congestion_events': 0,
            'slow_start_exits': 0,
            'fast_recoveries': 0,
            'timeouts': 0,
            'window_reductions': 0
        }
    
    def on_ack_received(self, sequence_number: int, rtt_ms: float, bytes_acked: int) -> None:
        """Handle acknowledgment received."""
        # Update RTT
        self._update_rtt(rtt_ms)
        
        # Update bandwidth estimation
        self._update_bandwidth_estimate(bytes_acked, rtt_ms)
        
        # Handle duplicate ACKs
        if sequence_number == self.last_ack_seq:
            self.duplicate_ack_count += 1
            self._handle_duplicate_ack(sequence_number)
        else:
            self.duplicate_ack_count = 0
            self.last_ack_seq = sequence_number
            self._handle_new_ack(sequence_number, bytes_acked)
    
    def on_timeout(self, sequence_number: int) -> None:
        """Handle timeout event."""
        logger.debug(f"Timeout for sequence {sequence_number}")
        
        # Timeout indicates severe congestion
        self.slow_start_threshold = max(2, int(self.congestion_window / 2))
        self.congestion_window = 1.0
        self.state = CongestionState.SLOW_START
        
        # Reset duplicate ACK tracking
        self.duplicate_ack_count = 0
        self.last_ack_seq = -1
        
        # Update statistics
        self.stats['timeouts'] += 1
        self.stats['congestion_events'] += 1
        self.stats['window_reductions'] += 1
    
    def _update_rtt(self, rtt_ms: float) -> None:
        """Update RTT measurements."""
        self.rtt_samples.append(rtt_ms)
        
        if self.smoothed_rtt == 0.0:
            # First RTT measurement
            self.smoothed_rtt = rtt_ms
            self.rtt_variance = rtt_ms / 2
        else:
            # Exponential weighted moving average (RFC 6298)
            alpha = 0.125
            beta = 0.25
            
            self.rtt_variance = (1 - beta) * self.rtt_variance + beta * abs(self.smoothed_rtt - rtt_ms)
            self.smoothed_rtt = (1 - alpha) * self.smoothed_rtt + alpha * rtt_ms
    
    def _update_bandwidth_estimate(self, bytes_acked: int, rtt_ms: float) -> None:
        """Update bandwidth estimation."""
        if rtt_ms > 0:
            # Calculate bandwidth for this sample
            bandwidth_bps = (bytes_acked * 8 * 1000) / rtt_ms  # bits per second
            
            sample = BandwidthSample(
                timestamp=time.time(),
                bytes_sent=bytes_acked,
                rtt_ms=rtt_ms
            )
            
            self.bandwidth_samples.append(sample)
            
            # Calculate estimated bandwidth as average of recent samples
            if self.bandwidth_samples:
                total_bandwidth = sum(s.bytes_sent * 8 * 1000 / s.rtt_ms for s in self.bandwidth_samples)
                self.estimated_bandwidth = total_bandwidth / len(self.bandwidth_samples)
    
    def _handle_duplicate_ack(self, sequence_number: int) -> None:
        """Handle duplicate ACK."""
        if self.duplicate_ack_count == 3:
            # Fast retransmit trigger
            if self.state != CongestionState.FAST_RECOVERY:
                self._enter_fast_recovery(sequence_number)
        elif self.duplicate_ack_count > 3 and self.state == CongestionState.FAST_RECOVERY:
            # Inflate congestion window
            self.congestion_window += 1.0
    
    def _handle_new_ack(self, sequence_number: int, bytes_acked: int) -> None:
        """Handle new ACK."""
        if self.state == CongestionState.FAST_RECOVERY:
            # Exit fast recovery
            self.congestion_window = self.slow_start_threshold
            self.state = CongestionState.CONGESTION_AVOIDANCE
        elif self.state == CongestionState.SLOW_START:
            # Slow start: increase window exponentially
            self.congestion_window += 1.0
            
            # Check if we should exit slow start
            if self.congestion_window >= self.slow_start_threshold:
                self.state = CongestionState.CONGESTION_AVOIDANCE
                self.stats['slow_start_exits'] += 1
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            # Congestion avoidance: increase window linearly
            self.congestion_window += 1.0 / self.congestion_window
        
        # Ensure window doesn't exceed maximum
        self.congestion_window = min(self.congestion_window, self.max_window)
    
    def _enter_fast_recovery(self, sequence_number: int) -> None:
        """Enter fast recovery state."""
        logger.debug(f"Entering fast recovery for sequence {sequence_number}")
        
        # Set slow start threshold
        self.slow_start_threshold = max(2, int(self.congestion_window / 2))
        
        # Set congestion window
        self.congestion_window = self.slow_start_threshold + 3
        
        # Change state
        self.state = CongestionState.FAST_RECOVERY
        self.fast_recovery_start_seq = sequence_number
        
        # Update statistics
        self.stats['fast_recoveries'] += 1
        self.stats['congestion_events'] += 1
        self.stats['window_reductions'] += 1
    
    def get_congestion_window(self) -> int:
        """Get current congestion window size."""
        return max(1, int(self.congestion_window))
    
    def get_timeout_value(self) -> float:
        """Get recommended timeout value."""
        if self.smoothed_rtt > 0:
            # Calculate RTO according to RFC 6298
            rto = self.smoothed_rtt + max(0.1, 4 * self.rtt_variance)
            return rto / 1000.0  # Convert to seconds
        else:
            return 1.0  # Default timeout
    
    def get_stats(self) -> Dict[str, any]:
        """Get congestion control statistics."""
        return {
            'state': self.state.value,
            'congestion_window': self.congestion_window,
            'slow_start_threshold': self.slow_start_threshold,
            'smoothed_rtt': self.smoothed_rtt,
            'rtt_variance': self.rtt_variance,
            'estimated_bandwidth': self.estimated_bandwidth,
            'duplicate_ack_count': self.duplicate_ack_count,
            'stats': self.stats.copy()
        }


class FlowControlManager:
    """Manages flow control for UDP transport."""
    
    def __init__(self, max_window_size: int = 1024, enable_congestion_control: bool = True):
        self.max_window_size = max_window_size
        self.enable_congestion_control = enable_congestion_control
        
        # Congestion control
        self.congestion_controller = CongestionController(max_window=max_window_size) if enable_congestion_control else None
        
        # Flow control window
        self.advertised_window = max_window_size
        self.effective_window = max_window_size
        
        # In-flight data tracking
        self.bytes_in_flight = 0
        self.packets_in_flight = 0
        
        # Rate limiting
        self.rate_limit_bps = 0  # 0 = no limit
        self.rate_limit_window = 1.0  # 1 second window
        self.rate_limit_bytes_sent = 0
        self.rate_limit_window_start = time.time()
        
        # Statistics
        self.stats = {
            'packets_sent': 0,
            'packets_dropped': 0,
            'bytes_sent': 0,
            'rate_limit_hits': 0,
            'window_full_events': 0
        }
    
    def can_send_packet(self, packet_size: int) -> bool:
        """Check if we can send a packet."""
        # Check rate limit
        if not self._check_rate_limit(packet_size):
            return False
        
        # Check flow control window
        if not self._check_flow_control_window(packet_size):
            return False
        
        return True
    
    def _check_rate_limit(self, packet_size: int) -> bool:
        """Check rate limiting."""
        if self.rate_limit_bps <= 0:
            return True  # No rate limit
        
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.rate_limit_window_start >= self.rate_limit_window:
            self.rate_limit_bytes_sent = 0
            self.rate_limit_window_start = current_time
        
        # Check if sending this packet would exceed rate limit
        bits_to_send = packet_size * 8
        if self.rate_limit_bytes_sent * 8 + bits_to_send > self.rate_limit_bps:
            self.stats['rate_limit_hits'] += 1
            return False
        
        return True
    
    def _check_flow_control_window(self, packet_size: int) -> bool:
        """Check flow control window."""
        # Calculate effective window size
        if self.congestion_controller:
            congestion_window = self.congestion_controller.get_congestion_window()
            self.effective_window = min(self.advertised_window, congestion_window)
        else:
            self.effective_window = self.advertised_window
        
        # Check if we have room in the window
        if self.bytes_in_flight + packet_size > self.effective_window * 1400:  # Assume 1400 byte MTU
            self.stats['window_full_events'] += 1
            return False
        
        return True
    
    def on_packet_sent(self, packet_size: int) -> None:
        """Called when a packet is sent."""
        self.bytes_in_flight += packet_size
        self.packets_in_flight += 1
        self.rate_limit_bytes_sent += packet_size
        
        self.stats['packets_sent'] += 1
        self.stats['bytes_sent'] += packet_size
    
    def on_packet_acked(self, packet_size: int, rtt_ms: float) -> None:
        """Called when a packet is acknowledged."""
        self.bytes_in_flight = max(0, self.bytes_in_flight - packet_size)
        self.packets_in_flight = max(0, self.packets_in_flight - 1)
        
        # Update congestion control
        if self.congestion_controller:
            # Use a dummy sequence number for now
            self.congestion_controller.on_ack_received(0, rtt_ms, packet_size)
    
    def on_packet_lost(self, packet_size: int) -> None:
        """Called when a packet is lost."""
        self.bytes_in_flight = max(0, self.bytes_in_flight - packet_size)
        self.packets_in_flight = max(0, self.packets_in_flight - 1)
        
        self.stats['packets_dropped'] += 1
        
        # Update congestion control
        if self.congestion_controller:
            # Use a dummy sequence number for now
            self.congestion_controller.on_timeout(0)
    
    def set_rate_limit(self, bits_per_second: int) -> None:
        """Set rate limit in bits per second."""
        self.rate_limit_bps = bits_per_second
        logger.info(f"Rate limit set to {bits_per_second} bps")
    
    def set_window_size(self, window_size: int) -> None:
        """Set advertised window size."""
        self.advertised_window = min(window_size, self.max_window_size)
        logger.info(f"Window size set to {self.advertised_window}")
    
    def get_effective_window_size(self) -> int:
        """Get current effective window size."""
        return self.effective_window
    
    def get_congestion_window_size(self) -> int:
        """Get current congestion window size."""
        if self.congestion_controller:
            return self.congestion_controller.get_congestion_window()
        return self.advertised_window
    
    def get_timeout_value(self) -> float:
        """Get recommended timeout value."""
        if self.congestion_controller:
            return self.congestion_controller.get_timeout_value()
        return 1.0  # Default timeout
    
    def get_stats(self) -> Dict[str, any]:
        """Get flow control statistics."""
        stats = {
            'advertised_window': self.advertised_window,
            'effective_window': self.effective_window,
            'bytes_in_flight': self.bytes_in_flight,
            'packets_in_flight': self.packets_in_flight,
            'rate_limit_bps': self.rate_limit_bps,
            'rate_limit_bytes_sent': self.rate_limit_bytes_sent,
            'stats': self.stats.copy()
        }
        
        # Add congestion control stats
        if self.congestion_controller:
            stats['congestion_control'] = self.congestion_controller.get_stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset flow control statistics."""
        self.stats = {
            'packets_sent': 0,
            'packets_dropped': 0,
            'bytes_sent': 0,
            'rate_limit_hits': 0,
            'window_full_events': 0
        }
        
        if self.congestion_controller:
            self.congestion_controller.stats = {
                'congestion_events': 0,
                'slow_start_exits': 0,
                'fast_recoveries': 0,
                'timeouts': 0,
                'window_reductions': 0
            }


class AdaptiveFlowController:
    """Adaptive flow controller that adjusts based on network conditions."""
    
    def __init__(self, initial_window: int = 10, max_window: int = 1024):
        self.flow_manager = FlowControlManager(max_window, enable_congestion_control=True)
        self.initial_window = initial_window
        self.max_window = max_window
        
        # Network condition tracking
        self.loss_rate = 0.0
        self.avg_rtt = 0.0
        self.bandwidth_estimate = 0.0
        
        # Adaptive parameters
        self.adaptation_interval = 5.0  # seconds
        self.last_adaptation = time.time()
        self.measurement_window = deque(maxlen=100)
        
        # Performance history
        self.performance_history = deque(maxlen=20)
        
    def adapt_to_conditions(self) -> None:
        """Adapt flow control parameters based on network conditions."""
        current_time = time.time()
        
        if current_time - self.last_adaptation < self.adaptation_interval:
            return
        
        # Calculate current performance metrics
        stats = self.flow_manager.get_stats()
        
        # Calculate loss rate
        total_packets = stats['stats']['packets_sent']
        dropped_packets = stats['stats']['packets_dropped']
        
        if total_packets > 0:
            self.loss_rate = dropped_packets / total_packets
        
        # Adapt window size based on conditions
        if self.loss_rate > 0.05:  # High loss rate
            # Reduce window size
            new_window = max(self.initial_window, int(self.flow_manager.advertised_window * 0.8))
            self.flow_manager.set_window_size(new_window)
            logger.info(f"High loss rate ({self.loss_rate:.2%}), reducing window to {new_window}")
        elif self.loss_rate < 0.01:  # Low loss rate
            # Increase window size
            new_window = min(self.max_window, int(self.flow_manager.advertised_window * 1.2))
            self.flow_manager.set_window_size(new_window)
            logger.info(f"Low loss rate ({self.loss_rate:.2%}), increasing window to {new_window}")
        
        # Record performance
        performance_metric = {
            'timestamp': current_time,
            'loss_rate': self.loss_rate,
            'avg_rtt': self.avg_rtt,
            'window_size': self.flow_manager.advertised_window,
            'throughput': stats['stats']['bytes_sent'] / self.adaptation_interval
        }
        
        self.performance_history.append(performance_metric)
        
        # Update adaptation timestamp
        self.last_adaptation = current_time
        
        # Reset stats for next measurement window
        self.flow_manager.reset_stats()
    
    def get_performance_history(self) -> List[Dict[str, any]]:
        """Get performance history."""
        return list(self.performance_history)
    
    def get_current_conditions(self) -> Dict[str, any]:
        """Get current network conditions."""
        return {
            'loss_rate': self.loss_rate,
            'avg_rtt': self.avg_rtt,
            'bandwidth_estimate': self.bandwidth_estimate,
            'adaptation_interval': self.adaptation_interval,
            'window_size': self.flow_manager.advertised_window,
            'effective_window': self.flow_manager.effective_window
        }
