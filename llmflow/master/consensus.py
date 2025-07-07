"""
LLMFlow Master Queue Consensus

This module implements the distributed consensus system for LLMFlow optimization decisions.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib

from ..queue import QueueManager
from ..molecules.optimization import OptimizationRecommendation

logger = logging.getLogger(__name__)


class ConsensusState(Enum):
    """States of consensus process."""
    PENDING = "pending"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class VoteType(Enum):
    """Types of votes."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class ConsensusNode:
    """A node in the consensus system."""
    node_id: str
    name: str
    node_type: str  # 'master', 'conductor', 'validator'
    trust_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    vote_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_trust_score(self, correct_votes: int, total_votes: int) -> None:
        """Update trust score based on voting accuracy."""
        if total_votes > 0:
            accuracy = correct_votes / total_votes
            # Adjust trust score based on accuracy
            self.trust_score = max(0.1, min(2.0, accuracy * 1.5))


@dataclass
class Vote:
    """A vote in the consensus process."""
    vote_id: str
    node_id: str
    proposal_id: str
    vote_type: VoteType
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None


@dataclass
class ConsensusProposal:
    """A proposal for consensus."""
    proposal_id: str
    proposal_type: str
    title: str
    description: str
    proposer_id: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    state: ConsensusState = ConsensusState.PENDING
    votes: List[Vote] = field(default_factory=list)
    required_votes: int = 3
    approval_threshold: float = 0.6
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_vote_summary(self) -> Dict[str, Any]:
        """Get summary of votes."""
        approve_votes = [v for v in self.votes if v.vote_type == VoteType.APPROVE]
        reject_votes = [v for v in self.votes if v.vote_type == VoteType.REJECT]
        abstain_votes = [v for v in self.votes if v.vote_type == VoteType.ABSTAIN]
        
        return {
            'total_votes': len(self.votes),
            'approve_votes': len(approve_votes),
            'reject_votes': len(reject_votes),
            'abstain_votes': len(abstain_votes),
            'approval_ratio': len(approve_votes) / len(self.votes) if self.votes else 0
        }
    
    def is_expired(self) -> bool:
        """Check if proposal has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


class ConsensusManager:
    """Manages distributed consensus for optimization decisions."""
    
    def __init__(self, queue_manager: QueueManager, node_id: str = None):
        self.queue_manager = queue_manager
        self.node_id = node_id or str(uuid.uuid4())
        self.running = False
        
        # Consensus state
        self.nodes: Dict[str, ConsensusNode] = {}
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.my_votes: Dict[str, Vote] = {}
        
        # Configuration
        self.config = {
            'vote_timeout_minutes': 30,
            'proposal_expiry_hours': 24,
            'min_nodes_for_consensus': 3,
            'trust_score_threshold': 0.5,
            'heartbeat_interval': 60.0,
            'consensus_check_interval': 30.0,
            'node_timeout_minutes': 5
        }
        
        # Background tasks
        self._consensus_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.consensus_stats = {
            'total_proposals': 0,
            'approved_proposals': 0,
            'rejected_proposals': 0,
            'expired_proposals': 0,
            'active_nodes': 0,
            'total_votes_cast': 0
        }
        
        # Register self as a node
        self.nodes[self.node_id] = ConsensusNode(
            node_id=self.node_id,
            name=f"master_{self.node_id[:8]}",
            node_type="master"
        )
    
    async def start(self) -> None:
        """Start the consensus manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self._consensus_task = asyncio.create_task(self._consensus_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Send initial heartbeat
        await self._send_heartbeat()
        
        logger.info(f"Consensus manager started with node ID: {self.node_id}")
    
    async def stop(self) -> None:
        """Stop the consensus manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in [self._consensus_task, self._heartbeat_task, self._cleanup_task]:
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            *[task for task in [self._consensus_task, self._heartbeat_task, self._cleanup_task] if task],
            return_exceptions=True
        )
        
        logger.info("Consensus manager stopped")
    
    async def propose_optimization(self, optimization: OptimizationRecommendation) -> str:
        """Propose an optimization for consensus."""
        proposal_id = str(uuid.uuid4())
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposal_type="optimization",
            title=f"Optimize {optimization.target_component}",
            description=optimization.description,
            proposer_id=self.node_id,
            data={
                'optimization': optimization.to_dict(),
                'target_component': optimization.target_component,
                'recommendation_type': optimization.recommendation_type,
                'expected_improvement': optimization.expected_improvement,
                'confidence_score': optimization.confidence_score
            },
            expires_at=datetime.utcnow() + timedelta(hours=self.config['proposal_expiry_hours']),
            required_votes=max(3, len(self.nodes) // 2 + 1),
            approval_threshold=0.6
        )
        
        # Store proposal
        self.proposals[proposal_id] = proposal
        self.consensus_stats['total_proposals'] += 1
        
        # Broadcast proposal
        await self._broadcast_proposal(proposal)
        
        logger.info(f"Proposed optimization for consensus: {proposal_id}")
        return proposal_id
    
    async def vote_on_proposal(self, proposal_id: str, vote_type: VoteType, 
                              confidence: float, reasoning: str) -> bool:
        """Vote on a proposal."""
        if proposal_id not in self.proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.state != ConsensusState.PENDING and proposal.state != ConsensusState.VOTING:
            logger.warning(f"Proposal {proposal_id} is not in voting state")
            return False
        
        # Check if already voted
        if proposal_id in self.my_votes:
            logger.warning(f"Already voted on proposal {proposal_id}")
            return False
        
        # Create vote
        vote = Vote(
            vote_id=str(uuid.uuid4()),
            node_id=self.node_id,
            proposal_id=proposal_id,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Sign vote
        vote.signature = await self._sign_vote(vote)
        
        # Store vote
        self.my_votes[proposal_id] = vote
        proposal.votes.append(vote)
        self.consensus_stats['total_votes_cast'] += 1
        
        # Update proposal state
        proposal.state = ConsensusState.VOTING
        
        # Broadcast vote
        await self._broadcast_vote(vote)
        
        logger.info(f"Voted on proposal {proposal_id}: {vote_type.value}")
        return True
    
    async def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a proposal."""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        vote_summary = proposal.get_vote_summary()
        
        return {
            'proposal_id': proposal.proposal_id,
            'proposal_type': proposal.proposal_type,
            'title': proposal.title,
            'description': proposal.description,
            'proposer_id': proposal.proposer_id,
            'created_at': proposal.created_at.isoformat(),
            'expires_at': proposal.expires_at.isoformat() if proposal.expires_at else None,
            'state': proposal.state.value,
            'required_votes': proposal.required_votes,
            'approval_threshold': proposal.approval_threshold,
            'vote_summary': vote_summary,
            'is_expired': proposal.is_expired()
        }
    
    async def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics."""
        active_nodes = len([n for n in self.nodes.values() if n.active])
        
        return {
            'node_id': self.node_id,
            'active_nodes': active_nodes,
            'total_nodes': len(self.nodes),
            'active_proposals': len([p for p in self.proposals.values() if p.state in [ConsensusState.PENDING, ConsensusState.VOTING]]),
            'total_proposals': len(self.proposals),
            'stats': self.consensus_stats.copy(),
            'config': self.config.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _consensus_loop(self) -> None:
        """Background consensus processing loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['consensus_check_interval'])
                
                # Process pending proposals
                await self._process_proposals()
                
                # Process incoming votes
                await self._process_votes()
                
                # Update node statuses
                await self._update_node_statuses()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consensus loop: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config['heartbeat_interval'])
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired_proposals()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _process_proposals(self) -> None:
        """Process pending proposals."""
        for proposal in self.proposals.values():
            if proposal.state == ConsensusState.PENDING:
                # Start voting if we have enough nodes
                if len(self.nodes) >= self.config['min_nodes_for_consensus']:
                    proposal.state = ConsensusState.VOTING
                    logger.info(f"Started voting on proposal {proposal.proposal_id}")
            
            elif proposal.state == ConsensusState.VOTING:
                # Check if we have enough votes
                if len(proposal.votes) >= proposal.required_votes:
                    await self._finalize_proposal(proposal)
                
                # Check if expired
                elif proposal.is_expired():
                    proposal.state = ConsensusState.EXPIRED
                    self.consensus_stats['expired_proposals'] += 1
                    logger.info(f"Proposal {proposal.proposal_id} expired")
    
    async def _finalize_proposal(self, proposal: ConsensusProposal) -> None:
        """Finalize a proposal based on votes."""
        vote_summary = proposal.get_vote_summary()
        
        # Calculate weighted approval ratio (considering trust scores)
        total_weight = 0
        approval_weight = 0
        
        for vote in proposal.votes:
            if vote.node_id in self.nodes:
                node = self.nodes[vote.node_id]
                weight = node.trust_score * vote.confidence
                total_weight += weight
                
                if vote.vote_type == VoteType.APPROVE:
                    approval_weight += weight
        
        weighted_approval_ratio = approval_weight / total_weight if total_weight > 0 else 0
        
        # Make decision
        if weighted_approval_ratio >= proposal.approval_threshold:
            proposal.state = ConsensusState.APPROVED
            self.consensus_stats['approved_proposals'] += 1
            
            # Execute the proposal
            await self._execute_proposal(proposal)
            
            logger.info(f"Proposal {proposal.proposal_id} approved (ratio: {weighted_approval_ratio:.2f})")
        else:
            proposal.state = ConsensusState.REJECTED
            self.consensus_stats['rejected_proposals'] += 1
            
            logger.info(f"Proposal {proposal.proposal_id} rejected (ratio: {weighted_approval_ratio:.2f})")
    
    async def _execute_proposal(self, proposal: ConsensusProposal) -> None:
        """Execute an approved proposal."""
        try:
            if proposal.proposal_type == "optimization":
                # Send approval to optimization execution queue
                await self.queue_manager.enqueue(
                    'system.optimization.approved',
                    {
                        'proposal_id': proposal.proposal_id,
                        'optimization': proposal.data['optimization'],
                        'consensus_result': {
                            'approved': True,
                            'vote_summary': proposal.get_vote_summary(),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    },
                    domain='system'
                )
                
                logger.info(f"Executed optimization proposal {proposal.proposal_id}")
        
        except Exception as e:
            logger.error(f"Error executing proposal {proposal.proposal_id}: {e}")
    
    async def _process_votes(self) -> None:
        """Process incoming votes from other nodes."""
        try:
            # Check for votes in the consensus queue
            vote_data = await self.queue_manager.dequeue('system.consensus.votes')
            
            if vote_data:
                await self._handle_incoming_vote(vote_data)
        
        except Exception as e:
            logger.error(f"Error processing votes: {e}")
    
    async def _handle_incoming_vote(self, vote_data: Dict[str, Any]) -> None:
        """Handle an incoming vote from another node."""
        try:
            # Validate vote
            if not self._validate_vote(vote_data):
                logger.warning("Invalid vote received")
                return
            
            # Create vote object
            vote = Vote(
                vote_id=vote_data['vote_id'],
                node_id=vote_data['node_id'],
                proposal_id=vote_data['proposal_id'],
                vote_type=VoteType(vote_data['vote_type']),
                confidence=vote_data['confidence'],
                reasoning=vote_data['reasoning'],
                timestamp=datetime.fromisoformat(vote_data['timestamp']),
                signature=vote_data.get('signature')
            )
            
            # Add to proposal if it exists
            if vote.proposal_id in self.proposals:
                proposal = self.proposals[vote.proposal_id]
                
                # Check if vote already exists
                existing_vote = any(v.vote_id == vote.vote_id for v in proposal.votes)
                
                if not existing_vote:
                    proposal.votes.append(vote)
                    logger.info(f"Added vote from {vote.node_id} for proposal {vote.proposal_id}")
        
        except Exception as e:
            logger.error(f"Error handling incoming vote: {e}")
    
    def _validate_vote(self, vote_data: Dict[str, Any]) -> bool:
        """Validate a vote."""
        required_fields = ['vote_id', 'node_id', 'proposal_id', 'vote_type', 'confidence', 'reasoning']
        
        for field in required_fields:
            if field not in vote_data:
                return False
        
        # Validate vote type
        try:
            VoteType(vote_data['vote_type'])
        except ValueError:
            return False
        
        # Validate confidence
        if not (0 <= vote_data['confidence'] <= 1):
            return False
        
        return True
    
    async def _broadcast_proposal(self, proposal: ConsensusProposal) -> None:
        """Broadcast a proposal to other nodes."""
        try:
            proposal_data = {
                'proposal_id': proposal.proposal_id,
                'proposal_type': proposal.proposal_type,
                'title': proposal.title,
                'description': proposal.description,
                'proposer_id': proposal.proposer_id,
                'data': proposal.data,
                'created_at': proposal.created_at.isoformat(),
                'expires_at': proposal.expires_at.isoformat() if proposal.expires_at else None,
                'required_votes': proposal.required_votes,
                'approval_threshold': proposal.approval_threshold
            }
            
            await self.queue_manager.enqueue(
                'system.consensus.proposals',
                proposal_data,
                domain='system'
            )
        
        except Exception as e:
            logger.error(f"Error broadcasting proposal: {e}")
    
    async def _broadcast_vote(self, vote: Vote) -> None:
        """Broadcast a vote to other nodes."""
        try:
            vote_data = {
                'vote_id': vote.vote_id,
                'node_id': vote.node_id,
                'proposal_id': vote.proposal_id,
                'vote_type': vote.vote_type.value,
                'confidence': vote.confidence,
                'reasoning': vote.reasoning,
                'timestamp': vote.timestamp.isoformat(),
                'signature': vote.signature
            }
            
            await self.queue_manager.enqueue(
                'system.consensus.votes',
                vote_data,
                domain='system'
            )
        
        except Exception as e:
            logger.error(f"Error broadcasting vote: {e}")
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to other nodes."""
        try:
            heartbeat_data = {
                'node_id': self.node_id,
                'timestamp': datetime.utcnow().isoformat(),
                'node_type': 'master',
                'active_proposals': len([p for p in self.proposals.values() if p.state in [ConsensusState.PENDING, ConsensusState.VOTING]]),
                'total_votes_cast': self.consensus_stats['total_votes_cast']
            }
            
            await self.queue_manager.enqueue(
                'system.consensus.heartbeat',
                heartbeat_data,
                domain='system'
            )
        
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    async def _update_node_statuses(self) -> None:
        """Update node statuses based on heartbeats."""
        try:
            # Process heartbeats
            heartbeat_data = await self.queue_manager.dequeue('system.consensus.heartbeat')
            
            if heartbeat_data:
                node_id = heartbeat_data['node_id']
                
                if node_id != self.node_id:  # Don't process our own heartbeat
                    if node_id not in self.nodes:
                        # New node
                        self.nodes[node_id] = ConsensusNode(
                            node_id=node_id,
                            name=f"node_{node_id[:8]}",
                            node_type=heartbeat_data.get('node_type', 'unknown')
                        )
                    
                    # Update last seen
                    self.nodes[node_id].last_seen = datetime.utcnow()
                    self.nodes[node_id].active = True
            
            # Mark inactive nodes
            timeout_threshold = datetime.utcnow() - timedelta(minutes=self.config['node_timeout_minutes'])
            
            for node in self.nodes.values():
                if node.last_seen < timeout_threshold:
                    node.active = False
            
            # Update stats
            self.consensus_stats['active_nodes'] = len([n for n in self.nodes.values() if n.active])
        
        except Exception as e:
            logger.error(f"Error updating node statuses: {e}")
    
    async def _cleanup_expired_proposals(self) -> None:
        """Clean up expired proposals."""
        expired_proposals = []
        
        for proposal in self.proposals.values():
            if proposal.is_expired() and proposal.state not in [ConsensusState.APPROVED, ConsensusState.REJECTED]:
                proposal.state = ConsensusState.EXPIRED
                expired_proposals.append(proposal.proposal_id)
        
        if expired_proposals:
            logger.info(f"Cleaned up {len(expired_proposals)} expired proposals")
    
    async def _sign_vote(self, vote: Vote) -> str:
        """Sign a vote (simplified implementation)."""
        # In a real implementation, this would use cryptographic signing
        vote_data = f"{vote.vote_id}{vote.node_id}{vote.proposal_id}{vote.vote_type.value}{vote.confidence}"
        return hashlib.sha256(vote_data.encode()).hexdigest()
