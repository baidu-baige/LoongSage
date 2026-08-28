"""Trajectory data models and store for AgentFlow."""

from enum import Enum
import logging
from typing import Any, Literal

from pydantic import BaseModel as PydanticModel, Field

logger = logging.getLogger(__name__)


class TrajectoryStatus(str, Enum):
    """Lifecycle states for a Trajectory.

    State transitions (normal path):
        PENDING → GENERATING → COMPLETED

    Error paths:
        PENDING → GENERATING → FAILED   (all retries exhausted)
        PENDING → GENERATING → ABORTED  (external cancellation)
    """

    # Created and registered in the store; not yet dispatched to the LLM.
    PENDING = "pending"

    # LLM is actively producing tokens for this trajectory.
    GENERATING = "generating"

    # Generation finished successfully and reward has been stored.
    COMPLETED = "completed"

    # All retry attempts exhausted or an unrecoverable error occurred.
    FAILED = "failed"

    # Explicitly cancelled by the AgentFlow (see AgentFlow's shutdown/abort path,
    # which marks all non-terminal trajectories as ABORTED before exiting).
    ABORTED = "aborted"


class Triplet(PydanticModel):
    """One LLM interaction turn, stored as index ranges into the parent Trajectory's arrays.

    token_start = segment.token_start for all turns after the first in a segment.
    This means tokens[token_start:token_end] covers the full context from the segment
    start — including all prior turns' tokens (p1+a1+...+pN+aN).

    Usage:
        context_this_turn  = trajectory.tokens[t.token_start : t.token_end]
        logprobs_this_turn = trajectory.rollout_log_probs[t.logprob_start : t.logprob_end]
        loss_this_turn     = trajectory.loss_masks[t.logprob_start : t.logprob_end]
    """

    # Start index in trajectory.tokens (inclusive).
    # For the first turn in a segment: equals the delta_prompt start position.
    # For subsequent turns in the same segment: equals segment.token_start,
    # so the slice covers the full accumulated context from segment start.
    token_start: int

    # End index in trajectory.tokens (exclusive). Always the end of this turn's response.
    token_end: int

    # Start index in the response_length space (loss_masks / rollout_log_probs) for this
    # turn's LLM-generated response tokens (inclusive).  Skips any tool-output tokens
    # that precede this turn's response in the response_length array.
    logprob_start: int

    # End index in the response_length space for this turn's LLM response (exclusive).
    logprob_end: int

    # Reserved: per-triplet reward.
    reward: float | None = None

    # Optional debug fields, e.g. response_text.
    metadata: dict[str, Any] = Field(default_factory=dict)


class Segment(PydanticModel):
    """A node in the trajectory's segment tree.

    In the common (white-box, no compaction, no subagent) case this tree degenerates into
    a single linear chain — identical to today's behavior. Segments branch when the agent's
    prompt is no longer a prefix extension of the active segment:

      * origin="compact"  — context compaction (e.g. auto-compact). The new segment carries
        a fresh prompt (usually "summary + recent turns") that does not share a token prefix
        with the old segment, but is still the SAME timeline continuing — not a parallel branch.
      * origin="subagent" — the agent forked a subtask with its own independent message
        sequence. This is a genuine tree branch: the mainline resumes from the parent segment
        once the subagent finishes, and the subagent's own turns never reappear as a mainline
        prefix. Currently a placeholder only: no tokens/loss are stored for it (see
        `trainable`/`token_start`/`token_end` below); it exists purely to reserve the tree
        shape for a future iteration that does train on subagent content.

    `trajectory.segments` remains a flat list (for zero-cost pydantic / Ray serialization and
    to keep `data_processor.py`'s `for seg in traj.segments` iteration unchanged); the tree is
    encoded via `segment_id` / `parent_segment_id` as a parent pointer, flattened into an
    indexable list instead of a linked object graph.
    """

    triplets: list[Triplet] = Field(default_factory=list)

    # = triplets[0].token_start
    token_start: int = 0

    # = triplets[-1].token_end  (updated as triplets are appended)
    token_end: int = 0

    # = triplets[0].logprob_start
    logprob_start: int = 0

    # = triplets[-1].logprob_end  (updated as triplets are appended)
    logprob_end: int = 0

    # Reserved: segment-level reward (None = inherit trajectory.reward).
    reward: float | None = None

    # --- Tree addressing ---

    # This segment's own index into trajectory.segments. Assigned by the parser when the
    # segment is created; stable for the segment's lifetime.
    segment_id: int = 0

    # Parent segment's segment_id. None only for the trajectory's very first segment.
    # For origin="compact": the segment whose context this compaction replaced.
    # For origin="subagent": the mainline (or ancestor subagent) segment this branch forked from.
    parent_segment_id: int | None = None

    # Why this segment was opened:
    #   "root"     — the trajectory's first segment.
    #   "compact"  — context compaction; mainline continuation, NOT a parallel branch.
    #   "subagent" — a forked subtask branch; a genuine tree branch under parent_segment_id.
    origin: Literal["root", "compact", "subagent"] = "root"

    # Nesting depth. 0 for mainline segments (root/compact); parent.depth + 1 for a subagent
    # branch, so nested subagents (subagent calling a subagent) are distinguishable.
    depth: int = 0

    # Whether this segment's tokens participate in loss. Defaults to True for mainline
    # (root/compact) segments. For origin="subagent", currently always False: subagent
    # branches are not trained on yet — the Segment node is created purely as a tree-shape
    # placeholder (see token_start/token_end above), so a later iteration can flip this to
    # True and start storing real token/loss data without changing the tree schema.
    trainable: bool = True


class Trajectory(PydanticModel):
    """A complete rollout trajectory consisting of one or more Segments.

    Three flat arrays span the entire Trajectory; Segment / Triplet store index ranges:

        tokens             length = initial_prompt_len + response_len
        loss_masks         length = response_len   (final train mask: 1 = include in loss;
                                                    0 = exclude, e.g. tool/rebuilt context
                                                    or off-policy partial-rollout prefix)
        rollout_log_probs  length = response_len   (LLM tokens=real logprob, rest=0.0)
        token_rewards      length = response_len   (all 0.0 except reward position(s))

    Index spaces:
      - token_start / token_end     → index into `tokens`  (full sequence)
      - logprob_start / logprob_end → index into response space
                                      (loss_masks / rollout_log_probs / token_rewards)

    Adjacent Triplets are NOT contiguous in logprob space — tool-output tokens and
    rebuilt-prompt tokens occupy positions between them (loss_mask=0, logprob=0.0).

    Concrete token-level example (matching the diagram above):

    Symbol legend:
        P1     = original prompt        (3 tokens)
        R1     = LLM Turn 1 response    (2 tokens)
        O1     = tool observation       (2 tokens)
        Vm     = summary instruction    (1 token)
        S1     = LLM-generated summary  (2 tokens)
        P2     = rebuilt prompt        (3 tokens)
        R2     = LLM Turn 2 in Seg 1   (2 tokens)

    tokens (flat, len=17):
        seg0_tokens = [ P1  P1  P1 | R1 R1 | O1 O1 | Vm | S1  S1 ]      
        seg0_response = [ R1 R1 | O1 O1 | Vm | S1  S1 ]        
        seg0_loss_mask = [ 1  1 | 0  0 | 0 | 1  1 ] 
        seg0_log_probs = [ lp lp | 0.0 0.0 | 0.0 | lp lp ] 

        seg1_tokens = [ P2  P2  P2 | S1  S1  | R2 R2 ]
        seg1_response = [ R2 R2 ]
        seg1_loss_mask = [ 1  1 ] 
        seg1_log_probs = [ lp lp]

        traj_tokens = seq0_tokens + seq1_tokens
             = [ P1  P1  P1 | R1 R1 | O1 O1 | Vm | S1  S1  |  P2  P2  P2 | S1  S1  | R2 R2 ]
        traj_response = seg0_response + seg1_tokens
             = [ R1 R1 | O1 O1 | Vm | S1  S1  |  P2  P2  P2 | S1  S1  | R2 R2 ]
        traj_loss_mask = [ 1  1 | 0  0 | 0 | 1  1  |   0  0  0 | 0  0  | 1  1 ]
        traj_log_prob = [ lp lp | 0.0 0.0 | 0.0 | lp lp |  0.0 0.0 0.0 | 0.0 0.0 | lp lp ]
        token_rewards = [ 0   0    0   0    0    0   0    |  0   0   0     0   0     0   x  ]

        Notice: traj_response = seg0_response + seg1_all (NOT seg1_response).

    segments=[
        Segment(token_start=0, token_end=10, logprob_start=0, logprob_end=7,
            triplets=[
                # Triplet 0: first turn — tokens[0:5] = P1(3)+R1(2); logprob covers R1 at [0, 2)
                Triplet(token_start=0, token_end=5,
                        logprob_start=0, logprob_end=2),
                # Triplet 1: second turn — token_start=seg.token_start=0, so tokens[0:10] =
                #   P1+R1+O1+Vm+S1 (full context from segment start); logprob covers S1 at [5, 7)
                Triplet(token_start=0, token_end=10,
                        logprob_start=5, logprob_end=7),
            ],
        ),
        # Seg 1: tokens[10:17] = P2(3)+S1(2)+R2(2).
        Segment(token_start=10, token_end=17, logprob_start=12, logprob_end=14,
            triplets=[
                # Triplet 0: tokens[10:17] = P2(3)+S1(2)+R2(2);
                #   logprob covers R2 at [12, 14); gap [7, 12) = P2+S1 (loss_mask=0)
                Triplet(token_start=10, token_end=17,
                        logprob_start=12, logprob_end=14),
            ],
        ),
    ]
    """

    # --- Identity ---
    trajectory_id: str = ""

    # Structured group key, e.g. "epoch{N}_step{N}_ds{N}[_eval]_prompt{N}".
    prompt_id: str = ""

    # Index into data_sources; used to select per-data-source agent/reward/token config.
    ds_index: int = 0

    # True for eval-dataset trajectories (routed out of the training batch).
    is_eval: bool = False

    prompt: str | list[dict[str, str]] | dict[str, Any] = ""
    label: Any = None

    # Retry counter for this trajectory (0 = first attempt).
    attempt_id: int = 0

    # First / last weight version under which this trajectory produced a generate response,
    # across its whole lifetime (including partial-rollout resumes). Stamped per-turn in
    # update_trajectory from payload.weight_version — independent of whether the turn emitted
    # tokens — so a version crossing is always recorded even for a 0-token resume turn, and
    # they survive prune/restore that mutate rollout_weight_versions.
    # span = end - start measures how far this trajectory drifted off-policy. -1 = unset.
    start_rollout_weight_version: int = -1
    end_rollout_weight_version: int = -1

    # --- Core training arrays ---
    # All token ids concatenated across turns (initial_prompt + all subsequent tokens).
    # Length = initial_prompt_len + response_len.
    tokens: list[int] = Field(default_factory=list)

    # Final train mask in response space: 1 = include this token in loss, 0 = ignore it.
    # Zeros may come from non-trainable context tokens (tool outputs / rebuilt prompts /
    # re-appearing context after a Summary Reset) or from policy-validity filters such as
    # partial-rollout off-policy masking. Code that prunes or slices response tokens must
    # preserve existing mask values instead of recreating masks from token type alone.
    # Length = response_len = len(tokens) - initial_prompt_len.
    # Includes ALL tokens after the initial prompt — tool outputs and re-appearing
    # context tokens (P_dup, S_dup) from Summary Reset are present but with mask=0.
    loss_masks: list[int] = Field(default_factory=list)

    # Log-probabilities, length = response_len (same as loss_masks).
    # LLM-generated positions: real log-probability (≤ 0).
    # All other positions (tool outputs, re-appearing prompt/summary tokens): 0.0.
    rollout_log_probs: list[float] = Field(default_factory=list)

    # Weight version for each response-space token, aligned with rollout_log_probs.
    # Non-LLM positions use -1.
    rollout_weight_versions: list[int] = Field(default_factory=list)

    rollout_routed_experts: Any = None

    # --- Rewards ---
    # Final ORM scalar reward returned by the Agent.
    reward: float = 0.0

    # Answer correctness. Defaults to final_reward > 0; a reward function can override it
    # by returning Reward(is_correct=...) when shaping makes the sign misleading.
    is_correct: bool = False

    # Shape = len(rollout_log_probs) = response_len.
    # All zeros except the very last position = final_reward.
    token_rewards: list[float] = Field(default_factory=list)

    # --- Auxiliary ---
    # Plaintext messages grouped by segment, keyed by segment_id. chat_completions[sid]
    # holds the messages for the segment whose segment_id == sid. Keyed (not positional)
    # because subagent placeholder segments occupy a segments[] slot without producing a
    # chat entry, so segment_ids are not contiguous with append order. The chat segment at
    # active_segment_id is the live context used for resume/prefix detection.
    chat_completions: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)

    # True if this trajectory should be excluded from the training batch.
    masked_out: bool = False

    num_turns: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.PENDING

    # --- Segment / Triplet index (no token duplication) ---
    # Access a specific turn via segments[seg_i].triplets[turn_j].
    segments: list[Segment] = Field(default_factory=list)

    # segment_id of the segment currently being appended to. In the linear case this always
    # equals segments[-1].segment_id. When a subagent branch is open, it stays pointing at
    # the parent (mainline) segment the branch forked from — a subagent placeholder segment
    # is appended to `segments` but never becomes active. Lets the parser locate "which
    # segment do I append this turn's triplet to" in O(1) instead of re-walking the tree.
    active_segment_id: int = 0

class TrajectoryGroup(PydanticModel):
    """A group of Trajectory objects that are n rollout trajectories from the same prompt.

    All trajectories in the group share the same prompt and prompt_id.  The group is the minimal scheduling unit used
    by the training pipeline — it is never split across DP ranks.

    Attributes:
        prompt_id:    Identifier shared by every Trajectory in this group.
        trajectories: The n rollout trajectories produced for the prompt.
    """

    prompt_id: str
    trajectories: list[Trajectory] = Field(default_factory=list)

    @property
    def token_length(self) -> int:
        """Total number of tokens across all trajectories in this group."""
        return sum(len(t.tokens) for t in self.trajectories)

    @property
    def segment_count(self) -> int:
        """Number of trainable Segments across all trajectories in this group.

        Placeholder segments (``trainable=False``, e.g. subagent branches) carry
        no tokens and are dropped before the forward pass, so they must not skew
        DP load balancing.
        """
        return sum(1 for t in self.trajectories for s in t.segments if s.trainable)


class TrajectoryStore:
    """Store for Trajectory objects in AgentFlow.

    Serves as the single source of truth shared between AgentFlow and the Router's ParserMiddleware.

    trajectory_data layout:

      {
        "epoch0_step0_prompt0_traj0": [                                  # key = trajectory_id
            Trajectory(attempt_id=0, status=FAILED,    segments=[...]),  # attempt 0
            Trajectory(attempt_id=1, status=COMPLETED, segments=[...]),  # retry
        ],
        "epoch0_step0_prompt0_traj1": [
            Trajectory(attempt_id=0, status=COMPLETED, segments=[...]),
        ],
      }

    get() without attempt_id always returns the last element (latest attempt).
    get_trajectory_attempt_ids() returns [0, 1] for "epoch0_step0_prompt0_traj0" above.
    """

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.trajectory_data: dict[str, list[Trajectory]] = {}

    def update(self, trajectory_id: str, trajectory: Trajectory) -> None:
        """Replace the matching attempt for trajectory_id with the given trajectory.

        Matches by attempt_id to avoid overwriting a newer attempt when a stale background thread delivers a response
        for an older attempt. If no matching attempt_id is found, the update is silently ignored.
        """
        attempts = self.trajectory_data.get(trajectory_id, [])
        for i, t in enumerate(attempts):
            if t.attempt_id == trajectory.attempt_id:
                attempts[i] = trajectory
                return
        if not attempts:
            logger.warning("%s: No attempts found in store, cannot update", trajectory_id)

    def get(
        self, ids: list[str], attempt_id: int | None = None
    ) -> dict[str, list[Trajectory]]:
        """Return trajectories for the given ids.

        Args:
            ids: Trajectory ids to look up.
            attempt_id: If given, return only attempts whose attempt_id matches.
                        If None, return only the latest attempt for each id.

        Returns:
            Dict mapping trajectory_id to a list of Trajectory objects. When attempt_id is None, each list contains
            only the latest attempt. Ids with no stored trajectories are omitted from the result.

        Example:
            Store contains:
                "traj_0": [attempt_0, attempt_1, attempt_2]  # 3 attempts
                "traj_1": [attempt_0]                        # 1 attempt

            get(["traj_0", "traj_1"]) returns:
                {
                    "traj_0": [attempt_2],  # latest only
                    "traj_1": [attempt_0],
                }

            get(["traj_0", "traj_1"], attempt_id=0) returns:
                {
                    "traj_0": [attempt_0],  # matching attempt_id
                    "traj_1": [attempt_0],
                }
        """
        result: dict[str, list[Trajectory]] = {}
        for tid in ids:
            attempts = self.trajectory_data.get(tid, [])
            if not attempts:
                continue

            if attempt_id is None:
                # Return only the latest attempt
                result[tid] = [attempts[-1]]
            else:
                # Filter to matching attempt_id
                matches = [t for t in attempts if t.attempt_id == attempt_id]
                if matches:
                    result[tid] = matches

        return result

    def add(self, trajectory_id: str, trajectory: Trajectory) -> None:
        """Append a new attempt for trajectory_id to the store."""
        self.trajectory_data.setdefault(trajectory_id, []).append(trajectory)

    def get_trajectory_attempt_ids(self, ids: list[str]) -> dict[str, list[int]]:
        """Return all attempt_ids for the given trajectory ids in insertion order.

        Args:
            ids: Trajectory ids to query.

        Returns:
            Dict mapping trajectory_id to a list of attempt_ids in insertion order.
            Ids with no stored trajectories are omitted from the result.
        """
        return {
            tid: [t.attempt_id for t in attempts]
            for tid in ids
            if (attempts := self.trajectory_data.get(tid))
        }

    def delete(self, ids: list[str]) -> None:
        """Remove all attempts for the given trajectory ids from the store."""
        for tid in ids:
            self.trajectory_data.pop(tid, None)

    def clear(self) -> None:
        """Remove all trajectories from the store."""
        self.trajectory_data.clear()
