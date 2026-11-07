"""Trajectory data models and store for AgentFlow."""

from array import array
from enum import Enum
import logging
from typing import Any, Literal

from pydantic import BaseModel as PydanticModel, Field
from pydantic_core import core_schema

logger = logging.getLogger(__name__)


class _PackedSeq(array):
    """Per-token sequence packed in an ``array`` that behaves like a ``list``.

    A list of N ints is N GC-visible pointers that every full collection walks; with
    long-context trajectories that made gen2 pauses take seconds. Slices read as lists,
    writes accept any iterable, ``==`` compares with lists, and copy/pickle keep the type.
    """

    __slots__ = ()
    _TYPECODE = "q"

    def __new__(cls, values: Any = ()):
        if isinstance(values, array) and values.typecode == cls._TYPECODE:
            obj = super().__new__(cls, cls._TYPECODE)
            obj.frombytes(values.tobytes())
            return obj
        return super().__new__(cls, cls._TYPECODE, values)

    @classmethod
    def _from_bytes(cls, data: bytes) -> "_PackedSeq":
        obj = super().__new__(cls, cls._TYPECODE)
        obj.frombytes(data)
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            return super().__getitem__(index).tolist()
        return super().__getitem__(index)

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice) and not (
            isinstance(value, array) and value.typecode == self._TYPECODE
        ):
            value = array(self._TYPECODE, value)
        super().__setitem__(index, value)

    def extend(self, values) -> None:
        """Append values, converting a non-matching iterable to this packed typecode first."""
        if not (isinstance(values, array) and values.typecode == self._TYPECODE):
            values = array(self._TYPECODE, values)
        super().extend(values)

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __add__(self, other):
        return self.tolist() + list(other)

    def __radd__(self, other):
        return list(other) + self.tolist()

    def __eq__(self, other) -> bool:
        if isinstance(other, array):
            return array.__eq__(self, other)
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and self.tolist() == list(other)
        return NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = None

    def __repr__(self) -> str:
        return repr(self.tolist())

    def __copy__(self):
        return type(self)(self)

    def __deepcopy__(self, memo):
        return type(self)(self)

    def __reduce_ex__(self, protocol):
        # array's own __reduce_ex__ rebuilds via cls(typecode, list) for protocol < 3.
        return (type(self)._from_bytes, (self.tobytes(),))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Pydantic schema: validate incoming values into this type, serialize as a plain list."""
        def validate(value):
            return value if type(value) is cls else cls(value)

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: v.tolist()),
        )


class IntSeq(_PackedSeq):
    """Packed int64 sequence (token ids, masks, weight versions)."""

    __slots__ = ()
    _TYPECODE = "q"


class FloatSeq(_PackedSeq):
    """Packed float64 sequence (log-probs, token rewards)."""

    __slots__ = ()
    _TYPECODE = "d"


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

    # Half-open range in trajectory.tokens. The first turn of a segment starts at its delta prompt; later
    # turns start at segment.token_start, so the slice covers the full context. token_end is the end of
    # this turn's response.
    token_start: int
    token_end: int

    # Half-open range in response space (loss_masks / rollout_log_probs) covering only this turn's LLM
    # response; tool-output tokens before it are skipped.
    logprob_start: int
    logprob_end: int

    # Reserved: per-triplet reward.
    reward: float | None = None

    # Optional debug fields, e.g. response_text.
    metadata: dict[str, Any] = Field(default_factory=dict)


class Segment(PydanticModel):
    """A node in the trajectory's segment tree.

    Normally (white-box, no compaction, no subagent) the tree is a single linear chain. A new segment
    starts when the agent's prompt is no longer a prefix extension of the active segment:

      * origin="compact": context compaction (e.g. auto-compact). The new prompt (usually "summary +
        recent turns") shares no token prefix with the old segment, but it is the same timeline continuing.
      * origin="subagent": the agent forked a subtask with its own message sequence, a real branch. The
        mainline resumes from the parent segment when the subagent finishes, and the subagent's own turns
        never reappear as a mainline prefix. Currently a placeholder: no tokens or loss are stored for it
        (see `trainable` / `token_start` / `token_end`), it only reserves the tree shape for training on
        subagent content later.

    `trajectory.segments` stays a flat list (cheap pydantic / Ray serialization, and `data_processor.py`
    keeps iterating `for seg in traj.segments`). The tree is encoded as a parent pointer
    (`segment_id` / `parent_segment_id`) in that list, not as a linked object graph.
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

    # Index into trajectory.segments, assigned by the parser at creation; stable for the segment's lifetime.
    segment_id: int = 0

    # segment_id of the parent; None only for the first segment.
    # "compact": the segment whose context was replaced.
    # "subagent": the mainline (or ancestor subagent) segment this branch forked from.
    parent_segment_id: int | None = None

    # Why this segment was opened:
    #   "root"     : the trajectory's first segment.
    #   "compact"  : context compaction, a mainline continuation (not a parallel branch).
    #   "subagent" : a forked subtask, a real branch under parent_segment_id.
    origin: Literal["root", "compact", "subagent"] = "root"

    # 0 for mainline segments (root/compact), parent.depth + 1 for subagent branches, so nested
    # subagents (a subagent calling a subagent) are distinguishable.
    depth: int = 0

    # Whether this segment's tokens participate in loss. True by default (mainline root/compact segments);
    # always False for origin="subagent" for now (tree-shape placeholder, see the class docstring), and
    # flipping it later needs no schema change.
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

    # Dataset row at input_key: the prompt (RL) or the full conversation (SFT).
    prompt: str | list[dict[str, Any]] | dict[str, Any] = ""
    label: Any = None

    # Retry counter for this trajectory (0 = first attempt).
    attempt_id: int = 0

    # First / last weight version under which this trajectory produced a generate response, across its
    # whole lifetime (partial-rollout resumes included). Stamped per turn in update_trajectory from
    # payload.weight_version, independent of whether the turn emitted tokens, so a version crossing is
    # recorded even for a 0-token resume turn. They also survive prune/restore, which mutate
    # rollout_weight_versions. end - start shows how far it drifted off-policy. -1 = not set yet.
    start_rollout_weight_version: int = -1
    end_rollout_weight_version: int = -1

    # --- Core training arrays ---
    # All token ids concatenated across turns (initial_prompt + all subsequent tokens).
    # Length = initial_prompt_len + response_len.
    tokens: IntSeq = Field(default_factory=IntSeq)

    # Final train mask in response space: 1 = include this token in loss, 0 = ignore it.
    # Zeros may come from non-trainable context tokens (tool outputs / rebuilt prompts /
    # re-appearing context after a Summary Reset) or from policy-validity filters such as
    # partial-rollout off-policy masking. Code that prunes or slices response tokens must
    # preserve existing mask values instead of recreating masks from token type alone.
    # Length = response_len = len(tokens) - initial_prompt_len.
    # Includes ALL tokens after the initial prompt — tool outputs and re-appearing
    # context tokens (P_dup, S_dup) from Summary Reset are present but with mask=0.
    loss_masks: IntSeq = Field(default_factory=IntSeq)

    # Log-probabilities, length = response_len (same as loss_masks).
    # LLM-generated positions: real log-probability (≤ 0).
    # All other positions (tool outputs, re-appearing prompt/summary tokens): 0.0.
    rollout_log_probs: FloatSeq = Field(default_factory=FloatSeq)

    # Weight version for each response-space token, aligned with rollout_log_probs.
    # Non-LLM positions use -1.
    rollout_weight_versions: IntSeq = Field(default_factory=IntSeq)

    rollout_routed_experts: Any = None

    # --- Rewards ---
    # Final ORM scalar reward returned by the Agent.
    reward: float = 0.0

    # Answer correctness. Defaults to final_reward > 0; a reward function can override it
    # by returning Reward(is_correct=...) when shaping makes the sign misleading.
    is_correct: bool = False

    # Shape = len(rollout_log_probs) = response_len.
    # All zeros except the very last position = final_reward.
    token_rewards: FloatSeq = Field(default_factory=FloatSeq)

    # --- Auxiliary ---
    # The chat messages of each segment, as {segment_id: messages}. Subagent placeholder segments have
    # no entry, so look up by segment_id, not by position in `segments`.
    # chat_completions[active_segment_id] is the current context, used to detect prefix extension on resume.
    chat_completions: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)

    # True if this trajectory should be excluded from the training batch.
    masked_out: bool = False

    num_turns: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.PENDING

    # --- Segment / Triplet index (no token duplication) ---
    # Access a turn via segments[seg_i].triplets[turn_j].
    segments: list[Segment] = Field(default_factory=list)

    # segment_id of the segment that new turns are appended to.
    # In the linear case it always equals segments[-1].segment_id. While a subagent branch is open it
    # stays on the parent mainline segment: the subagent placeholder is appended to `segments` but never
    # becomes active. Lets the parser find "which segment does this turn's triplet go to" in O(1)
    # instead of walking the tree.
    active_segment_id: int = 0

    # --- Multimodal training inputs ---
    # One dict per image in prompt order, written by the Router; None for text-only trajectories.
    #   token_start / token_end : half-open range of the image's expanded pads in `tokens`
    #   image_hash              : sha256 of the image bytes; keys the per-shard image table
    #   pad_id                  : image pad token id
    #   tensors                 : processor outputs (`pixel_values`, `image_grid_thw`)
    # Excluded from `model_dump` so the tensors never reach reward functions or trajectory dumps.
    multimodal_train_inputs: list[dict[str, Any]] | None = Field(default=None, exclude=True)


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
