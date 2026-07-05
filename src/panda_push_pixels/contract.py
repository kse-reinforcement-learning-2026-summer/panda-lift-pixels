"""The frozen contract for Project 2.

Every constant here is part of the graded contract. The package is installed from a pinned
git tag, so these values are identical on the student's Colab, in their CI, and in the
instructor's final grader. Students may *read* these (e.g. to know the threshold) but cannot
change what the grader uses.

Hidden at grading time: only the evaluation seeds (``EVAL_SEED_OFFSET`` env var, injected as a
GitHub Secret). Everything else here is public on purpose.
"""

# ---------------------------------------------------------------------------
# Observation contract — what the model receives as input
# ---------------------------------------------------------------------------
N_STACK = 4                                   # DQN-style frame stack
FRAME_HW = 112                                # rendered frame is FRAME_HW x FRAME_HW
CHANNELS_PER_FRAME = 3                        # RGB (the cube is green, the target is red — color matters)
OBS_SHAPE = (N_STACK * CHANNELS_PER_FRAME, FRAME_HW, FRAME_HW)  # (12, 112, 112), channels-first
OBS_LOW = 0.0
OBS_HIGH = 1.0                                # observations are float32 in [0, 1] (already /255)

# ---------------------------------------------------------------------------
# Action contract — what the model must output
# ---------------------------------------------------------------------------
ACTION_DIM = 7                               # 7 joint position deltas (joints control; gripper stays closed)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0                            # the grader clips outputs into [ACTION_LOW, ACTION_HIGH]

# ---------------------------------------------------------------------------
# Task definition — the behavior we grade (Push: push the cube onto the target)
# ---------------------------------------------------------------------------
BASE_ENV_ID = "PandaPushJoints-v3"
OBJECT_SIZE = 0.04           # cube (and target marker) side length (m)
DISTANCE_THRESHOLD = 0.05    # success: object-to-target center distance (m) below this
MAX_EPISODE_STEPS = 50       # grading horizon (one physics step per decision, no frame skip)

# Canonical reward is sparse: 0.0 on a step where the cube is within DISTANCE_THRESHOLD of the
# target, else -1.0. The episode TERMINATES the instant the cube reaches the target (matching
# vanilla PandaPush) — so the return is -(steps taken to succeed), or -MAX_EPISODE_STEPS if the
# episode times out without success.

# ---------------------------------------------------------------------------
# Grading thresholds
# ---------------------------------------------------------------------------
PARAM_LIMIT = 10_000_000     # max number of parameters in the submitted model.pt
REWARD_THRESHOLD = -40.0     # PLACEHOLDER median cumulative reward to pass — set after calibration
LATENCY_BUDGET_S = 0.05      # max seconds per forward pass on CPU (keeps eval within time budget)

EVAL_EPISODES_CI = 30        # episodes run in student CI / local testing
EVAL_EPISODES_FINAL = 100    # episodes run in the instructor's final grading
