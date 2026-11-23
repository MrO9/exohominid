"""Settings for the push-box experiments. Edit, save, then reset the simulation in Webots.
The controller places the box and the goal line at every reset from the numbers below, so moving them by hand
in the Webots scene tree has no lasting effect: change them here."""

# What to do when the simulation starts:
#   "check"     : walk a few episodes with the box far away (does the calibration load, does it still walk?)
#   "arms_scan" : try several arm postures against the box, report which pushes best, save it (results/best_arms.json)
#   "baseline"  : walk into the box with the saved gait and the chosen arm posture (no learning), EVAL_EPISODES episodes
#   "res_check" : build the reservoir from the settings below, print its size and speed, run it with zero readout
#                 (must behave like the baseline) and with a random readout (must visibly change the walk)
#   "train"     : train the readout of the reservoir with an evolution strategy (saves results/readout_<tag>.json)
#   "queue"     : train + evaluate every run listed in QUEUE_* below, one after the other (restartable)
#   "calibrate" : open-loop baseline for every (box mass, offset range) in CALIBRATE_GRID; pick a setting with ~30-60 % success
#   "eval"      : run the trained readout for EVAL_EPISODES episodes on the same fixed box positions as the baseline
MODE = "check"

# Where the robot stands at every reset (x, y in the world). The box and the goal line are placed relative to this point,
# so it no longer matters where the robot was when you last saved the world in Webots.
ROBOT_HOME = (0.0, 0.0)

# ---- box (the world file has a 0.6 m cube; keep BOX_SIZE equal to it; mass is set in the world file) ----
BOX_SIZE = 0.60           # edge length of the cube (m)
BOX_FACE_X = 0.5          # distance from the robot's root to the box's FRONT face at the start (m). Smaller = closer.
GOAL_DX = 0.40            # delivered once the box has moved this far; the green line is drawn there
FAR_BOX_FACE_X = 2.0      # "check" mode: the box starts this far away so it is never touched

# ---- arms while pushing ----
#   None   : keep the arm pose of the walking project (arms bent at the sides, swinging)
#   "best" : use results/best_arms.json written by the "arms_scan" mode (falls back to None if it does not exist yet)
#   dict   : {"shoulder": (R, L), "upper": (R, L), "elbow": (R, L) or "mid"} in rad, held fixed (no arm swing).
#            "mid" puts each elbow at the middle of its joint range.
PUSH_ARMS = {"shoulder": (1.2, -1.2), "upper": (-0.67, 0.76), "elbow": "mid"}   # best of the arm scan; "mid" = middle of each elbow's range
LEAN_EXTRA = 0.0          # extra forward lean (rad) added to the saved calibration; raise it a little if raised arms make it tip back

EPISODES = 5
EPISODE_STEPS = 400       # control steps per episode (32 ms each, about 12.8 s)

# ---- task rules ----
STOP_AT_GOAL = True       # the episode ends as soon as the box has moved GOAL_DX (the robot stops and stands afterwards)
BOX_Y_RANGE = 0.15   # the box starts up to +-this many metres to the side (seeded, same for every controller)
BOX_MASS = 0.7   # kg; None = the mass in the world file (0.5). Set a number to make the box heavier at every reset.

# ---- task difficulty (v3) ----
PRIVILEGED = False   # False: the four supervisor inputs (box gap, box side offset, heading, lateral offset) are set to 0;
                          # the policy then only has gyro, accelerometer and gait phase
SENSE_NOISE = 0.05   # standard deviation of Gaussian noise added to every (scaled) input at every step; 0 = none
NOISE_SEED = 12345        # noise is reproducible: same seed for every run
CALIBRATE_GRID = [(0.6, 0.10), (0.7, 0.10), (0.8, 0.10), (0.9, 0.10), (0.7, 0.15), (0.8, 0.15)]   # mode "calibrate": (box mass kg, offset range m)
EVAL_EPISODES = 30        # episodes per evaluation; episode i always uses the same box offset (seed i)

# ---- reservoir (modes res_check / train / eval) ----
SUBGRAPH_DIR = "../../subgraphs"   # folder with nodes_<name>.tsv / edges_<name>.tsv (from make_subgraph.py)
RES_SUBGRAPH = "A_300_s0"          # <tier>_<neurons>_s<replicate>: tiers A, B, C; sizes 100, 300, 1000
RES_GRAPH = "h01"                  # "h01" the real wiring | "shuffle" degree-preserving rewire | "er" matched random graph
                                   # | "esn" standard echo-state network (random signs per edge)
RES_SIGN = "data"                  # "data" | "random17" | "allpos" | "flipped"
RES_WEIGHT = "sqrt"                # edge strength from synapse rows: "sqrt" | "binary" | "count"
RES_SEED = 0                       # seeds the input wiring, the sign draw and the control graphs
RES_RHO = 0.9                      # spectral radius of the recurrent matrix
RES_LEAK = 0.3                     # leak rate (1 = no memory in the state update)
RES_IN_SCALE = 0.5                 # strength of the input weights
RES_TURN_MAX = 0.6                 # largest steering command (walker turn units; 1.0 = hip amplitude +-50 % on each leg)
RES_LEAN_MAX = 0.08                # largest extra ankle lean (rad)
RES_STRIDE_MAX = 0.25              # stride multiplier ranges over 1 +- this

# ---- evolution strategy (mode "train") ----
ES_POP = 16                        # candidates per generation (even)
ES_GENS = 60   # generations
ES_SIGMA = 0.05                    # size of the random perturbations of the readout weights
ES_LR = 0.03   # step size
ES_EPISODES = 3   # episodes per candidate (every candidate sees the same box offsets in a generation)
ES_RESUME = True                   # continue from results/readout_<tag>.json if it exists
ES_VAL_EPISODES = 10               # fixed validation offsets used to pick the checkpoint (open loop is the starting candidate)
ES_VAL_EVERY = 3                   # validate the centre every this many generations
RUN_TAG = "_v3"   # appended to every run name so old results are neither resumed nor mixed in
EVAL_WHICH = "best"   # mode "eval": use the readout of the "last" generation or the "best" centre seen in training

# ---- queue (mode "queue") ----
QUEUE_SEEDS = (0, 1, 2)   # seed k: reservoir randomness k, ES seed k, and (below) subgraph replicate k
QUEUE_SUBGRAPH_BY_SEED = True                 # True: seed k uses subgraph QUEUE_SUBGRAPH.format(seed=k), a different H01 region;
                                              # controls of seed k are built from that same subgraph. False: always RES_SUBGRAPH
QUEUE_SUBGRAPH = "A_300_s{seed}"
QUEUE_CONDITIONS = [("h01", "data"),          # (RES_GRAPH, RES_SIGN); trim this list to shorten the queue
                    ("h01", "permute"),
                    ("shuffle", "data"),
                    ("er", "data"),
                    ("esn", "data"),
                    ("shuffle", "permute"),   # v3b control: scrambled wiring + permuted signs
                    ("direct", "data"),       # v3b control: linear readout straight from the 12 inputs, no reservoir
                    ("bias", "data")]         # v3b control: one fixed correction, no reservoir, no inputs
