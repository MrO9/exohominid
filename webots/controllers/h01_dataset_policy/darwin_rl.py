"""
Darwin-OP / ROBOTIS OP2 -- self-calibrating walking controller for Webots
==========================================================================
Pure Python (standard library only). Put this file in
    <project>/controllers/darwin_rl/darwin_rl.py
and set the robot node to:  supervisor TRUE,  controller "darwin_rl".

What it does, in order (each stage is cached in darwin_state.json, so a restart
skips finished stages; delete that file to redo everything):

  1. CALIBRATE  read joint limits, work out which way each leg joint bends, and
                find a standing pose that really stands (tries all hip/ankle
                sign combinations and picks the one that stays upright).
  2. GAIT SEARCH  open-loop walking pattern (CPG) tuned by random search.
                No learning yet, just "which sine-wave parameters walk furthest".
  3. TRAIN      TD3 learns small residual corrections on top of the best gait.

Set STAGES below to run only part of it. Everything is printed so you can see
what each stage decided. Use Fast mode with rendering off for speed.
"""
from controller import Supervisor
import math
import random
import json
import os

# ============================================================
# CONFIG
# ============================================================
PROFILE = "auto"           # which robot: "auto" (detect from the device names), "darwin", "atlas", or use CUSTOM_PROFILE
CUSTOM_PROFILE = None      # a dict in the same format as the profiles below, for any other Webots humanoid
KEYBOARD_MODE = False      # True = after the gait is ready, drive the robot with the keyboard instead of training
KEYBOARD_WAIT_FOR_HUMANIZE = False   # False = keyboard mode starts right away with the best gait saved so far
TURN_GAIN = 0.5            # keyboard turning: how much longer one leg steps than the other (0..1)
ROBUST_SEARCH = True       # judge every gait under small random command noise, so only gaits that survive a nudge are kept
ROBUST_NOISE = 0.006       # rad of uniform noise added to every leg command during robust evaluation of NEW searches
ROBUST_LEVELS = (0.003, 0.006, 0.010, 0.015)   # the polish raises the noise step by step, only while the gait keeps up
ROBUST_EVALS = 30          # candidates tried at each noise level of the polish
FORCE_GAIT = {"ankle_follow": 0.9556, "arm_amp": 0.2728, "arm_mirror": -1, "foot_ratio": 0.4815,
              "hip_amp": 0.1855, "knee_amp": 0.2966, "lean": 0.0352, "period": 0.6398, "roll_amp": 0.2104,
              "roll_mirror": 1, "roll_phase": 0.1821, "yaw_amp": 0.0143}
              # ^ restores your earlier forward gait ONCE (the v17 polish had overwritten it); set to None to disable
ROBUST_VERSION = 2
FIT_VERSION = 2            # bump when the fitness function changes in a way that makes old non-Darwin searches obsolete         # bump to polish saved gaits again
OMNI = True                # also search backward, left and right walking gaits (stored next to the forward one)
OMNI_EVALS = 100           # candidates tried per extra direction
TRAIN_DIRECTION = "fwd"    # which gait the TD3 residual is trained on: "fwd", "back", "left" or "right"
OMNI_STEP = {"back": 0.08, "left": 0.05, "right": 0.05}   # m, step length that earns the full stride bonus
RUN_TRAINING = True        # False = stop after the gait search and just replay it
HUMANIZE = True            # after the search, add human-like sway, arm swing and pelvis rotation
HUMAN_EVALS = 150          # candidates tried by the humanize stage
HUMAN_VERSION = 7          # bump to make the humanize stage run again on the next start
SWAY_TARGET = 0.025        # m, sideways body sway (half-range) that counts as fully human-like
ELBOW_L = 1.62            # ArmLowerL while walking (rad); held from the start of every reset
ELBOW_R = -1.65            # ArmLowerR while walking (rad)
ARMUP_L = 0.60             # ArmUpperL while walking (rad); held from the start of every reset
ARMUP_R = -0.60            # ArmUpperR while walking (rad)
STEP_TARGET = 0.13         # m, length of one step (one foot ahead of the other) that counts as a long stride
LIFT_TARGET = 0.60         # rad, extra swing-phase knee bend that counts as a clearly lifted foot
LIFT_W = 1.2               # weight of the foot-lift bonus (only if the robot stays up)
STRIDE_W = 1.4             # weight of the long-stride bonus (only if the robot stays up)
HUMAN_W = 0.5              # weight of the human-likeness bonus in the fitness (only if it stays up)
TRAIN_EVEN_IF_GAIT_FAILS = False   # True = train the RL residual even if the searched gait does not walk
REDO = ()                  # force stages to rerun, e.g. ("search",) or ("calibrate", "search")
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = "darwin_rl_model.json"
STATE_PATH = "darwin_state.json"     # calibration + best gait, delete to start over
STATE_VERSION = 4
LOG_PATH = "training_log_v2.csv"
SEED = 1
MOTOR_SPEED = 8.0          # max joint speed commanded (rad/s)
CROUCH = 0.30              # knee flexion of the standing pose (rad); small = easy to hold
RAMP_S = 0.6               # seconds to fade the walking motion in after each reset
UPRIGHT_ROT = [0.0, 0.0, 1.0, 0.0]   # robot is always reset upright, facing +x
FELL_TILT = 0.70           # torso tilt (rad) counted as a fall
FELL_DROP = 0.07           # root height drop (m) counted as a fall

CONTROL_MS = 32            # control period; rounded to a multiple of basicTimeStep
SPAWN_Z = 0.32             # drop height of the root node (m)
HOLD_STEPS = 14            # steps the body is held in the air while joints reach pose
PROBE_STEP = 0.01          # m, how far the held body is lowered between floor probes
SETTLE_STEPS = 20          # steps after release before an episode starts

# ---- calibration ----
CAL_CROUCH = 0.40          # knee flexion tried first (rad)
CAL_TRIAL_STEPS = 40       # ~1.3 s per sign combination

# ---- gait search ----
SEARCH_EVALS = 150
MARCH_KNEE = 0.30          # knee lift (rad) that counts as a proper marching step
MARCH_STEPS = 150          # episode length while learning to march in place
WALK_STEP_FRAC = 0.20      # hard robots count as walking from this fraction of STEP_TARGET per step (about 7 cm for the Atlas)
CREEP_EVALS = 30           # balance re-tuning trials at each stride size
MARCH_EVALS = 70           # hill-climb candidates in the marching stage
STAB_EVALS = 30            # candidates per gait in the feedback-tuning stage (hard robots only)
SEARCH_EP_STEPS = 240      # ~7.7 s per candidate (longer = rejects gaits that fall late)
SEARCH_SIGMA = 0.25        # mutation size, as a fraction of each parameter range

# ---- RL ----
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
BATCH_SIZE = 16
BUFFER_CAP = 50_000
WARMUP_STEPS = 1500
UPDATE_EVERY = 4
POLICY_DELAY = 2
POLICY_NOISE = 0.10
NOISE_CLIP = 0.30
GRAD_CLIP = 5.0
EXPLORE_START, EXPLORE_END, EXPLORE_STEPS = 0.30, 0.05, 30_000
MAX_ACTION = 0.06          # residual size on top of the gait (rad); small = safe
ACTOR_START_UPDATES = 1500 # critic trains alone for this many updates before the policy moves
HIDDEN = 64
ACTION_DIM = 12
# ---- ideas from DeepWalk (Rodriguez & Behnke) and URMA (Bohlinger et al.), switchable for A/B tests ----
SHARED_POLICY = True       # ONE residual network for all gaits, told which direction it is walking (one-hot input)
DIR_CURRICULUM_STEPS = 6000   # DeepWalk-style velocity curriculum: forward only first, then all directions
PENALTY_RAMP_STEPS = 6000  # URMA-style penalty curriculum: penalty terms fade in from 0 to full over this many steps
LAYER_NORM = True          # URMA / FlashSAC: normalise the hidden layer of every network
EXPERIMENT = ""            # label for A/B runs: model + log files get this suffix, e.g. "baseline"
DIR_NAMES = ("fwd", "back", "left", "right")
STATE_DIM = 12 + 12 + 3 + 3 + 2 + 12 + (4 if SHARED_POLICY else 0)    # 44, +4 direction one-hot
MAX_EP_STEPS = 300

# ---- ideas borrowed from FlashSAC / FastTD3 that make sense for one CPU simulation ----
NOISE_REPEAT = True        # FlashSAC "noise repetition": one exploration noise vector is held for k steps
REPEAT_S = 1.5             # k is drawn with P(k) ~ k^-REPEAT_S (mostly short, sometimes long)
REPEAT_MAX = 8
REWARD_NORM = True         # FlashSAC reward scaling: divide rewards by the running std of the discounted return

# ============================================================
# NAMES
# ============================================================
ALL_MOTORS = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL",
    "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR",
    "AnkleL", "FootR", "FootL", "Neck", "Head",
]
# leg joint order used everywhere below: index = 2*pair + side  (0 = R, 1 = L)
LEG = ["PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL",
       "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR", "FootL"]
YAW, ROLL, HIP, KNEE, ANKLE, FOOT = 0, 1, 2, 3, 4, 5   # pair indices


def J(pair, side):
    return 2 * pair + side


# ============================================================
# SMALL HELPERS
# ============================================================
def zeros(n): return [0.0] * n
def zeros2d(r, c): return [[0.0] * c for _ in range(r)]
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def mv(W, x): return [dot(row, x) for row in W]
def addv(a, b): return [x + y for x, y in zip(a, b)]
def relu(x): return [v if v > 0.0 else 0.0 for v in x]
def tanh(x): return [math.tanh(v) for v in x]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_list(v, lo, hi):
    return [clamp(x, lo, hi) for x in v]


def copy_into(dst, src):
    if isinstance(dst[0], list):
        for i in range(len(dst)):
            for j in range(len(dst[i])):
                dst[i][j] = src[i][j]
    else:
        for i in range(len(dst)):
            dst[i] = src[i]


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================
# MLP + ADAM + REPLAY + TD3
# ============================================================
class MLP:
    def __init__(self, in_dim, hid, out_dim, out_act="tanh", out_scale=1.0):
        self.W1 = [[random.gauss(0, math.sqrt(2.0 / in_dim)) for _ in range(in_dim)]
                   for _ in range(hid)]
        self.b1 = zeros(hid)
        self.W2 = [[random.gauss(0, math.sqrt(2.0 / hid) * out_scale) for _ in range(hid)]
                   for _ in range(out_dim)]
        self.b2 = zeros(out_dim)
        self.out_act = out_act

    def forward(self, x):
        z1 = addv(mv(self.W1, x), self.b1)
        inv = 1.0
        if LAYER_NORM:                                   # zero mean, unit variance across the hidden units
            n = len(z1)
            mu = sum(z1) / n
            inv = 1.0 / math.sqrt(sum((v - mu) ** 2 for v in z1) / n + 1e-5)
            z1 = [(v - mu) * inv for v in z1]
        h1 = relu(z1)
        z2 = addv(mv(self.W2, h1), self.b2)
        y = tanh(z2) if self.out_act == "tanh" else z2[:]
        return y, (x, z1, h1, z2, inv)

    def backward(self, dy, cache, g):
        x, z1, h1, z2, inv = cache
        if self.out_act == "tanh":
            dz2 = [dy[i] * (1.0 - math.tanh(z2[i]) ** 2) for i in range(len(z2))]
        else:
            dz2 = list(dy)
        dh1 = zeros(len(h1))
        for i, d in enumerate(dz2):
            if d == 0.0:
                continue
            gw, w = g["W2"][i], self.W2[i]
            for j in range(len(h1)):
                gw[j] += d * h1[j]
                dh1[j] += w[j] * d
            g["b2"][i] += d
        dx = zeros(len(x))
        dn = [dh1[i] if z1[i] > 0.0 else 0.0 for i in range(len(z1))]
        if LAYER_NORM:                                   # backward through the normalisation
            n = len(dn)
            m1 = sum(dn) / n
            m2 = sum(dn[i] * z1[i] for i in range(n)) / n
            dn = [inv * (dn[i] - m1 - z1[i] * m2) for i in range(n)]
        for i in range(len(z1)):
            d = dn[i]
            if d == 0.0:
                continue
            gw, w = g["W1"][i], self.W1[i]
            for j in range(len(x)):
                gw[j] += d * x[j]
                dx[j] += w[j] * d
            g["b1"][i] += d
        return dx

    def zero_grads(self):
        return {"W1": zeros2d(len(self.W1), len(self.W1[0])), "b1": zeros(len(self.b1)),
                "W2": zeros2d(len(self.W2), len(self.W2[0])), "b2": zeros(len(self.b2))}

    def as_dict(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def copy_from(self, o):
        for k in ("W1", "b1", "W2", "b2"):
            copy_into(getattr(self, k), getattr(o, k))


class Adam:
    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = {k: (zeros2d(len(P), len(P[0])) if isinstance(P[0], list) else zeros(len(P)))
                  for k, P in params.items()}
        self.v = {k: (zeros2d(len(P), len(P[0])) if isinstance(P[0], list) else zeros(len(P)))
                  for k, P in params.items()}

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, P in self.p.items():
            G, M, V = grads[k], self.m[k], self.v[k]
            if isinstance(P[0], list):
                for i in range(len(P)):
                    for j in range(len(P[i])):
                        g = G[i][j]
                        M[i][j] = self.b1 * M[i][j] + (1 - self.b1) * g
                        V[i][j] = self.b2 * V[i][j] + (1 - self.b2) * g * g
                        P[i][j] -= self.lr * (M[i][j] / c1) / (math.sqrt(V[i][j] / c2) + self.eps)
            else:
                for i in range(len(P)):
                    g = G[i]
                    M[i] = self.b1 * M[i] + (1 - self.b1) * g
                    V[i] = self.b2 * V[i] + (1 - self.b2) * g * g
                    P[i] -= self.lr * (M[i] / c1) / (math.sqrt(V[i] / c2) + self.eps)


class ReplayBuffer:
    def __init__(self, cap):
        self.data, self.cap, self.ptr = [], cap, 0

    def push(self, *item):
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            self.data[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, n):
        return random.sample(self.data, n)

    def __len__(self):
        return len(self.data)


class RewardNorm:
    """Scales rewards by the running standard deviation of the discounted return, so the critic
    sees values of a stable size whatever the reward function is (FlashSAC, eq. 6, simplified)."""

    def __init__(self):
        self.n, self.mean, self.m2, self.G = 0, 0.0, 0.0, 0.0

    def reset_episode(self):
        self.G = 0.0

    def __call__(self, r):
        self.G = GAMMA * self.G + r
        self.n += 1
        d = self.G - self.mean
        self.mean += d / self.n
        self.m2 += d * (self.G - self.mean)
        std = math.sqrt(self.m2 / self.n) if self.n > 100 else 1.0
        return r / max(std, 1.0)

    def as_list(self):
        return [self.n, self.mean, self.m2]

    def set(self, v):
        self.n, self.mean, self.m2 = int(v[0]), float(v[1]), float(v[2])


class TD3Agent:
    """Actor outputs a residual in [-MAX_ACTION, MAX_ACTION] for the 12 leg joints."""

    def __init__(self):
        n_in = STATE_DIM + ACTION_DIM
        self.actor = MLP(STATE_DIM, HIDDEN, ACTION_DIM, "tanh", 0.1)   # starts near zero
        self.actor_t = MLP(STATE_DIM, HIDDEN, ACTION_DIM, "tanh", 0.1)
        self.c1, self.c2 = MLP(n_in, HIDDEN, 1, "lin", 0.1), MLP(n_in, HIDDEN, 1, "lin", 0.1)
        self.c1_t, self.c2_t = MLP(n_in, HIDDEN, 1, "lin", 0.1), MLP(n_in, HIDDEN, 1, "lin", 0.1)
        for t, s in ((self.actor_t, self.actor), (self.c1_t, self.c1), (self.c2_t, self.c2)):
            t.copy_from(s)
        self.oa, self.o1, self.o2 = (Adam(self.actor.as_dict(), lr=1e-4), Adam(self.c1.as_dict()),
                                     Adam(self.c2.as_dict()))
        self.scratch = self.c1.zero_grads()      # reused for dQ/da, avoids reallocating
        self.updates = 0
        self.rnorm = RewardNorm()

    def act(self, s, noise=0.0, eps=None):
        """eps: optional held unit-Gaussian noise vector (noise repetition); otherwise fresh noise."""
        a = [v * MAX_ACTION for v in self.actor.forward(s)[0]]
        if noise > 0:
            a = [clamp(v + noise * MAX_ACTION * (eps[i] if eps is not None else random.gauss(0, 1)),
                       -MAX_ACTION, MAX_ACTION) for i, v in enumerate(a)]
        return a

    @staticmethod
    def _scale(g):
        for k in g:
            if isinstance(g[k][0], list):
                for row in g[k]:
                    for j in range(len(row)):
                        row[j] = clamp(row[j] / BATCH_SIZE, -GRAD_CLIP, GRAD_CLIP)
            else:
                for j in range(len(g[k])):
                    g[k][j] = clamp(g[k][j] / BATCH_SIZE, -GRAD_CLIP, GRAD_CLIP)

    @staticmethod
    def _soft(net, tgt):
        a, b = net.as_dict(), tgt.as_dict()
        for k in a:
            P, T = a[k], b[k]
            if isinstance(P[0], list):
                for i in range(len(P)):
                    for j in range(len(P[i])):
                        T[i][j] += TAU * (P[i][j] - T[i][j])
            else:
                for i in range(len(P)):
                    T[i] += TAU * (P[i] - T[i])

    def update(self, buf):
        batch = buf.sample(BATCH_SIZE)
        g1, g2 = self.c1.zero_grads(), self.c2.zero_grads()
        for s, a, r, s2, term in batch:                   # term = 1 only for a real fall
            a2 = [clamp(v * MAX_ACTION + clamp(random.gauss(0, POLICY_NOISE), -NOISE_CLIP, NOISE_CLIP),
                        -MAX_ACTION, MAX_ACTION) for v in self.actor_t.forward(s2)[0]]
            q_t = min(self.c1_t.forward(s2 + a2)[0][0], self.c2_t.forward(s2 + a2)[0][0])
            y = r + GAMMA * (1.0 - term) * q_t
            q1, k1 = self.c1.forward(s + a)
            q2, k2 = self.c2.forward(s + a)
            self.c1.backward([q1[0] - y], k1, g1)
            self.c2.backward([q2[0] - y], k2, g2)
        self._scale(g1); self._scale(g2)
        self.o1.step(g1); self.o2.step(g2)
        self.updates += 1
        if self.updates % POLICY_DELAY == 0:
            if self.updates >= ACTOR_START_UPDATES:      # let the critic learn what falling costs first
                ga = self.actor.zero_grads()
                for s_, a_, r_, s2_, term_ in batch:
                    ap, ka = self.actor.forward(s_)
                    sa = [v * MAX_ACTION for v in ap]
                    _, kc = self.c1.forward(s_ + sa)
                    for k in self.scratch:                # zero the scratch grads
                        if isinstance(self.scratch[k][0], list):
                            for row in self.scratch[k]:
                                for j in range(len(row)): row[j] = 0.0
                        else:
                            for j in range(len(self.scratch[k])): self.scratch[k][j] = 0.0
                    dx = self.c1.backward([1.0], kc, self.scratch)
                    self.actor.backward([-dx[STATE_DIM + i] * MAX_ACTION for i in range(ACTION_DIM)], ka, ga)
                self._scale(ga)
                self.oa.step(ga)
                self._soft(self.actor, self.actor_t)
            self._soft(self.c1, self.c1_t)
            self._soft(self.c2, self.c2_t)

    def save(self, path, tag=""):
        with open(path, "w") as f:
            json.dump({"tag": tag, "actor": self.actor.as_dict(), "c1": self.c1.as_dict(),
                       "c2": self.c2.as_dict(), "updates": self.updates,
                       "rnorm": self.rnorm.as_list() if REWARD_NORM else None,
                       "ln": LAYER_NORM}, f)

    def load(self, path, tag=""):
        """Returns True if a compatible model was loaded."""
        if not os.path.exists(path):
            print(f"[load] {path} not found, starting fresh")
            return False
        with open(path) as f:
            d = json.load(f)
        if d.get("tag") != tag or len(d["actor"]["W1"][0]) != STATE_DIM:
            print("[load] saved model belongs to a different gait / network size, starting fresh")
            return False
        if d.get("ln", False) != LAYER_NORM:
            print("[load] saved model used a different network layout, starting fresh")
            return False
        if REWARD_NORM != (d.get("rnorm") is not None):
            print("[load] saved model used a different reward scaling, starting fresh")
            return False
        try:
            for name, net, tgt in (("actor", self.actor, self.actor_t),
                                   ("c1", self.c1, self.c1_t), ("c2", self.c2, self.c2_t)):
                for k in ("W1", "b1", "W2", "b2"):
                    copy_into(getattr(net, k), d[name][k])
                tgt.copy_from(net)
            self.updates = d.get("updates", 0)
            if REWARD_NORM:
                self.rnorm.set(d["rnorm"])
            print(f"[load] {path} (updates={self.updates})")
            return True
        except Exception as e:
            print(f"[load] {path} is incompatible ({e}); starting fresh")
            return False


# ============================================================
# GAIT (central pattern generator) PARAMETERS
# name: (low, high, default).  Angles are PHYSICAL angles: hip/ankle positive =
# flexion, mapped to each joint's real sign by the calibration stage.
# ============================================================
PARAMS = {
    "period":       (0.50, 1.40, 0.80),   # seconds per full gait cycle
    "hip_amp":      (0.00, 0.60, 0.12),   # hip swing amplitude
    "knee_amp":     (0.00, 0.90, 0.25),   # extra knee flexion during swing
    "ankle_follow": (0.00, 1.00, 0.80),   # 1 = ankle keeps foot flat
    "lean":         (-0.12, 0.12, 0.00),  # constant forward/back lean at the ankles
    "roll_amp":     (-0.40, 0.40, 0.08),  # side-to-side pelvis sway (signed)
    "roll_phase":   (0.00, 1.00, 0.00),   # timing of the sway within the cycle
    "foot_ratio":   (-2.00, 2.00, 0.50),  # ankle-roll compensation of the sway
    "yaw_amp":      (-0.15, 0.15, 0.00),  # pelvis rotation about the vertical axis (signed)
    "arm_amp":      (-0.80, 0.80, 0.00),  # arm swing, opposite to the same-side leg (signed)
    "side_amp":     (-0.50, 0.50, 0.00),  # sideways leg shift for side-stepping (signed, 0 = off)
    "side_phase":   (0.00, 1.00, 0.00),   # timing of that shift within the cycle
    "side_foot":    (-2.00, 2.00, 0.50),  # ankle-roll compensation of that shift
    "fb_kp":        (0.00, 1.50, 0.00),   # ankle pitch feedback on body lean (0 = off; only searched if the robot profile asks)
    "fb_kd":        (0.00, 0.30, 0.00),   # ... on lean rate
    "fb_roll":      (-1.00, 1.00, 0.00),  # ankle roll feedback on sideways lean (signed)
    "fb_hip":       (-0.80, 0.80, 0.00),  # hip pitch feedback on forward lean (signed: the search finds the direction)
}
FB_KEYS = ("fb_kp", "fb_kd", "fb_roll", "fb_hip")
EXTRA_KEYS = ("side_amp", "side_phase", "side_foot", "phase_dir") + FB_KEYS
# walking direction in the robot's own frame: (forward, left)
DIRS = {"fwd": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
SIDE_KEYS = ("side_amp", "side_phase", "side_foot")
FLIPS = {"roll_mirror": 1, "arm_mirror": -1}   # +1 / -1: left joint mirrored vs the right (searched)
FLIP_PARAM = "roll_mirror"


def default_gait():
    g = {k: v[2] for k, v in PARAMS.items()}
    g.update(FLIPS)
    g["phase_dir"] = 1            # -1 = run the pattern backwards in time (walk backward); never mutated
    return g


def random_gait(direction="fwd"):
    g = {k: (random.uniform(v[0], v[1]) if (FEEDBACK_ON or k not in FB_KEYS) else 0.0) for k, v in PARAMS.items()}
    for k in FLIPS:
        g[k] = random.choice((1, -1))
    g["phase_dir"] = -1 if direction == "back" else 1
    if direction in ("fwd", "back"):
        g["side_amp"] = 0.0
    return g


def frozen_params(direction):
    """Side-stepping parameters only matter (and are only searched) for the sideways gaits."""
    base = tuple(SIDE_KEYS) if direction in ("fwd", "back") else ()
    return base if FEEDBACK_ON else base + FB_KEYS


def mutate_gait(g, sigma, frozen=()):
    n = dict(g)
    for k, (lo, hi, _) in PARAMS.items():
        if k in frozen:
            continue
        if random.random() < 0.5:
            v = n.get(k, PARAMS[k][2]) + random.gauss(0.0, sigma * (hi - lo))
            n[k] = (v % 1.0) if k == "roll_phase" else clamp(v, lo, hi)
    for k in FLIPS:
        if random.random() < 0.08:
            n[k] = -n[k]
    return n


def gait_id(g):
    plain = (g.get("side_amp", 0.0) == 0.0 and g.get("phase_dir", 1) == 1
             and all(g.get(k, 0.0) == 0.0 for k in FB_KEYS))
    return json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in sorted(g.items())
                       if not (plain and k in EXTRA_KEYS)})     # forward-only gaits keep their old id


def arms_sig():
    we, wu = PROF["walk_elbow"], PROF["walk_upper"]
    return json.dumps([we[1], we[0], wu[1], wu[0]])


# ============================================================
# ROBOT PROFILES -- everything that is specific to one robot lives here.
# The learning pipeline (calibration, gait search, RL residual, keyboard) is the same for every robot.
#   leg    : (right, left) device name for yaw / roll / hip pitch / knee / ankle pitch / ankle roll
#   swing / elbow / upper : (right, left) arm joints used for arm swing, elbow pose and upper-arm pose
#   rest   : joints held at a fixed angle (everything else listed here just stays put)
#   length / time : size and time scale relative to the DARwIn-OP (leg length ratio, sqrt of it for gait period)
# ============================================================
DEFAULT_PROFILE = dict(
    name="robot", prefix="robot", match=(), leg={}, swing=(None, None), elbow=(None, None), upper=(None, None),
    rest={}, hold_others=False, walk_elbow=(0.0, 0.0), walk_upper=(0.0, 0.0), sensor_suffix="",
    acc=("Accelerometer", "accelerometer"), gyro=("Gyro", "gyro"),
    length=1.0, time=1.0, motor_speed=8.0, min_root_z=0.23, feedback=False, seeds=(), ranges={}, gate_bonus=False, noise_scale=1.0, action_scale=1.0)


def darwin_profile():
    return dict(DEFAULT_PROFILE, name="darwin", prefix="darwin", match=("LegUpperR", "PelvYR"),
                leg=dict(yaw=("PelvYR", "PelvYL"), roll=("PelvR", "PelvL"), hip=("LegUpperR", "LegUpperL"),
                         knee=("LegLowerR", "LegLowerL"), ankle=("AnkleR", "AnkleL"), foot=("FootR", "FootL")),
                swing=("ShoulderR", "ShoulderL"), elbow=("ArmLowerR", "ArmLowerL"), upper=("ArmUpperR", "ArmUpperL"),
                rest={"ShoulderR": -0.10, "ShoulderL": -0.10, "ArmUpperR": 0.05, "ArmUpperL": 0.05,
                      "ArmLowerR": -0.05, "ArmLowerL": -0.05, "Neck": 0.0, "Head": 0.0},
                walk_elbow=(ELBOW_R, ELBOW_L), walk_upper=(ARMUP_R, ARMUP_L),
                sensor_suffix="S", acc=("Accelerometer",), gyro=("Gyro",))


def atlas_profile():
    """Boston Dynamics Atlas as shipped with Webots (joint names and limits from the Webots Atlas page).
    Arm swing = shoulder pitch (Usy); upper arm = shoulder roll (Shx, +-1.2 as in your C sample, arms down);
    elbow flexion = Elx.  Everything else listed under rest is held at 0."""
    hold = {}
    for n in ("BackLbz", "BackMby", "BackUbx", "NeckAy", "LArmEly", "RArmEly", "LArmUwy", "RArmUwy", "LArmMwx", "RArmMwx"):
        hold[n] = 0.0
    return dict(DEFAULT_PROFILE, name="atlas", prefix="atlas", match=("LLegLhy", "RLegLhy", "LLegKny"),
                leg=dict(yaw=("RLegUhz", "LLegUhz"), roll=("RLegMhx", "LLegMhx"), hip=("RLegLhy", "LLegLhy"),
                         knee=("RLegKny", "LLegKny"), ankle=("RLegUay", "LLegUay"), foot=("RLegLax", "LLegLax")),
                swing=("RArmUsy", "LArmUsy"), elbow=("RArmElx", "LArmElx"), upper=("RArmShx", "LArmShx"),
                rest=hold, hold_others=True, gate_bonus=True,
                walk_elbow=(0.0, 0.0), walk_upper=(1.39, -1.39),  # shoulder roll at its limit and straight elbows: arms hang by the hips
                length=2.6, time=1.6, motor_speed=8.0, min_root_z=0.9, feedback=True,
                # gentler search ranges than the small robot's: random gaits at Darwin amplitudes topple Atlas within a second
                # lean: the standing scan shows only about -0.02 .. +0.05 rad of ankle lean keeps it up
                ranges={"period": (0.9, 2.4, 1.3), "hip_amp": (0.0, 0.30, 0.10), "knee_amp": (0.0, 0.50, 0.15),
                        "roll_amp": (-0.30, 0.30, 0.05), "lean": (-0.02, 0.05, 0.0), "yaw_amp": (-0.10, 0.10, 0.0),
                        "arm_amp": (-0.60, 0.60, 0.0), "fb_kp": (0.0, 1.0, 0.0), "fb_kd": (0.0, 0.25, 0.0),
                        "fb_hip": (-0.6, 0.15, 0.0)},   # hip feedback must be negative to oppose a forward lean
                noise_scale=0.5, action_scale=0.4,   # the standing window is only about +-0.05 rad at the ankle
                seeds=({"period": 2.2, "hip_amp": 0.08, "knee_amp": 0.10},          # a slow shuffle like your C gait
                       {"period": 1.3, "hip_amp": 0.20, "knee_amp": 0.35, "fb_kp": 0.30, "fb_kd": 0.10},
                       {"period": 1.0, "hip_amp": 0.30, "knee_amp": 0.45, "fb_kp": 0.50, "fb_kd": 0.12}))


PROF = dict(DEFAULT_PROFILE)
FEEDBACK_ON = False
_APPLIED = [None]
KNOWN_PROFILES = {"darwin": darwin_profile, "atlas": atlas_profile}


def apply_profile(prof):
    """Make the module-level settings match the robot (sizes, time scale, file names). Applied once per run."""
    global PROF, LEG, FEEDBACK_ON, STEP_TARGET, SWAY_TARGET, OMNI_STEP, FELL_DROP, PROBE_STEP
    global MOTOR_SPEED, MODEL_PATH, STATE_PATH, LOG_PATH, ROBUST_NOISE, ROBUST_LEVELS, MAX_ACTION
    PROF = prof
    LEG = [prof["leg"][r][side] for r in ("yaw", "roll", "hip", "knee", "ankle", "foot") for side in (0, 1)]
    FEEDBACK_ON = bool(prof["feedback"])
    if _APPLIED[0] == prof["name"]:
        return
    _APPLIED[0] = prof["name"]
    L, T = prof["length"], prof["time"]
    if L != 1.0:
        STEP_TARGET *= L
        SWAY_TARGET *= L
        FELL_DROP *= L
        PROBE_STEP *= L
        OMNI_STEP = {k: v * L for k, v in OMNI_STEP.items()}
    if T != 1.0 and "period" not in prof["ranges"]:
        PARAMS["period"] = tuple(v * T for v in PARAMS["period"])
    for k, v in prof["ranges"].items():
        PARAMS[k] = tuple(v)
    MOTOR_SPEED = prof["motor_speed"]
    ROBUST_NOISE *= prof["noise_scale"]                # a heavy robot starts with gentler command noise
    MAX_ACTION *= prof["action_scale"]                 # ... and a smaller learned residual
    ROBUST_LEVELS = tuple(v * prof["noise_scale"] for v in ROBUST_LEVELS)
    if prof["prefix"] != "darwin":                     # the Darwin-OP keeps its original file names
        MODEL_PATH = prof["prefix"] + "_rl_model.json"
        STATE_PATH = prof["prefix"] + "_state.json"
        LOG_PATH = prof["prefix"] + "_training_log.csv"


def device_names(robot):
    """Names of every device on the robot (so we never probe for devices that are not there), or None if unavailable."""
    try:
        return {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}
    except Exception:
        return None


def enable_feedback():
    """Switch the balance-feedback terms on (used when the ordinary search fails and the robot is treated as hard)."""
    global FEEDBACK_ON
    FEEDBACK_ON = True


def detect_profile(robot):
    if CUSTOM_PROFILE is not None:
        return dict(DEFAULT_PROFILE, **CUSTOM_PROFILE)
    names = [PROFILE] if PROFILE != "auto" else list(KNOWN_PROFILES)
    for nm in names:
        if nm not in KNOWN_PROFILES:
            print("[!] unknown PROFILE '%s' (known: %s)" % (nm, ", ".join(KNOWN_PROFILES)))
            raise SystemExit(1)
        prof = KNOWN_PROFILES[nm]()
        names = device_names(robot)
        if names is not None:
            ok = all(n in names for n in prof["match"])
        else:
            ok = all(robot.getDevice(n) is not None for n in prof["match"])
        if ok:
            return prof
    print("[!] this robot matches none of the profiles (%s)." % ", ".join(names))
    print("    Copy atlas_profile() above, fill in this robot's joint names, and pass it as CUSTOM_PROFILE.")
    for n in sorted(device_names(robot) or ()):
        print("      device: %s" % n)
    raise SystemExit(1)


class SimEnd(Exception):
    pass


# ============================================================
# WEBOTS CONTROLLER
# ============================================================
REPEAT_W = [k ** -REPEAT_S for k in range(1, REPEAT_MAX + 1)]


class Walker(Supervisor):

    # ---------------------------------------------------------------- setup
    def __init__(self):
        super().__init__()
        random.seed(SEED)
        bt = int(round(self.getBasicTimeStep()))
        self.ts = max(bt, int(round(CONTROL_MS / bt)) * bt)
        self.dt = self.ts / 1000.0

        prof = detect_profile(self)
        apply_profile(prof)
        self.prof = PROF
        leg_names = [n for n in LEG if n]
        rest = dict(prof["rest"])
        for pair in (prof["swing"], prof["elbow"], prof["upper"]):
            for n in pair:
                if n:
                    rest.setdefault(n, 0.0)
        want = list(dict.fromkeys(leg_names + list(rest)))
        if prof["hold_others"]:
            for n in self._rotational_motor_names():
                if n not in want:
                    want.append(n)
                    rest[n] = 0.0
        self.names = device_names(self)
        self.motors, self.sens = {}, {}
        missing = []
        for n in want:
            m = self.getDevice(n) if (self.names is None or n in self.names) else None
            if m is None:
                if n in leg_names:
                    missing.append(n)
                else:
                    print("[init] note: joint %s not found, it is left alone" % n)
                    rest.pop(n, None)
                continue
            self.motors[n] = m
            s = self.getDevice(n + prof["sensor_suffix"]) if prof["sensor_suffix"] else None
            if s is None:
                try:
                    s = m.getPositionSensor()
                except Exception:
                    s = None
            if s is None:
                for suf in ("_sensor", "S", "Sensor"):
                    if self.names is not None and (n + suf) not in self.names:
                        continue
                    s = self.getDevice(n + suf)
                    if s is not None:
                        break
            self.sens[n] = s
            if s is not None:
                s.enable(self.ts)
        if missing:
            print("[!] leg motors not found:", missing)
            raise SystemExit(1)
        for m in self.motors.values():
            mv_ = m.getMaxVelocity()
            m.setVelocity(MOTOR_SPEED if mv_ <= 0 else min(MOTOR_SPEED, mv_))
        self.acc = self._first_device(prof["acc"])
        self.gyro = self._first_device(prof["gyro"])
        for d in (self.acc, self.gyro):
            if d is not None:
                d.enable(self.ts)
        self.cmd_q = [0.0] * 12              # last commanded leg angles (stands in for joint sensors that a robot may lack)
        self.fit_mode = "normal"             # "march" while the marching-in-place stage runs
        self.marched = False
        self.fb_sign = 1.0                   # flips the body-pitch sign if the feedback check finds it reversed
        self.walkers = {}                    # gaits that really step but fall after a while: raw material for the feedback tuning
        self.last_fall = None
        self.fbs = [0.0, 0.0, 0.0, 0.0]      # body pitch, roll and their rates (used by the feedback terms)
        self._fb_prev = None

        self.me = self.getSelf()
        if self.me is None:
            print("[!] Not a Supervisor. Set  supervisor TRUE  on the robot node in the world.")
            raise SystemExit(1)
        self.t_field = self.me.getField("translation")
        self.r_field = self.me.getField("rotation")
        if self.t_field is None or self.r_field is None:
            print("[!] This robot is not a Supervisor, so it cannot reset itself.")
            print("    In Webots: click the robot in the scene tree, set  supervisor  to TRUE, save the world, reset the simulation.")
            raise SystemExit(1)
        p0 = self.t_field.getSFVec3f()
        self.home_xy = (p0[0], p0[1])
        self.home_rot = list(self.r_field.getSFRotation())
        self.spawn_z = max(p0[2], PROF["min_root_z"]) + 0.01
        self.z_stand = p0[2]
        self.z_hint = max(p0[2], PROF["min_root_z"])
        self.z_floor = None            # root height at which the feet just touch the floor (found once)
        self.z_land = p0[2]
        print("[init] world start pose: x=%.3f y=%.3f z=%.3f rotation=%s" % (p0[0], p0[1], p0[2],
              [round(v, 3) for v in self.home_rot]))
        print("[init] NOTE: that saved rotation is ignored; every reset places the robot upright facing +x.")
        self.home_rot = list(UPRIGHT_ROT)

        self.leg = [self.motors.get(n) if n else None for n in LEG]
        self.leg_s = [self.sens.get(n) if n else None for n in LEG]
        if any(x is None for n, x in zip(LEG, self.leg_s) if n):
            print("[init] note: this robot has no joint position sensors; joint angles are taken from the commands instead")
        self.lim = []
        for m in self.leg:
            if m is None:
                self.lim.append((-3.0, 3.0))
                continue
            lo, hi = float(m.getMinPosition()), float(m.getMaxPosition())
            self.lim.append((-3.0, 3.0) if (lo == 0.0 and hi == 0.0) else (lo, hi))
        self.arm_pose = {n: v for n, v in rest.items() if n in self.motors}

        self.tick()                                   # sensors need one step before they read
        self.gyro_raw = self.acc_raw = False
        if self.gyro is not None:
            self.gyro_raw = max(abs(v) for v in self.gyro.getValues()) > 60.0
        if self.acc is not None:
            self.acc_raw = max(abs(v) for v in self.acc.getValues()) > 60.0

        print("=" * 66)
        print(" %s walker | control step %d ms | gyro raw=%s acc raw=%s | feedback terms %s"
              % (prof["name"], self.ts, self.gyro_raw, self.acc_raw, "searched" if FEEDBACK_ON else "off"))
        for n, (lo, hi) in zip(LEG, self.lim):
            if n:
                print("   %-10s limits [%+.2f, %+.2f]" % (n, lo, hi))
        targets = [(prof["upper"][0], prof["walk_upper"][0]), (prof["upper"][1], prof["walk_upper"][1]),
                   (prof["elbow"][0], prof["walk_elbow"][0]), (prof["elbow"][1], prof["walk_elbow"][1])]
        for n, want in targets:
            if not n or n not in self.motors:
                continue
            lo, hi = self.motors[n].getMinPosition(), self.motors[n].getMaxPosition()
            sent = want if (lo == 0.0 and hi == 0.0) else clamp(want, lo + 0.005, hi - 0.005)
            try:
                tq = "  max torque %.0f N.m" % self.motors[n].getMaxTorque()
            except Exception:
                tq = ""
            print("   %-10s limits [%+.3f, %+.3f]  walking target %+.3f  -> commanded %+.3f rad (%.0f deg)%s%s" % (
                  n, lo, hi, want, sent, math.degrees(sent), tq,
                  "   <-- target is beyond the joint limit, so it is clipped" if abs(sent - want) > 0.01 else ""))
        held = [n for n in self.arm_pose if n not in [t[0] for t in targets] and n not in prof["swing"]]
        if held:
            print("   held at a fixed angle: %s" % ", ".join(held))
        try:
            hn = prof["leg"]["hip"][0]
            print("   max torque (%s): %.2f N.m" % (hn, self.motors[hn].getMaxTorque()))
        except Exception:
            pass
        print("=" * 66)

        # calibration results (filled by calibrate() or loaded)
        self.sg = {"hip": [1, 1], "knee": [1, 1], "ankle": [1, 1]}
        self.crouch = CROUCH
        self.lean0 = 0.0                      # constant forward lean found by the balance scan
        self._arm_cmd = {}
        self.turn, self.turn_sign = 0.0, 1.0   # keyboard steering (-1..1) and its calibrated direction
        self.arms_walk = False                # True = arms already in the walking pose (set after calibration)
        self.last_human = (0.0, 0.0, 0.0)
        self.train_dir = "fwd"
        self.pen_scale = 1.0
        self.robust_noise = ROBUST_NOISE
        self.tols = {}                         # per direction: largest command noise the gait is known to survive
        self.model_path = MODEL_PATH
        self.clipped = 0
        self.fwd = (1.0, 0.0)
        self.origin = (0.0, 0.0)
        self.yaw0 = 0.0
        self.last_pos = (0.0, 0.0, 0.0)
        self.ep_t = 0.0
        self.prev_jp = [0.0] * 12
        self.q_stand = [0.0] * 12

    def tick(self):
        if self.step(self.ts) == -1:
            raise SimEnd()
        if FEEDBACK_ON:
            self._update_fb()

    def _update_fb(self):
        """Body pitch / roll from the supervisor orientation plus smoothed rates, for the balance feedback."""
        R = self.me.getOrientation()
        pitch, roll = -self.fb_sign * math.asin(clamp(R[6], -1.0, 1.0)), math.asin(clamp(R[7], -1.0, 1.0))
        if self._fb_prev is None:
            dp = dr = 0.0
        else:
            dp = 0.5 * self.fbs[2] + 0.5 * (pitch - self._fb_prev[0]) / self.dt
            dr = 0.5 * self.fbs[3] + 0.5 * (roll - self._fb_prev[1]) / self.dt
        self._fb_prev = (pitch, roll)
        self.fbs = [pitch, roll, dp, dr]

    def _first_device(self, names):
        for n in names:
            if self.names is not None and n not in self.names:
                continue
            d = self.getDevice(n)
            if d is not None:
                return d
        return None

    def _rotational_motor_names(self):
        names = []
        try:
            from controller import Node as WNode
            rot = getattr(WNode, "ROTATIONAL_MOTOR", None)
            if rot is not None:
                for i in range(self.getNumberOfDevices()):
                    d = self.getDeviceByIndex(i)
                    if d.getNodeType() == rot:
                        names.append(d.getName())
        except Exception:
            pass
        return names

    def _profile_seeds(self):
        return [dict(default_gait(), **ov) for ov in PROF.get("seeds", ())]

    # -------------------------------------------------------------- sensors
    def read_gyro(self):          # rad/s
        if self.gyro is None:
            return [0.0, 0.0, 0.0]
        v = self.gyro.getValues()
        if self.gyro_raw:
            return [(x - 512.0) / 512.0 * math.radians(1600.0) for x in v]
        return list(v)

    def read_acc(self):           # in g
        if self.acc is None:
            return [0.0, 0.0, 0.0]
        v = self.acc.getValues()
        if self.acc_raw:
            return [(x - 512.0) / (512.0 / 3.0) for x in v]
        return [x / 9.81 for x in v]

    def joint_pos(self):
        return [s.getValue() if s is not None else self.cmd_q[i] for i, s in enumerate(self.leg_s)]

    def tilt(self):
        R = self.me.getOrientation()
        return math.acos(clamp(R[8], -1.0, 1.0))

    def fell(self):
        return self.tilt() > FELL_TILT or self.me.getPosition()[2] < self.z_stand - FELL_DROP

    # ------------------------------------------------------------- actuation
    def command(self, q):
        for i, m in enumerate(self.leg):
            if m is None:
                continue
            lo, hi = self.lim[i]
            t = clamp(q[i], lo + 0.005, hi - 0.005)
            if t != q[i]:
                self.clipped += 1
            self.cmd_q[i] = t
            m.setPosition(t)
        for n, v0 in self.arm_pose.items():
            v = self._arm_cmd.get(n, v0)
            m = self.motors[n]
            lo, hi = m.getMinPosition(), m.getMaxPosition()
            m.setPosition(v if (lo == 0.0 and hi == 0.0) else clamp(v, lo + 0.005, hi - 0.005))

    def gait_pose(self, g, ph, ramp):
        """12 leg joint targets for gait g at phase ph in [0,1)."""
        c = self.crouch
        out = [0.0] * 12
        if g.get("phase_dir", 1) < 0:
            ph = (-ph) % 1.0                                   # backward walking = the pattern run in reverse
        rs = (1.0, float(g[FLIP_PARAM]))
        roll = g["roll_amp"] * ramp * math.sin(2.0 * math.pi * (ph + g["roll_phase"]))
        yaw = g["yaw_amp"] * ramp * math.sin(2.0 * math.pi * ph)
        asg = (1.0, float(g["arm_mirror"]))
        for side in (0, 1):
            pl = (ph + 0.5 * side) % 1.0
            hip_amp = g["hip_amp"] * (1.0 + self.turn * TURN_GAIN * (1.0, -1.0)[side])   # self.turn is 0 when training
            hip = 0.5 * c - hip_amp * ramp * math.cos(2.0 * math.pi * pl)
            if FEEDBACK_ON:                                    # hip strategy (like the PD term in the C sample)
                hip += ramp * g.get("fb_hip", 0.0) * (self.fbs[0] + 0.1 * self.fbs[2])
            knee = c + g["knee_amp"] * ramp * max(0.0, math.sin(2.0 * math.pi * pl))
            ank = (g["ankle_follow"] * (knee - hip) + (1.0 - g["ankle_follow"]) * 0.5 * c
                   + self.lean0 + g["lean"] * ramp)
            if FEEDBACK_ON:                                    # ankle strategy: lean forward -> reduce the forward lean
                ank -= ramp * (g.get("fb_kp", 0.0) * self.fbs[0] + g.get("fb_kd", 0.0) * self.fbs[2])
            out[J(HIP, side)] = self.sg["hip"][side] * hip
            out[J(KNEE, side)] = self.sg["knee"][side] * knee
            out[J(ANKLE, side)] = self.sg["ankle"][side] * ank
            lat = g.get("side_amp", 0.0) * ramp * math.cos(2.0 * math.pi * (pl + g.get("side_phase", 0.0)))
            out[J(ROLL, side)] = rs[side] * (roll + lat)
            out[J(FOOT, side)] = rs[side] * (g["foot_ratio"] * roll + g.get("side_foot", 0.0) * lat)
            if FEEDBACK_ON:
                out[J(FOOT, side)] += ramp * rs[side] * g.get("fb_roll", 0.0) * (self.fbs[1] + 0.1 * self.fbs[3])
            out[J(YAW, side)] = (1.0, -1.0)[side] * yaw          # the two yaw joints are mirrored
            name, elbow, upper = PROF["swing"][side], PROF["elbow"][side], PROF["upper"][side]
            k_arm = 1.0 if self.arms_walk else 0.0       # arms start in the walking pose, no swinging at reset
            if name in self.arm_pose:
                self._arm_cmd[name] = self.arm_pose[name] + asg[side] * g["arm_amp"] * ramp * math.cos(2.0 * math.pi * pl)
            if elbow in self.arm_pose:
                self._arm_cmd[elbow] = self.arm_pose[elbow] + (PROF["walk_elbow"][side] - self.arm_pose[elbow]) * k_arm
            if upper in self.arm_pose:
                self._arm_cmd[upper] = self.arm_pose[upper] + (PROF["walk_upper"][side] - self.arm_pose[upper]) * k_arm
        return out

    def stand_gait(self):
        g = default_gait()
        g.update(hip_amp=0.0, knee_amp=0.0, roll_amp=0.0, lean=0.0, yaw_amp=0.0, arm_amp=0.0)
        return g

    # ----------------------------------------------------------------- reset
    def _pin(self, z):
        self.t_field.setSFVec3f([self.home_xy[0], self.home_xy[1], z])
        self.r_field.setSFRotation(self.home_rot)
        self.me.resetPhysics()

    def find_floor(self, q):
        """Find the root height at which the feet touch the floor, without any contact API:
        hold the robot at a height, let go for 3 steps, and see whether it fell. Lower until it does not."""
        print("[floor] looking for the floor height (lowering the held robot in %.0f mm steps) ..." %
              (PROBE_STEP * 1000))
        z = self.z_hint + 0.08 * PROF["length"]
        while z > 0.43 * PROF["min_root_z"]:
            for _ in range(HOLD_STEPS):
                self._pin(z)
                self.command(q)
                self.tick()
            for _ in range(3):
                self.command(q)
                self.tick()
            fallen = z - self.me.getPosition()[2]
            print("   held at z=%.3f -> dropped %.3f m after release" % (z, fallen))
            if fallen < 0.012 * PROF["length"]:
                self.z_floor = self.me.getPosition()[2]      # where it actually came to rest
                self.z_hint = self.z_floor
                print("[floor] feet touch the floor at root height ~%.3f m" % self.z_floor)
                return
            z -= PROBE_STEP
        self.z_floor = self.z_hint
        print("[floor] !! could not find the floor; using %.3f m" % self.z_floor)

    def reset_to(self, q):
        """Place the robot upright just above the floor in pose q, hold it while the joints get there,
        release it gently, let it settle."""
        if self.z_floor is None:
            self.find_floor(q)
        z = self.z_floor + 0.004
        for _ in range(HOLD_STEPS):
            self._pin(z)
            self.command(q)
            self.tick()
        self.z_land = z
        for _ in range(SETTLE_STEPS):
            self.command(q)
            self.tick()

    def begin_episode(self):
        self.fbs = [0.0, 0.0, 0.0, 0.0]          # never carry the previous episode's (fallen) attitude into this one
        self._fb_prev = None
        p = self.me.getPosition()
        R = self.me.getOrientation()
        n = math.hypot(R[0], R[3]) or 1.0
        self.fwd = (R[0] / n, R[3] / n)
        self.yaw0 = math.atan2(R[3], R[0])
        self.origin = (p[0], p[1])
        self.last_pos = tuple(p)
        self.ep_t = 0.0
        self.prev_jp = self.joint_pos()

    def progress(self):
        p = self.me.getPosition()
        dx, dy = p[0] - self.origin[0], p[1] - self.origin[1]
        fx, fy = self.fwd
        R = self.me.getOrientation()
        yaw = wrap_pi(math.atan2(R[3], R[0]) - self.yaw0)
        return dx * fx + dy * fy, -dx * fy + dy * fx, yaw

    # ----------------------------------------------------------- calibration
    def _limit_prior(self, pair):
        """Sign of the larger-range side of a joint (flexion has the bigger range), per leg."""
        pri = []
        for side in (0, 1):
            lo, hi = self.lim[J(pair, side)]
            if abs(hi) > 1.4 * abs(lo):
                pri.append(1)
            elif abs(lo) > 1.4 * abs(hi):
                pri.append(-1)
            else:
                pri.append(0)
        return pri

    def calibrate(self):
        """Find the sign of hip/knee/ankle on each leg that gives a stable crouch."""
        print("[calibrate] searching joint sign conventions (crouch %.2f rad) ..." % CROUCH)
        self.crouch = CROUCH
        p_hip, p_knee = self._limit_prior(HIP), self._limit_prior(KNEE)
        print("[calibrate] joint-limit hint: hip %s knee %s (+1/-1 = side with the larger range)" %
              (p_hip, p_knee))
        knee_opts = [[p] if p != 0 else [1, -1] for p in p_knee]
        best = None
        for sk0 in knee_opts[0]:
            for sk1 in knee_opts[1]:
                for sh0 in (1, -1):
                    for sh1 in (1, -1):
                        for sa0 in (1, -1):
                            for sa1 in (1, -1):
                                self.sg = {"hip": [sh0, sh1], "knee": [sk0, sk1], "ankle": [sa0, sa1]}
                                score = self._cal_trial()
                                bonus = 0.3 * ((sh0 == p_hip[0]) + (sh1 == p_hip[1])) + 0.15 * (sa0 == -sa1)
                                print("   hip %+d/%+d knee %+d/%+d ankle %+d/%+d -> stand score %7.3f (+%.2f hint)"
                                      % (sh0, sh1, sk0, sk1, sa0, sa1, score, bonus))
                                if best is None or score + bonus > best[0]:
                                    best = (score + bonus, dict((k, list(v)) for k, v in self.sg.items()))
        self.sg = best[1]
        print("[calibrate] chosen:", self.sg)
        self._balance_scan()
        self.z_floor = None
        ok = self._verify_stand()
        return ok

    def _balance_scan(self):
        """The OP2 is rear-heavy, so a perfectly upright crouch tips backward. Try a few constant
        forward/backward leans (and a shallower crouch) and keep the one that stands best."""
        print("[balance] scanning standing lean (positive = lean forward) ...")
        best = None
        for crouch in (CROUCH, 0.15):
            self.crouch = crouch
            self.z_floor = None
            for lean in (-0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20):
                self.lean0 = lean
                sc = self._cal_trial(70)
                print("   crouch %.2f lean %+.2f -> stand score %7.3f" % (crouch, lean, sc))
                if best is None or sc - 0.01 * abs(lean) > best[0]:
                    best = (sc - 0.01 * abs(lean), crouch, lean)
            if best[0] > -0.2:
                break
        self.crouch, self.lean0 = best[1], best[2]
        print("[balance] chosen crouch %.2f lean %+.2f" % (self.crouch, self.lean0))

    def _cal_trial(self, steps=CAL_TRIAL_STEPS):
        q = self.gait_pose(self.stand_gait(), 0.0, 0.0)
        self.z_stand = 0.0                       # disable the height rule during the trial
        self.reset_to(q)
        z0 = self.me.getPosition()[2]
        tilts, drops = [], []
        for _ in range(steps):
            self.command(q)
            self.tick()
            tilts.append(self.tilt())
            drops.append(z0 - self.me.getPosition()[2])
        tail = tilts[-max(10, steps // 2):]
        return -(sum(tail) / len(tail)) - 2.0 * max(0.0, max(drops)) - (5.0 if max(tilts) > 1.0 else 0.0)

    def _verify_stand(self):
        """Stand for ~3 s with the chosen signs and print what happens (so problems are visible)."""
        q = self.gait_pose(self.stand_gait(), 0.0, 0.0)
        self.z_stand = 0.0
        self.reset_to(q)
        print("[verify] landed at root height %.3f m; now standing 3 s (t, z, tilt, right-leg tracking error "
              "hip/knee/ankle in rad)" % self.z_land)
        zs, tilts = [], []
        for k in range(96):
            self.command(q)
            self.tick()
            z, t = self.me.getPosition()[2], self.tilt()
            zs.append(z)
            tilts.append(t)
            if k % 12 == 0:
                jp = self.joint_pos()
                print("   t=%4.2fs z=%.3f tilt=%.2f  err %+.2f %+.2f %+.2f" %
                      (k * self.dt, z, t, q[J(HIP, 0)] - jp[J(HIP, 0)], q[J(KNEE, 0)] - jp[J(KNEE, 0)],
                       q[J(ANKLE, 0)] - jp[J(ANKLE, 0)]))
        R = self.me.getOrientation()
        a = self.read_acc()
        print("[verify] body x axis in world = (%.2f, %.2f, %.2f), body z axis in world = (%.2f, %.2f, %.2f)"
              % (R[0], R[3], R[6], R[2], R[5], R[8]))
        print("[verify] accelerometer (g) = (%.2f, %.2f, %.2f)   <- an upright robot shows about 1 g on one axis"
              % tuple(a))
        ok = max(tilts[-24:]) < 0.30 and (self.z_land - min(zs[-24:])) < 0.04
        self.z_stand = sum(zs[-24:]) / 24.0
        self.spawn_z = self.z_stand + 0.012
        if ok:
            print("[verify] OK: standing, root height %.3f m" % self.z_stand)
        else:
            print("[verify] !! the robot does NOT stay standing with the best sign combination.")
            print("         (a standing robot has tilt near 0 and body z axis near (0,0,1))")
            print("         Paste this whole console output. Things to check: the robot is on flat ground, the")
            print("         world 'basicTimeStep' is 16 or 8 ms, and no other controller moves the joints.")
        return ok

    # ------------------------------------------------------------ gait search
    @staticmethod
    def _sway_amp(ts, ys):
        """Half-range of the sideways oscillation after removing a straight-line drift."""
        n = len(ts)
        if n < 20:
            return 0.0
        mx, my = sum(ts) / n, sum(ys) / n
        sxx = sum((t - mx) ** 2 for t in ts)
        b = sum((t - mx) * (y - my) for t, y in zip(ts, ys)) / sxx if sxx > 0 else 0.0
        r = [y - my - b * (t - mx) for t, y in zip(ts, ys)]
        return 0.5 * (max(r) - min(r))

    @staticmethod
    def human_score(g, sway):
        """0..1: how human-like the motion is (visible lateral sway, arm swing, pelvis rotation,
        swing-phase knee bend)."""
        return (0.3 * clamp(sway / SWAY_TARGET, 0.0, 1.0) + 0.3 * clamp(abs(g["arm_amp"]) / 0.7, 0.0, 1.0)
                + 0.2 * clamp(abs(g["yaw_amp"]) / 0.08, 0.0, 1.0) + 0.2 * clamp(g["knee_amp"] / LIFT_TARGET, 0.0, 1.0))

    def eval_gait(self, g, steps=SEARCH_EP_STEPS, direction="fwd", noise=0.0):
        q0 = self.gait_pose(g, 0.0, 0.0)
        self.reset_to(q0)
        self.begin_episode()
        alive, fell = 0, False
        ts, lats = [], []
        for k in range(steps):
            t = k * self.dt
            ph = (t / g["period"]) % 1.0
            q = self.gait_pose(g, ph, min(1.0, t / RAMP_S))
            if noise > 0.0:
                q = [v + noise * random.uniform(-1.0, 1.0) for v in q]
            self.command(q)
            self.tick()
            if self.fell():
                fell = True
                break
            alive += 1
            if t > 1.0:
                ts.append(t)
                lats.append(self.progress()[1])
        fw, lt, yaw = self.progress()
        d = DIRS[direction]
        dist = d[0] * fw + d[1] * lt                    # distance travelled in the wanted direction
        lat = -d[1] * fw + d[0] * lt                    # drift away from that direction
        sway = self._sway_amp(ts, lats) if direction == "fwd" else 0.0
        h = self.human_score(g, sway) if direction == "fwd" else 0.0
        n_steps = max(1.0, alive * self.dt / (0.5 * g["period"]))        # a step = half a gait cycle
        step_len = max(0.0, dist) / n_steps
        self.last_human = (sway, h, step_len)
        if fell and FEEDBACK_ON:
            self.last_fall = (self.fbs[0], self.fbs[1], alive * self.dt)
        if (fell and PROF.get("gate_bonus") and direction == "fwd" and alive * self.dt >= 1.0
                and step_len >= 0.25 * STEP_TARGET):
            self._note_walker(g, alive * self.dt, dist, step_len)
        if self.fit_mode == "march":                    # marching in place: stay up, lift the knees, do not drift
            lift = clamp(g["knee_amp"] / MARCH_KNEE, 0.0, 1.0)
            fm = (-1.0 + 0.9 * alive / steps) if fell else (0.4 + 0.8 * lift - 0.8 * abs(dist) - 0.8 * abs(lat) - 0.3 * abs(yaw))
            return fm, dist, alive * self.dt, fell
        f = dist + 0.15 * alive / steps - 0.8 * abs(lat) - 0.3 * abs(yaw)
        if fell:                                        # falls rank by how long the robot stayed up, never by distance
            f = -1.0 + 0.9 * alive / steps + 0.2 * clamp(dist, -0.5, 0.5)
        if not fell:                                    # bonuses only count if it stayed up
            pf = 1.0
            if PROF.get("gate_bonus"):                  # ... and (hard robots) only in proportion to real stepping, so
                tgt = STEP_TARGET if direction == "fwd" else OMNI_STEP[direction]     # standing and swaying cannot win
                pf = clamp(step_len / (0.5 * tgt), 0.0, 1.0)
            if direction == "fwd":
                f += pf * (HUMAN_W * h + STRIDE_W * clamp(step_len / STEP_TARGET, 0.0, 1.0)
                           + LIFT_W * clamp(g["knee_amp"] / LIFT_TARGET, 0.0, 1.0))
            else:
                f += pf * (STRIDE_W * clamp(step_len / OMNI_STEP[direction], 0.0, 1.0)
                           + 0.5 * LIFT_W * clamp(g["knee_amp"] / LIFT_TARGET, 0.0, 1.0))
        return f, dist, alive * self.dt, fell

    def eval_robust(self, g, steps=SEARCH_EP_STEPS, direction="fwd"):
        """Fitness of a gait under command noise: two noisy runs, the worse one counts. A gait that only
        works in the one exact noise-free run (knife-edge) scores badly here."""
        if not ROBUST_SEARCH:
            return self.eval_gait(g, steps, direction)
        r1 = self.eval_gait(g, steps, direction, noise=self.robust_noise)
        if r1[3]:
            return r1                                          # fell on the first run, no need for a second
        lh1 = self.last_human
        r2 = self.eval_gait(g, steps, direction, noise=self.robust_noise)
        if r2[0] < r1[0]:
            return r2
        self.last_human = lh1
        return r1

    def survive_count(self, g, direction="fwd", n=3, noise=None, steps=None):
        noise = self.robust_noise if noise is None else noise
        ok = 0
        for _ in range(n):
            ok += 0 if self.eval_gait(g, steps or MAX_EP_STEPS, direction, noise=noise)[3] else 1
        return ok

    def stable_rate(self, g, direction="fwd", noise=None, n_clean=2, n_noisy=2, steps=None):
        """Hard robots do not repeat exactly (the same gait can pass once and fall the next time), so judge a gait by how
        many of several runs stay up. Returns (clean passes, noisy passes, mean distance and step length of the
        passing clean runs, mean fitness of those)."""
        noise = self.robust_noise if noise is None else noise
        steps = steps or MAX_EP_STEPS
        cp, ds, sl, fs = 0, [], [], []
        for _ in range(n_clean):
            f, d, t, fell = self.eval_gait(g, steps, direction)
            if not fell:
                cp += 1
                ds.append(d)
                sl.append(self.last_human[2])
                fs.append(f)
        npass = self.survive_count(g, direction, n_noisy, noise, steps)
        mean = lambda v: sum(v) / len(v) if v else 0.0
        return cp, npass, mean(ds), mean(sl), mean(fs)

    def polish_gait(self, g, direction="fwd"):
        """Make a saved gait tolerate command noise WITHOUT ever making it worse: raise the noise level step by
        step, search a little at each level, and keep a result only if it still walks cleanly and survives
        the noise. Returns (gait, largest noise level it survives; 0.0 if none)."""
        f0, d0, t0, fell0 = self.eval_gait(g, MAX_EP_STEPS, direction)
        print("[polish-%s] clean run of the saved gait: %+.3f m, fell=%s" % (direction, d0, fell0))
        best, tol = g, 0.0
        try:
            for lvl in ROBUST_LEVELS:
                self.robust_noise = lvl
                if PROF.get("gate_bonus"):
                    cp, npass, _, _, _ = self.stable_rate(best, direction, lvl, 2, 3)
                    ok = cp >= 1 and npass >= 2
                    print("[polish-%s] noise +-%.3f rad: current gait stays up in %d/2 clean and %d/3 noisy runs -> %s"
                          % (direction, lvl, cp, npass, "OK" if ok else "needs work"))
                else:
                    ok = self.survive_count(best, direction, 2, lvl) == 2
                    print("[polish-%s] noise +-%.3f rad: current gait survives 2/2 = %s" % (direction, lvl, ok))
                if not ok:
                    cand, _ = self.search_gait(start=best, evals=ROBUST_EVALS, label="robust-" + direction,
                                               direction=direction, persist=False)
                    if PROF.get("gate_bonus"):
                        cp, npass, dc, slc, _ = self.stable_rate(cand, direction, lvl, 3, 3)
                        ok = cp >= 2 and npass >= 2 and dc >= 0.6 * max(d0, 0.0)
                        print("[polish-%s] candidate: stays up in %d/3 clean and %d/3 noisy runs, mean %+.2f m -> %s"
                              % (direction, cp, npass, dc, "kept" if ok else "rejected"))
                    else:
                        fc, dc, tc, fellc = self.eval_gait(cand, MAX_EP_STEPS, direction)
                        ok = (not fellc and dc >= 0.6 * max(d0, 0.0)
                              and self.survive_count(cand, direction, 2, lvl) == 2)
                        print("[polish-%s] candidate: clean %+.3f m, fell=%s, survives noise = %s -> %s"
                              % (direction, dc, fellc, ok, "kept" if ok else "rejected"))
                    if ok:
                        best = cand
                if not ok:
                    break
                tol = lvl
        finally:
            self.robust_noise = ROBUST_NOISE
        print("[polish-%s] result: survives noise up to +-%.3f rad%s" %
              (direction, tol, "" if tol > 0 else "  (no improvement found; the original gait is kept)"))
        return best, tol

    def _persist(self, direction, best, best_f, **extra):
        """Forward gait lives at the top level of the state file, the others under 'omni'."""
        if direction == "fwd":
            self._save_state(gait=best, gait_f=best_f, **extra)
            return
        om = dict(self._load_state().get("omni", {}))
        rec = dict(om.get(direction, {}))
        rec.update(gait=best, f=best_f)
        if "search_done" in extra:
            rec["done"] = extra["search_done"]
        if "robust" in extra:
            rec["robust"] = extra["robust"]
        if "tol" in extra:
            rec["tol"] = extra["tol"]
        om[direction] = rec
        self._save_state(omni=om)

    def search_gait(self, start=None, evals=None, label="search", seeds=(), direction="fwd", persist=True, only=None):
        evals = evals or SEARCH_EVALS
        frozen = frozen_params(direction)
        if only is not None:                                   # tune just these parameters, freeze the rest
            frozen = tuple(k for k in PARAMS if k not in only)
        print("[%s] hill-climbing the gait: %d candidates, %d steps each" % (label, evals, SEARCH_EP_STEPS))
        if start is None:
            pool = [self.stand_gait(), default_gait()] + list(seeds) + [random_gait(direction) for _ in range(14)]
            best, best_f = None, -1e9
            for i, g in enumerate(pool):
                f, d, t, fell = self.eval_robust(g, direction=direction)
                print("   init %2d fit %+.3f dist %+.3f m alive %.1fs%s" % (i, f, d, t, " FELL" if fell else ""))
                if f > best_f:
                    best, best_f = g, f
            best_f = min(best_f, self.eval_robust(best, direction=direction)[0])     # must repeat to count
            persist and self._persist(direction, best, best_f, search_done=False)
            sigma = SEARCH_SIGMA
        else:
            f, d, t, fell = self.eval_robust(start, direction=direction)
            best, best_f = start, f
            print("   start: fit %+.3f dist %+.3f m alive %.1fs%s | sway %.1f cm, step %.1f cm, human %.2f" %
                  (f, d, t, " FELL" if fell else "", self.last_human[0] * 100, self.last_human[2] * 100,
                   self.last_human[1]))
            sigma = 0.10 if label.startswith(("humanize", "robust")) else (0.15 if label.startswith("stabilize") else SEARCH_SIGMA)
        wins = tries = 0
        for it in range(evals):
            cand = mutate_gait(best, sigma, frozen)
            f, d, t, fell = self.eval_robust(cand, direction=direction)
            tries += 1
            if f > best_f + 1e-6:
                f = min(f, self.eval_robust(cand, direction=direction)[0])     # second run: keep only repeatable gaits
                if f > best_f + 1e-6 and PROF.get("gate_bonus") and label.startswith("humanize"):
                    cpv, npv, _, _, _ = self.stable_rate(cand, direction, None, 2, 2)   # hard robot: a lucky run is not enough
                    if cpv < 2 or npv < 2:
                        print("   it %3d  looked better (fit %+.3f) but stays up in only %d/2 clean and %d/2 noisy runs -> skipped"
                              % (it, f, cpv, npv))
                        f = best_f
                if f > best_f + 1e-6:
                    wins += 1
                    best, best_f = cand, f
                    print("   it %3d  fit %+.3f  dist %+.3f m  alive %.1fs | sway %.1f cm, step %.1f cm, human %.2f  (best so far)"
                          % (it, f, d, t, self.last_human[0] * 100, self.last_human[2] * 100, self.last_human[1]))
                    persist and self._persist(direction, best, best_f)
            if tries == 10:
                sigma = clamp(sigma * (0.8 if wins < 2 else 1.25), 0.03, 0.4)
                wins, tries = 0, 0
        rv = ROBUST_VERSION if ROBUST_SEARCH else None
        if direction != "fwd":
            persist and self._persist(direction, best, best_f, search_done=True, robust=rv)
        elif label.startswith("search"):
            persist and self._save_state(gait=best, gait_f=best_f, search_done=True, humanized=False,
                                         robust=(None if PROF.get("gate_bonus") else rv))   # hard robots: measure the tolerance next
        elif label == "robust":
            persist and self._save_state(gait=best, gait_f=best_f, robust=rv)
        else:
            persist and self._save_state(gait=best, gait_f=best_f, humanized=HUMAN_VERSION, robust=rv)
        f, d, t, fell = self.eval_gait(best, steps=MAX_EP_STEPS, direction=direction)
        print("[%s] best gait: %s" % (label, gait_id(best)))
        print("[%s] re-run for %.1f s: distance %+.3f m, fell=%s, sway %.1f cm, step length %.1f cm, human score %.2f" %
              (label, MAX_EP_STEPS * self.dt, d, fell, self.last_human[0] * 100, self.last_human[2] * 100,
               self.last_human[1]))
        if ROBUST_SEARCH and persist:
            print("[%s] robustness: stays up in %d of 3 noisy runs (noise +-%.3f rad)"
                  % (label, self.survive_count(best, direction), self.robust_noise))
        return best, best_f

    def _explore_cap(self, d):
        """Largest exploration noise (rad) for direction d. Hard robots get half the measured tolerance: at the tolerance
        itself roughly every second episode fell (Atlas v20.2 log), which leaves little to learn from."""
        tol = self.tols.get(d, 1.0)
        return max(0.5 * tol, 0.001) if PROF.get("gate_bonus") else max(tol, 0.002)

    def check_feedback_sign(self):
        """Lean the standing robot forward / backward with the ankle command and see which way the measured pitch moves,
        so the balance feedback pushes the right way whatever the sign conventions of this robot are."""
        res = []
        for lean in (-0.12, 0.12):
            g = dict(self.stand_gait(), lean=lean)
            self.fb_sign = 1.0
            self.reset_to(self.gait_pose(g, 0.0, 1.0))
            self.begin_episode()
            self._fb_prev = None
            for k in range(20):
                self.command(self.gait_pose(g, 0.0, 1.0))
                self.tick()
                if self.fell():
                    break
            res.append(self.fbs[0])
        d = res[1] - res[0]
        if d < -0.004:
            self.fb_sign = -1.0
        print("[feedback] ankle lean +0.12 vs -0.12 rad changes the measured pitch by %+.3f rad -> %s"
              % (d, "pitch sign flipped so that feedback opposes the lean" if self.fb_sign < 0
                 else ("pitch sign OK" if d > 0.004 else "too small to tell, keeping the default sign")))

    def march_stage(self):
        """Hard robots: learn to MARCH IN PLACE first (stay up, lift the knees alternately, do not drift). Balance while
        a foot is in the air is the hard part; forward stride is then added on top of a stable march."""
        tune = ("period", "knee_amp", "ankle_follow", "lean", "roll_amp", "roll_phase", "foot_ratio") + FB_KEYS
        frozen = tuple(k for k in PARAMS if k not in tune)
        saved = self._load_state().get("march")
        self.fit_mode = "march"
        try:
            if saved:
                g0 = dict(default_gait(), **saved)
                if all(not self.eval_gait(g0, MARCH_STEPS, noise=self.robust_noise)[3] for _ in range(2)):
                    print("[march] reusing the saved stable march")
                    return g0
            print("[march] learning to march in place first (knee lift >= %.2f rad, no drift), %d candidates"
                  % (MARCH_KNEE, MARCH_EVALS))
            base = dict(default_gait(), hip_amp=0.0, yaw_amp=0.0, arm_amp=0.0, knee_amp=MARCH_KNEE, period=1.3,
                        fb_kp=0.3, fb_kd=0.1)
            pool = [dict(base, roll_mirror=mi, roll_phase=rp, roll_amp=ra)
                    for mi in (1, -1) for rp in (0.0, 0.25, 0.5, 0.75) for ra in (0.08, 0.16)]
            best, best_f, bt = None, -1e9, 0.0
            for i, g in enumerate(pool):
                f, d, t, fell = self.eval_robust(g, steps=MARCH_STEPS)
                if f > best_f:
                    best, best_f, bt = g, f, t
                    print("   march init %2d fit %+.3f alive %.1fs%s" % (i, f, t, " FELL" if fell else ""))
            sigma, wins, tries = 0.15, 0, 0
            for it in range(MARCH_EVALS):
                if best_f >= 1.0:
                    ok = all(not self.eval_gait(best, MARCH_STEPS, noise=self.robust_noise)[3] for _ in range(2))
                    if ok:
                        break
                cand = mutate_gait(best, sigma, frozen)
                f, d, t, fell = self.eval_robust(cand, steps=MARCH_STEPS)
                tries += 1
                if f > best_f + 1e-6:
                    wins += 1
                    best, best_f, bt = cand, f, t
                    print("   march it %3d  fit %+.3f  alive %.1fs  knee lift %.2f  drift %+.2f m%s"
                          % (it, f, t, cand["knee_amp"], d, "  FELL" if fell else ""))
                if tries == 10:
                    sigma = clamp(sigma * (0.8 if wins < 2 else 1.25), 0.05, 0.4)
                    wins, tries = 0, 0
        finally:
            self.fit_mode = "normal"
        if best_f < 0.9:
            print("[march] no stable march found (best fit %+.3f, alive %.1f s). Paste this output." % (best_f, bt))
            return None
        print("[march] stable march found: %s" % gait_id(best))
        self._save_state(march=best)
        return best

    def stride_creep(self, march):
        """Grow the forward hip swing from a stable march. At each size the gait must stay up in noisy full-length runs AND
        really travel forward (a march that merely drifts does not count). If not, only the balance settings are
        re-tuned (up to twice); the progress so far is saved after every size. Returns (gait, fitness)."""
        tune = ("period", "knee_amp", "ankle_follow", "lean", "roll_amp", "roll_phase", "foot_ratio") + FB_KEYS

        def level_ok(c, h):
            cp, npass, d, sl, f = self.stable_rate(c, "fwd", None, 2, 2, steps=SEARCH_EP_STEPS)
            self._lv = (d, sl, f)
            need = min(0.4, 6.0 * h)
            return cp + npass >= 3 and cp >= 1 and d >= need, d, cp + npass

        good, good_f = dict(march), 0.0
        for h in (0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20):
            cand = dict(good, hip_amp=h)
            ok, d, npass = level_ok(cand, h)
            rounds = 0
            while not ok and rounds < 2:
                rounds += 1
                print("[stride] hip swing %.2f %s (%d/4 runs stayed up), re-tuning the balance (%d candidates, round %d)"
                      % (h, "falls too often" if d < min(0.4, 6.0 * h) and npass < 3 else
                         "only travels %+.2f m (needs %.2f)" % (d, min(0.4, 6.0 * h)), npass, CREEP_EVALS, rounds))
                cand, _ = self.search_gait(start=cand, evals=CREEP_EVALS, label="stabilize", persist=False, only=tune)
                ok, d, npass = level_ok(cand, h)
            print("[stride] hip swing %.2f: %s (%d/4 runs stayed up, step %.1f cm, %+.2f m in %.1f s)"
                  % (h, "OK" if ok else "not good enough, stopping here", npass, 100 * self._lv[1], d, SEARCH_EP_STEPS * self.dt))
            if not ok:
                break
            good, good_f = cand, self._lv[2]
            self._save_state(gait=good, gait_f=good_f, search_done=False, humanized=False, robust=None)
        print("[stride] largest good hip swing %.2f" % good["hip_amp"])
        self._save_state(gait=good, gait_f=good_f, search_done=False, humanized=False, robust=None)
        return good, good_f

    def _note_walker(self, g, alive_s, dist, step):
        key = gait_id(g)
        old = self.walkers.get(key)
        if old is None or alive_s > old[0]:
            self.walkers[key] = (alive_s, dist, step, dict(g))
        if len(self.walkers) > 40:
            self.walkers = dict(sorted(self.walkers.items(), key=lambda kv: -kv[1][0])[:20])

    def walks(self, g):
        """True if the gait stays up for a full episode AND really steps (not a shuffle / sway in place)."""
        f, d, t, fell = self.eval_gait(g, steps=MAX_EP_STEPS)
        return (not fell) and self.last_human[2] >= WALK_STEP_FRAC * STEP_TARGET, f

    def stabilize(self, current, current_f, tops=4):
        """Hard robots: take gaits that really step but fall after a few seconds and tune ONLY the balance
        feedback (ankle / hip / roll gains, lean, sway) so they stay up. Returns (gait, fit, stable)."""
        tune = FB_KEYS + ("lean", "roll_amp", "roll_phase", "foot_ratio", "ankle_follow")
        pick = [r[3] for r in sorted(self.walkers.values(), key=lambda r: -r[0])[:tops]]
        pick += self._profile_seeds()[:max(0, tops + 2 - len(pick))]
        print("[stabilize] tuning the balance feedback of %d stepping gaits that fall (%d collected, plus seeds)"
              % (len(pick), len(self.walkers)))
        best_g, best_f = current, current_f
        for n, g0 in enumerate(pick):
            trials = [dict(g0, fb_kp=kp, fb_kd=kd, fb_hip=0.0) for kp in (0.3, 0.7, 1.0) for kd in (0.05, 0.15)]
            trials += [dict(g0, fb_kp=0.7, fb_kd=0.1, fb_hip=h) for h in (-0.4, 0.4)]
            bl, bf, bt, bfell, att = None, -1e9, 0.0, True, None
            for g in trials:
                f, d, t, fell = self.eval_robust(g)
                if f > bf:
                    bl, bf, bt, bfell = g, f, t, fell
                    att = self.last_fall
            print("[stabilize] gait %d/%d: best of %d feedback settings: fit %+.3f, %s%s"
                  % (n + 1, len(pick), len(trials), bf, "FELL after %.1f s" % bt if bfell else "stays up",
                     " (leaning pitch %+.2f roll %+.2f rad when it fell)" % (att[0], att[1]) if bfell and att else ""))
            g1, f1 = self.search_gait(start=bl, evals=STAB_EVALS, label="stabilize", persist=False, only=tune)
            ok, fc = self.walks(g1)
            if ok:
                print("[stabilize] gait %d stays up and really steps: step %.1f cm" % (n + 1, 100 * self.last_human[2]))
                return g1, f1, True
            if f1 > best_f:
                best_g, best_f = g1, f1
        return best_g, best_f, False

    def omni_stage(self, fwd):
        """Search backward / left / right gaits (each from its own seeds) and return {direction: gait}."""
        out = {}
        for name in ("back", "left", "right"):
            om = self._load_state().get("omni", {})
            rec = om.get(name)
            if rec and rec.get("done") and ("omni-" + name) not in REDO:
                g0 = dict(default_gait(), **rec["gait"])
                if (ROBUST_SEARCH and rec.get("robust") != ROBUST_VERSION
                        and not (KEYBOARD_MODE and not KEYBOARD_WAIT_FOR_HUMANIZE)):
                    print("[omni-%s] saved gait was not tested under noise: polishing it (safely)" % name)
                    g0, tol = self.polish_gait(g0, name)
                    self.tols[name] = tol
                    self._persist(name, g0, rec.get("f", 0.0), search_done=True, robust=ROBUST_VERSION, tol=tol)
                else:
                    print("[omni-%s] loaded saved gait" % name)
                    self.tols[name] = rec.get("tol", 1.0)
                out[name] = g0
                continue
            if name == "back":
                seeds = [dict(fwd, phase_dir=-1, side_amp=0.0)]
            else:
                base = dict(fwd, hip_amp=0.02, roll_amp=0.0, yaw_amp=0.0, arm_amp=0.0, phase_dir=1)
                seeds = [dict(base, side_amp=a, side_phase=ph)
                         for a in (0.3, -0.3) for ph in (0.0, 0.25, 0.5, 0.75)]
                if name == "right" and "left" in out:          # a mirror image of the left gait is a good guess
                    mirror = dict(out["left"])
                    mirror["side_amp"] = -mirror.get("side_amp", 0.0)
                    seeds.insert(0, mirror)
            start = dict(default_gait(), **rec["gait"]) if rec and rec.get("gait") else None
            if start is not None:
                print("[omni-%s] resuming an unfinished search" % name)
            out[name], _ = self.search_gait(start=start, evals=OMNI_EVALS, label="omni-" + name,
                                            seeds=seeds, direction=name)
        return out

    # -------------------------------------------------------------------- RL
    def build_state(self, ph, prev_a):
        jp = self.joint_pos()
        g, a = self.read_gyro(), self.read_acc()
        s = [jp[i] - self.q_stand[i] for i in range(12)]
        s += [clamp((jp[i] - self.prev_jp[i]) / self.dt, -8.0, 8.0) / 4.0 for i in range(12)]
        s += [clamp(v, -6.0, 6.0) / 3.0 for v in g]
        s += [clamp(v, -3.0, 3.0) / 1.5 for v in a]
        s += [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]
        s += [v / MAX_ACTION for v in prev_a]
        if SHARED_POLICY:
            s += [1.0 if d == self.train_dir else 0.0 for d in DIR_NAMES]
        self.prev_jp = jp
        return s

    def reward(self, a, prev_a, v_tgt):
        p = self.me.getPosition()
        dx, dy = (p[0] - self.last_pos[0]) / self.dt, (p[1] - self.last_pos[1]) / self.dt
        self.last_pos = tuple(p)
        fx, fy = self.fwd
        v_f, v_l = dx * fx + dy * fy, -dx * fy + dy * fx
        dd = DIRS[self.train_dir]
        v_f, v_l = dd[0] * v_f + dd[1] * v_l, -dd[1] * v_f + dd[0] * v_l   # along / across the wanted direction
        _, _, yaw = self.progress()
        tilt = self.tilt()
        r = 0.4
        r += 2.0 * math.exp(-((v_f - v_tgt) / 0.06) ** 2)
        pen = self.pen_scale                              # 0 -> 1 over PENALTY_RAMP_STEPS (URMA curriculum)
        r -= pen * (1.5 * tilt * tilt + 2.0 * abs(v_l) + 0.5 * abs(yaw))
        r -= pen * 0.1 * sum((x / MAX_ACTION) ** 2 for x in a) / 12.0
        r -= pen * 0.1 * sum(((a[i] - prev_a[i]) / MAX_ACTION) ** 2 for i in range(12)) / 12.0
        return clamp(r, -4.0, 3.0)

    @staticmethod
    def _tagged(path):
        if not EXPERIMENT:
            return path
        root, ext = os.path.splitext(path)
        return "%s_%s%s" % (root, EXPERIMENT, ext)

    def _train_setup(self, gaits, tdir="fwd"):
        """Which directions are trained, the id that ties a saved model to its gaits, and the model file."""
        if SHARED_POLICY:
            dirs = [d for d in DIR_NAMES if d in gaits]
            path = MODEL_PATH.replace(".json", "_shared.json")
        else:
            dirs = [tdir if tdir in gaits else "fwd"]
            path = MODEL_PATH if dirs[0] == "fwd" else MODEL_PATH.replace(".json", "_%s.json" % dirs[0])
        tag = json.dumps({d: gait_id(gaits[d]) for d in dirs}, sort_keys=True)
        return dirs, tag, self._tagged(path)

    def train(self, gaits, speeds, tdir="fwd"):
        dirs, tag, self.model_path = self._train_setup(gaits, tdir)
        agent = TD3Agent()
        loaded = LOAD_MODEL and agent.load(self.model_path, tag)
        buf = ReplayBuffer(BUFFER_CAP)
        self.q_stand = self.gait_pose(self.stand_gait(), 0.0, 0.0)
        log_path = self._tagged(LOG_PATH)
        new_log = not os.path.exists(log_path)
        log = open(log_path, "a")
        if new_log:
            log.write("episode,total_steps,steps,return,distance_m,fell,updates,direction\n")
        print("[train] TD3 residual (%s) on: %s | layer norm %s, penalty ramp %d, experiment '%s'"
              % ("ONE shared network" if SHARED_POLICY else "single direction", ", ".join(
                 "%s %.2f m/s" % (d, clamp(speeds[d], 0.03, 0.25)) for d in dirs),
                 LAYER_NORM, PENALTY_RAMP_STEPS, EXPERIMENT))
        print("[train] exploration noise is capped at each gait's noise tolerance (rad): %s"
              % ", ".join("%s %.3f" % (d, self._explore_cap(d)) for d in dirs))
        try:
            self._train_loop(agent, buf, gaits, dirs, speeds, tag, loaded, log)
        finally:
            log.close()
            if SAVE_MODEL and agent.updates > 0:
                agent.save(self.model_path, tag)
                print("[train] model saved (%d updates)" % agent.updates)

    def _train_loop(self, agent, buf, gaits, dirs, speeds, tag, loaded, log):
        total, ep = 0, 0
        recent = []                                            # (distance in wanted direction, fell, steps)
        while True:
            ep += 1
            # velocity curriculum (DeepWalk): the first direction only, then every direction at random
            d = random.choice(dirs) if (len(dirs) > 1 and (total >= DIR_CURRICULUM_STEPS or loaded)) else dirs[0]
            self.train_dir = d
            gait, v_tgt = gaits[d], clamp(speeds[d], 0.03, 0.25)
            capn = self._explore_cap(d)                       # exploration may not exceed what this gait survives
            self.reset_to(self.gait_pose(gait, 0.0, 0.0))
            self.begin_episode()
            prev_a = zeros(12)
            ph = 0.0
            s = self.build_state(ph, prev_a)
            ret, fell, k = 0.0, False, 0
            agent.rnorm.reset_episode()
            eps, eps_left = None, 0
            for k in range(1, MAX_EP_STEPS + 1):
                noise = (EXPLORE_END + (EXPLORE_START - EXPLORE_END) * max(0.0, 1.0 - total / EXPLORE_STEPS)
                         if not loaded else EXPLORE_END)
                if total < WARMUP_STEPS and not loaded:
                    a = [random.uniform(-1, 1) * min(MAX_ACTION * 0.3, capn) for _ in range(12)]
                else:
                    if NOISE_REPEAT:
                        if eps_left <= 0:                     # draw a new noise vector and how long to hold it
                            eps = [random.gauss(0, 1) for _ in range(12)]
                            eps_left = random.choices(range(1, REPEAT_MAX + 1), weights=REPEAT_W)[0]
                        eps_left -= 1
                    a = agent.act(s, min(noise, capn / MAX_ACTION), eps if NOISE_REPEAT else None)
                t = (k - 1) * self.dt
                ph = (t / gait["period"]) % 1.0
                q = self.gait_pose(gait, ph, min(1.0, t / RAMP_S))
                self.command([q[i] + a[i] for i in range(12)])
                self.tick()
                ph2 = ((t + self.dt) / gait["period"]) % 1.0
                s2 = self.build_state(ph2, a)
                fell = self.fell()
                self.pen_scale = 1.0 if (loaded or PENALTY_RAMP_STEPS <= 0) else min(1.0, total / PENALTY_RAMP_STEPS)
                r = self.reward(a, prev_a, v_tgt) - (3.0 if fell else 0.0)
                ret += r                                          # logged raw, so curves stay comparable
                if REWARD_NORM:
                    r = agent.rnorm(r)
                buf.push(s, a, r, s2, 1.0 if fell else 0.0)
                total += 1
                if len(buf) >= WARMUP_STEPS and total % UPDATE_EVERY == 0:
                    agent.update(buf)
                s, prev_a = s2, a
                if fell:
                    break
            fw, lt, _ = self.progress()
            dv = DIRS[d]
            dist = dv[0] * fw + dv[1] * lt
            recent = (recent + [(dist, fell, k)])[-20:]
            log.write("%d,%d,%d,%.2f,%.3f,%d,%d,%s\n" % (ep, total, k, ret, dist, int(fell), agent.updates, d))
            log.flush()
            m10 = [x[0] for x in recent[-10:]]
            print("[ep %4d] %-5s steps %3d ret %7.1f dist %+.3f m %s | mean dist(10) %+.3f | upd %d"
                  % (ep, d, k, ret, dist, "FELL" if fell else "ok  ", sum(m10) / len(m10), agent.updates))
            if ep % 20 == 0:
                print("[summary] last %d episodes: mean dist %+.3f m, falls %d%%, mean steps %.0f  (use this to compare runs)"
                      % (len(recent), sum(x[0] for x in recent) / len(recent),
                         100 * sum(1 for x in recent if x[1]) / len(recent), sum(x[2] for x in recent) / len(recent)))
            if SAVE_MODEL and ep % 10 == 0 and agent.updates > 0:
                agent.save(self.model_path, tag)

    def play(self, gait, use_policy, gaits=None):
        agent = TD3Agent()
        dirs, tag, path = self._train_setup(gaits or {"fwd": gait})
        ok = use_policy and agent.load(path, tag)
        self.train_dir = "fwd"
        self.q_stand = self.gait_pose(self.stand_gait(), 0.0, 0.0)
        print("[play] replaying gait %s" % ("+ learned residual" if ok else "only"))
        while True:
            self.reset_to(self.gait_pose(gait, 0.0, 0.0))
            self.begin_episode()
            prev_a, s = zeros(12), None
            for k in range(MAX_EP_STEPS):
                t = k * self.dt
                ph = (t / gait["period"]) % 1.0
                a = agent.act(self.build_state(ph, prev_a), 0.0) if ok else zeros(12)
                q = self.gait_pose(gait, ph, min(1.0, t / RAMP_S))
                self.command([q[i] + a[i] for i in range(12)])
                self.tick()
                prev_a = a
                if self.fell():
                    break
            print("[play] distance %+.3f m after %.1f s" % (self.progress()[0], (k + 1) * self.dt))

    # --------------------------------------------------------------- keyboard
    def _drain_keys(self, kb):
        keys = set()
        while True:
            k = kb.getKey()
            if k == -1:
                break
            keys.add(k & 0xFFFF)
        return keys

    def _fresh_start(self, gait):
        self.reset_to(self.gait_pose(gait, 0.0, 0.0))
        self.begin_episode()

    def _calibrate_turn(self, gait):
        """Walk 3 s with turn=+1 and see which way the robot actually turns, so A always means left."""
        self._fresh_start(gait)
        self.turn = 1.0
        for k in range(int(3.0 / self.dt)):
            t = k * self.dt
            self.command(self.gait_pose(gait, (t / gait["period"]) % 1.0, min(1.0, t / RAMP_S)))
            self.tick()
            if self.fell():
                break
        yaw = self.progress()[2]
        self.turn = 0.0
        self.turn_sign = 1.0 if yaw >= 0 else -1.0
        print("[keyboard] turn test: positive turn changed heading by %+.0f deg -> %s"
              % (math.degrees(yaw), "A = left" if yaw >= 0 else "turn direction flipped so A = left"))
        if abs(yaw) < 0.15:
            print("[keyboard] turning is weak; raise TURN_GAIN (e.g. 0.8) if you want sharper turns")

    def drive(self, gaits):
        """Keyboard control. Click the 3D view first so it receives the keys."""
        try:
            from controller import Keyboard
        except ImportError:
            Keyboard = None
        K = lambda name, default: getattr(Keyboard, name, default)
        UP, DOWN, LEFT, RIGHT = K("UP", 315), K("DOWN", 317), K("LEFT", 314), K("RIGHT", 316)
        kb = self.getKeyboard()
        kb.enable(self.ts)
        self.q_stand = self.gait_pose(self.stand_gait(), 0.0, 0.0)
        self._calibrate_turn(gaits["fwd"])
        have = [d for d in ("fwd", "back", "left", "right") if d in gaits]
        print("=" * 62)
        print(" KEYBOARD CONTROL (click the 3D view first)")
        print("   W / Up      walk forward         S / Down   walk backward")
        print("   Q           side-step left       E          side-step right")
        print("   A / Left    turn left (hold)     D / Right  turn right (hold)")
        print("   Space       stop                 R          reset upright")
        print("   gaits available: %s" % ", ".join(have))
        print("=" * 62)
        keymap = {ord("W"): "fwd", UP: "fwd", ord("S"): "back", DOWN: "back", ord("Q"): "left", ord("E"): "right"}
        cur, want = "fwd", None
        gait = gaits[cur]
        self._fresh_start(gait)
        level, ph, last_turn_t, t_now, turn_cmd = 0.0, 0.0, -9.0, 0.0, 0.0
        while True:
            for k in self._drain_keys(kb):
                if k in keymap:
                    if keymap[k] in gaits:
                        want = keymap[k]
                    else:
                        print("[keyboard] no %s gait yet (set OMNI = True and let the search finish)" % keymap[k])
                elif k == ord(" "):
                    want = None
                elif k in (ord("A"), LEFT):
                    turn_cmd, last_turn_t = 1.0, t_now
                elif k in (ord("D"), RIGHT):
                    turn_cmd, last_turn_t = -1.0, t_now
                elif k == ord("R"):
                    want, level, ph = None, 0.0, 0.0
                    self._fresh_start(gait)
            if t_now - last_turn_t > 0.3:                       # turning lasts only while the key is held
                turn_cmd = 0.0
            sign = -1.0 if cur == "back" else 1.0               # turning is reversed when walking backward
            self.turn = 0.0 if cur in ("left", "right") else turn_cmd * self.turn_sign * sign
            step = self.dt / RAMP_S
            if want is not None and want == cur:
                level = min(1.0, level + step)
            else:                                               # stopping, or changing direction: fade out first
                level = max(0.0, level - step)
                if level == 0.0 and want is not None:
                    cur, gait, ph = want, gaits[want], 0.0
            if level > 0.0:
                ph = (ph + self.dt / gait["period"]) % 1.0
            self.command(self.gait_pose(gait, ph, level))
            self.tick()
            t_now += self.dt
            if self.fell():
                print("[keyboard] the robot fell, resetting upright (press a walk key to go again)")
                want, level, ph = None, 0.0, 0.0
                self._fresh_start(gait)

    # ------------------------------------------------------------ persistence
    def _load_state(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH) as f:
                    d = json.load(f)
                if d.get("ver") == STATE_VERSION:
                    return d
                print("[state] %s is from an older version of this script, ignoring it" % STATE_PATH)
            except Exception:
                pass
        return {}

    def _backup_state(self, tag):
        """Copy the state file at a milestone so a later stage can never cost you a good gait."""
        try:
            import shutil
            dst = STATE_PATH.replace(".json", "_%s.json" % tag)
            shutil.copyfile(STATE_PATH, dst)
            print("[backup] %s -> %s" % (STATE_PATH, dst))
        except Exception as e:
            print("[backup] could not copy the state file (%s)" % e)

    def _save_state(self, **kw):
        st = self._load_state()
        st.update(kw)
        st["ver"] = STATE_VERSION
        with open(STATE_PATH, "w") as f:
            json.dump(st, f, indent=1)

    # -------------------------------------------------------------------- main
    def run(self):
        st = self._load_state()
        try:
            if "sg" in st and "calibrate" not in REDO:
                self.sg, self.z_stand = st["sg"], st["z_stand"]
                self.z_floor = st.get("z_floor")
                self.crouch, self.lean0 = st.get("crouch", CROUCH), st.get("lean0", 0.0)
                self.spawn_z = self.z_stand + 0.012
                print("[calibrate] loaded from %s: %s, standing height %.3f, crouch %.2f, lean %+.2f"
                      % (STATE_PATH, self.sg, self.z_stand, self.crouch, self.lean0))
            else:
                if not self.calibrate():
                    print("[calibrate] stopping: no stable standing pose yet, nothing saved.")
                    return
                self._save_state(sg=self.sg, z_stand=self.z_stand, z_floor=self.z_floor,
                                 crouch=self.crouch, lean0=self.lean0)
                st = self._load_state()

            print("[size] standing root height %.2f m (the DARwIn-OP is about 0.20 m); profile leg-length scale %.1f, time scale %.1f"
                  % (self.z_stand, PROF["length"], PROF["time"]))
            # from here on the arms are held in their walking pose, including during every reset
            self.arms_walk = True
            if st.get("arms_sig") != arms_sig():
                print("[balance] the arm pose changed since the last calibration: re-checking the standing "
                      "balance with the arms in the walking pose")
                self._balance_scan()
                self.z_floor = None
                ok = self._verify_stand()
                self._save_state(z_stand=self.z_stand, z_floor=self.z_floor, crouch=self.crouch,
                                 lean0=self.lean0, arms_sig=arms_sig())
                if not ok:
                    print("[balance] stopping: the robot cannot stand with these arm angles. Try smaller "
                          "ELBOW_* / ARMUP_* values.")
                    return
                st = self._load_state()

            if PROF["name"] != "darwin" and (st.get("forced") or (PROF.get("gate_bonus") and st.get("fit_ver") != FIT_VERSION)):
                if st.get("gait") is not None:
                    print("[restore] the saved gait is from an older fitness function or from the Darwin FORCE_GAIT: dropping it and searching again")
                self._save_state(gait=None, gait_f=0.0, search_done=False, robust=None, forced=None, humanized=0,
                                 fit_ver=FIT_VERSION)
                st = self._load_state()
            if FORCE_GAIT is not None and PROF["name"] == "darwin":
                fg = dict(default_gait(), **FORCE_GAIT)
                if st.get("forced") != gait_id(fg):
                    print("[restore] using FORCE_GAIT as the forward gait (applied once; the previous saved gait is "
                          "kept as 'gait_backup' in %s)" % STATE_PATH)
                    self._save_state(gait_backup=st.get("gait"), gait=fg, gait_f=st.get("gait_f", 0.0),
                                     search_done=True, humanized=HUMAN_VERSION, robust=None, forced=gait_id(fg))
                    st = self._load_state()
            if FEEDBACK_ON:
                self.check_feedback_sign()
            gait = st.get("gait")
            fresh = False
            if gait is not None:
                gait = dict(default_gait(), **gait)          # old saved gaits lack the newer parameters
            hard = bool(PROF.get("gate_bonus"))
            if hard and (gait is None or "search" in REDO or not st.get("search_done", False)):
                print("[search] hard-robot mode: no blind search; learning to march in place first, then adding stride")
                if gait is None or "search" in REDO:
                    gait = self.stand_gait()
                f, fresh = 0.0, True
            elif gait is None or "search" in REDO:
                gait, f = self.search_gait(seeds=self._profile_seeds())
                fresh = True
            elif not st.get("search_done", False):
                print("[search] resuming an unfinished search")
                gait, f = self.search_gait(start=gait)
                fresh = True
            else:
                print("[search] loaded best gait:", gait_id(gait))
            if not fresh and not hard:
                f0, d0, t0, fell0 = self.eval_gait(gait)
                if fell0 and t0 < 4.0:
                    print("[search] the saved gait falls after %.1f s with the current arm pose / balance, so "
                          "searching again (the saved gait is kept as one starting candidate)" % t0)
                    gait, f = self.search_gait(seeds=[gait] + self._profile_seeds())
            if not PROF.get("gate_bonus") and not hard:
                f0, d0, t0, fell0 = self.eval_gait(gait, steps=MAX_EP_STEPS)
                if fell0 or d0 < 0.05:
                    print("[escalate] the ordinary search did not produce a gait that walks (%s); switching to the harder-robot "
                          "curriculum: march in place first, then grow the stride" % ("falls after %.1f s" % t0 if fell0 else "barely moves"))
                    PROF["gate_bonus"] = True
                    PROF["feedback"] = True
                    enable_feedback()
                    hard = True
            if PROF.get("gate_bonus"):
                for rnd in range(4):                          # a hard robot must really step before anything is polished
                    ok, f0 = self.walks(gait)
                    if ok:
                        break
                    print("[search] this gait does not really walk yet (step %.1f cm)" % (100 * self.last_human[2]))
                    if not self.marched:
                        self.marched = True
                        gm = self.march_stage()
                        if gm is not None:
                            gait, f = self.stride_creep(gm)
                            ok, _ = self.walks(gait)
                            if ok:
                                break
                    print("[search] searching %d candidates from the best stable gait so far" % SEARCH_EVALS)
                    gait, f = self.search_gait(start=gait, evals=SEARCH_EVALS, label="search+")
            st = self._load_state()
            if (ROBUST_SEARCH and st.get("robust") != ROBUST_VERSION
                    and not (KEYBOARD_MODE and not KEYBOARD_WAIT_FOR_HUMANIZE)):
                print("[robust] the saved gait was never tested under command noise: polishing it step by step. "
                      "The result replaces the saved gait only if it still walks and survives the noise.")
                self._save_state(gait_backup=gait)
                gait, tol = self.polish_gait(gait, "fwd")
                self._save_state(gait=gait, robust=ROBUST_VERSION, tol=tol)
                st = self._load_state()
            self.tols["fwd"] = st.get("tol", 1.0)
            self._backup_state("before_humanize")
            if (HUMANIZE and (st.get("humanized") != HUMAN_VERSION or "humanize" in REDO)
                    and not (KEYBOARD_MODE and not KEYBOARD_WAIT_FOR_HUMANIZE)):
                print("[humanize] adding human-like sway / arm swing / pelvis rotation on top of the stable gait")
                pre_h = gait
                gait, f = self.search_gait(start=gait, evals=HUMAN_EVALS, label="humanize")
                if PROF.get("gate_bonus") and gait is not pre_h:
                    cp1, np1, _, _, _ = self.stable_rate(gait, "fwd", None, 3, 3)
                    cp0, np0, _, _, _ = self.stable_rate(pre_h, "fwd", None, 3, 3)
                    print("[humanize] check: humanized gait stays up in %d/3 clean, %d/3 noisy; before humanize %d/3 clean, %d/3 noisy"
                          % (cp1, np1, cp0, np0))
                    if cp1 < 2 or cp1 + np1 < cp0 + np0:
                        print("[humanize] the humanized gait is less reliable than the one before it -> keeping the one before")
                        gait = pre_h
                        self._save_state(gait=gait, humanized=HUMAN_VERSION)
            self._backup_state("after_humanize")
            f, d, t, fell = self.eval_gait(gait, steps=MAX_EP_STEPS)
            speed = max(0.0, d) / (MAX_EP_STEPS * self.dt)
            print("[gait] open-loop check: %+.3f m in %.1f s, fell=%s (speed %.3f m/s)" %
                  (d, t, fell, speed))
            if d < 0.05 or (fell and t < 4.0):
                print("[gait] !! the gait does not stay up (%s). Rerun with REDO=('search',) and a larger "
                      "SEARCH_EVALS, or try smaller arm angles, or paste this console output."
                      % ("falls after %.1f s" % t if fell else "barely moves"))
                if not TRAIN_EVEN_IF_GAIT_FAILS:
                    print("[gait] stopping before RL training (it would only learn on top of a gait that "
                          "does not walk). Set TRAIN_EVEN_IF_GAIT_FAILS = True to override.")
                    return

            gaits = {"fwd": gait}
            if OMNI:
                gaits.update(self.omni_stage(gait))
                for name in ("back", "left", "right"):
                    if name in gaits:
                        f, d, t, fell = self.eval_gait(gaits[name], steps=MAX_EP_STEPS, direction=name)
                        print("[omni] %-5s open-loop check: %+.3f m in %.1f s, fell=%s" % (name, d, t, fell))
                        if PROF.get("gate_bonus") and not KEYBOARD_MODE and (fell or d < 0.15):
                            print("[omni] %s gait does not walk, so it is left out of RL training (a shared network that "
                                  "trains on gaits that fall made the forward gait worse in the Atlas runs)" % name)
                            del gaits[name]

            if KEYBOARD_MODE:
                self.drive(gaits)
            elif RUN_TRAINING:
                tdir = TRAIN_DIRECTION if TRAIN_DIRECTION in gaits else "fwd"
                want = [d for d in DIR_NAMES if d in gaits] if SHARED_POLICY else [tdir]
                speeds = {}
                for d in want:
                    if d == "fwd":
                        speeds[d] = speed
                    else:
                        speeds[d] = max(0.0, self.eval_gait(gaits[d], steps=MAX_EP_STEPS, direction=d)[1]) / (MAX_EP_STEPS * self.dt)
                    if speeds[d] <= 0.03:
                        speeds[d] = 0.08
                self.train(gaits, speeds, tdir)
            else:
                self.play(gait, use_policy=True, gaits=gaits)
        except SimEnd:
            print("[end] simulation closed")


if __name__ == "__main__":
    Walker().run()
