"""Reservoir policy: sensors -> reservoir -> linear readout -> three small gait corrections.

Inputs (12, all scaled to about +-1):
    gyro x,y,z (3), accelerometer x,y,z (3), gait phase sin,cos (2),
    box position in the robot frame: gap to the box's near face and sideways offset (2)   <- from the supervisor
    robot heading relative to its start, sideways offset from its start line (2)         <- from the supervisor
The last four are privileged information (a real robot would need a camera / odometry); stated in the paper.

Outputs (3), each squashed by tanh and scaled:
    turn   -> Walker.turn, the walker's own steering knob (the two legs step with different hip amplitude)
    lean   -> extra ankle angle on both legs (forward / backward lean), rad
    stride -> multiplies hip and knee amplitude (walk faster / slower)
With all readout weights at zero the corrections are 0, 0 and x1.0, which is exactly the open-loop baseline.
"""
import math
import random

import config

N_IN = 12
N_OUT = 3


def clip(v, lim=3.0):
    return max(-lim, min(lim, v))


class ReservoirPolicy:
    def __init__(self, reservoir):
        self.res = reservoir
        self.n_feat = reservoir.n + 1                       # states + constant 1
        self.W = [[0.0] * self.n_feat for _ in range(N_OUT)]
        self.last_y = [0.0] * N_OUT
        self.noise_rng = random.Random(config.NOISE_SEED)

    # ---- readout parameters as one flat list (what the trainer changes) ----
    @property
    def n_params(self):
        return N_OUT * self.n_feat

    def get_params(self):
        return [v for row in self.W for v in row]

    def set_params(self, theta):
        f = self.n_feat
        self.W = [list(theta[k * f:(k + 1) * f]) for k in range(N_OUT)]

    # ---- per episode / per step ----
    def reset(self):
        self.res.reset()
        self.last_y = [0.0] * N_OUT

    def features(self, p, t, ph):
        gx, gy, gz = p.read_gyro()
        ax, ay, az = p.read_acc()
        fw, lat, yaw = p.progress()
        b = p.box_pose()
        me = p.me.getPosition()
        fx, fy = p.fwd
        dx, dy = b["x"] - me[0], b["y"] - me[1]
        gap = dx * fx + dy * fy - config.BOX_SIZE / 2.0
        side = -dx * fy + dy * fx
        u = [gx / 2.0, gy / 2.0, gz / 2.0, ax, ay, az - 1.0,
             math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph),
             gap / 0.5, side / 0.3, yaw, lat / 0.3]
        if not config.PRIVILEGED:
            u[8:] = [0.0, 0.0, 0.0, 0.0]
        if config.SENSE_NOISE > 0:
            u = [v + self.noise_rng.gauss(0.0, config.SENSE_NOISE) for v in u]
        return [clip(v) for v in u]

    def act(self, p, t, ph):
        x = self.res.step(self.features(p, t, ph))
        y = []
        for row in self.W:
            s = row[-1]
            for wi, xi in zip(row, x):
                s += wi * xi
            y.append(math.tanh(s))
        self.last_y = y
        return {"turn": y[0] * config.RES_TURN_MAX,
                "lean": y[1] * config.RES_LEAN_MAX,
                "stride": 1.0 + y[2] * config.RES_STRIDE_MAX}
