"""The push-box body: your existing, calibrated DARwIn-OP walker plus a box the supervisor can place and read."""
import math

import darwin_rl as dr
from darwin_rl import Walker

import config


class Pusher(Walker):
    def __init__(self):
        super().__init__()
        self.box = self.getFromDef("BOX")
        if self.box is None:
            raise SystemExit("[!] DEF BOX not found in the world: open worlds/push_box.wbt")
        self.box_t = self.box.getField("translation")
        self.box_r = self.box.getField("rotation")
        goal = self.getFromDef("GOAL_LINE")
        self.goal_t = goal.getField("translation") if goal is not None else None
        self.hold_arms = False
        phys = self.box.getField("physics")
        node = phys.getSFNode() if phys is not None else None
        self.box_mass_field = node.getField("mass") if node is not None else None

    # ---- calibration saved by the earlier walking project ----
    def load_calibration(self):
        """Read darwin_state.json (next to this file) and return the best saved forward gait."""
        st = self._load_state()
        missing = [k for k in ("sg", "z_stand", "gait") if st.get(k) is None]
        if missing:
            raise SystemExit(f"[!] {dr.STATE_PATH} is missing {missing}. Copy the darwin_state.json from your walking "
                             f"project into this controller folder (or run darwin_rl.py once to create it).")
        self.sg, self.z_stand = st["sg"], st["z_stand"]
        self.z_floor = st.get("z_floor")
        self.crouch, self.lean0 = st.get("crouch", dr.CROUCH), st.get("lean0", 0.0)
        self.spawn_z = self.z_stand + 0.012
        self.arms_walk = True
        self.lean0 += config.LEAN_EXTRA
        print(f"[home] the world file had the robot at x={self.home_xy[0]:.3f} y={self.home_xy[1]:.3f}; "
              f"every reset now uses ROBOT_HOME {config.ROBOT_HOME} instead")
        self.home_xy = tuple(config.ROBOT_HOME)
        gait = dict(dr.default_gait(), **st["gait"])
        print(f"[calibration] loaded {dr.STATE_PATH}: signs {self.sg}, standing height {self.z_stand:.3f}, "
              f"crouch {self.crouch:.2f}, gait {dr.gait_id(gait)}")
        return gait

    # ---- arms ----
    def apply_arms(self, spec):
        """spec None keeps the walking arm pose. A dict {"shoulder": (R, L), "upper": (R, L), "elbow": (R, L)} holds
        those angles (rad) for the whole episode: it overrides the walker's arm targets and stops the arm swing."""
        self.hold_arms = bool(spec)
        if not spec:
            return
        swing = dr.PROF["swing"]
        self.arm_pose[swing[0]], self.arm_pose[swing[1]] = spec["shoulder"]
        dr.PROF["walk_upper"] = tuple(spec["upper"])
        elbow = spec["elbow"]
        if elbow == "mid":
            names = dr.PROF["elbow"]
            elbow = tuple(0.5 * (self.motors[n].getMinPosition() + self.motors[n].getMaxPosition()) for n in names)
        dr.PROF["walk_elbow"] = tuple(elbow)

    # ---- the box ----
    def place_box(self, face_x, y=0.0):
        """Put the cube so that its FRONT face is face_x metres ahead of the robot's root; move the goal line too."""
        s = config.BOX_SIZE
        hx, hy = self.home_xy
        self.box_t.setSFVec3f([hx + face_x + s / 2.0, hy + y, s / 2.0 + 0.002])
        self.box_r.setSFRotation([0.0, 0.0, 1.0, 0.0])
        if config.BOX_MASS is not None and self.box_mass_field is not None:
            self.box_mass_field.setSFFloat(float(config.BOX_MASS))
        self.box.resetPhysics()
        if self.goal_t is not None:
            gy = self.goal_t.getSFVec3f()
            self.goal_t.setSFVec3f([hx + face_x + config.GOAL_DX, hy, gy[2]])

    def box_pose(self):
        p = self.box.getPosition()
        R = self.box.getOrientation()
        return {"x": p[0], "y": p[1], "z": p[2], "yaw": math.atan2(R[3], R[0]), "up": R[8]}

    # ---- one episode ----
    def run_episode(self, gait, box_x, steps=None, policy=None, box_y=0.0):
        """Place the box (front face box_x ahead, box_y sideways), reset the robot upright at ROBOT_HOME, walk, score.
        policy.act(pusher, t, phase) -> None or {"turn", "lean", "stride"}: small corrections to the saved gait
        (turn: the walker's steering, lean: ankle offset in rad, stride: multiplier on hip and knee amplitude).
        With config.STOP_AT_GOAL the episode ends the moment the box has moved GOAL_DX (success)."""
        steps = steps or config.EPISODE_STEPS
        if self.hold_arms:
            gait = dict(gait, arm_amp=0.0)
        self.place_box(box_x, box_y)
        self.turn = 0.0
        self.reset_to(self.gait_pose(gait, 0.0, 0.0))
        self.begin_episode()
        if policy is not None:
            policy.reset()
        b0 = self.box_pose()
        fell, t_contact, t_goal, k = False, None, None, 0
        for k in range(steps):
            t = k * self.dt
            ph = (t / gait["period"]) % 1.0
            ramp = min(1.0, t / dr.RAMP_S)
            a = policy.act(self, t, ph) if policy is not None else None
            if a is None:
                q = self.gait_pose(gait, ph, ramp)
            else:
                self.turn = a["turn"]
                g2 = dict(gait, hip_amp=gait["hip_amp"] * a["stride"], knee_amp=gait["knee_amp"] * a["stride"])
                q = self.gait_pose(g2, ph, ramp)
                for side in (0, 1):
                    q[dr.J(dr.ANKLE, side)] += self.sg["ankle"][side] * a["lean"]
            self.command(q)
            self.tick()
            if k >= 3 and self.fell():            # first steps after a reset can still report the old (fallen) pose
                fell = True
                break
            if k % 2 == 0:
                b = self.box_pose()
                moved = math.hypot(b["x"] - b0["x"], b["y"] - b0["y"])
                if t_contact is None and moved > 0.02:
                    t_contact = t
                if config.STOP_AT_GOAL and b["x"] - b0["x"] >= config.GOAL_DX:
                    t_goal = t
                    break
        self.turn = 0.0
        b = self.box_pose()
        robot_fw, robot_lat, robot_yaw = self.progress()
        box_dx, box_dy = b["x"] - b0["x"], b["y"] - b0["y"]
        box_yaw = abs(math.atan2(math.sin(b["yaw"] - b0["yaw"]), math.cos(b["yaw"] - b0["yaw"])))
        tipped = b["up"] < 0.9
        success = (box_dx >= config.GOAL_DX) and not fell and not tipped
        fitness = box_dx - 0.5 * abs(box_dy) - 0.3 * box_yaw - (1.0 if fell else 0.0) - (0.5 if tipped else 0.0)
        # score (used for training and for comparing controllers): finishing fast and straight is best;
        # failures keep a gradient through box progress, approach and staying on the line
        T = steps * self.dt
        if success:
            tg = t_goal if t_goal is not None else T
            score = 1.0 + 0.5 * (1.0 - tg / T) - 0.5 * abs(box_dy) - 0.3 * box_yaw
        else:
            prog = max(0.0, min(1.0, box_dx / config.GOAL_DX))
            appr = max(0.0, min(1.0, robot_fw / max(box_x, 0.1)))
            score = (0.6 * prog + 0.2 * appr - 0.3 * min(1.0, abs(robot_lat)) - 0.5 * abs(box_dy)
                     - (1.0 if fell else 0.0) - (0.5 if tipped else 0.0))
        return {"score": score, "fitness": fitness, "box_dx": box_dx, "box_dy": box_dy, "box_yaw": box_yaw,
                "tipped": tipped, "robot_dx": robot_fw, "robot_dy": robot_lat, "robot_yaw": robot_yaw, "fell": fell,
                "t_contact": t_contact, "t_goal": t_goal, "success": success, "seconds": (k + 1) * self.dt}
