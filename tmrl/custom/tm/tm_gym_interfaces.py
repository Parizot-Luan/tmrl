# rtgym interfaces for Trackmania

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost

# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts


# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control TrackMania 2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False,
                 save_replay_interval_minutes: float = 0.0,
                 grayscale: bool = True,
                 resize_to=(64, 64)):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            save_replay_interval_minutes: float: if > 0, save a replay at most once per interval, including on reset
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.save_replay_interval_minutes = float(save_replay_interval_minutes) if save_replay_interval_minutes is not None else 0.0
        self.last_replay_save = time.time()
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']
        self.time_penalty = float(cfg.REWARD_CONFIG.get('TIME_PENALTY', 0.0))
        self.collision_speed_drop = float(cfg.REWARD_CONFIG.get('COLLISION_SPEED_DROP', 6.0))
        self.collision_min_speed = float(cfg.REWARD_CONFIG.get('COLLISION_MIN_SPEED', 10.0))
        self.collision_brake_threshold = float(cfg.REWARD_CONFIG.get('COLLISION_BRAKE_THRESHOLD', 0.1))
        self.collision_penalty_per_kmh = float(cfg.REWARD_CONFIG.get('COLLISION_PENALTY_PER_KMH', 0.1))
        self.collision_penalty_max = float(cfg.REWARD_CONFIG.get('COLLISION_PENALTY_MAX', 3.0))
        self.collision_cooldown_steps = int(cfg.REWARD_CONFIG.get('COLLISION_COOLDOWN_STEPS', 10))
        self.collision_warmup_steps = int(cfg.REWARD_CONFIG.get('COLLISION_WARMUP_STEPS', 10))
        self.deviation_penalty = float(cfg.REWARD_CONFIG.get('DEVIATION_PENALTY', 0.0))
        self.deviation_margin = float(cfg.REWARD_CONFIG.get('DEVIATION_MARGIN', 5.0))
        self._last_action = None
        self._prev_speed = None
        self._shaping_steps = 0
        self._collision_cooldown = 0

        self.initialized = False

    def initialize_common(self):
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        self._last_action = None if control is None else np.asarray(control, dtype=np.float32).reshape(-1)
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def _build_info(self, data, end_of_track):
        info = {"end_of_track": bool(end_of_track)}
        if len(data) > 11:
            info["race_time_s"] = float(data[11]) / 1000.0
        return info

    def reset_race(self):
        self._prev_speed = None
        self._last_action = None
        self._shaping_steps = 0
        self._collision_cooldown = 0
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def _compute_reward_shaping(self, data):
        info = {
            "time_penalty": 0.0,
            "collision_penalty": 0.0,
            "collision": False,
            "speed_drop": 0.0,
            "trajectory_dist": 0.0,
            "deviation_penalty": 0.0,
        }
        penalty = 0.0

        time_penalty = self.time_penalty
        penalty += time_penalty
        info["time_penalty"] = time_penalty

        if self.reward_function is not None:
            trajectory_dist = float(getattr(self.reward_function, "last_min_dist", 0.0))
            info["trajectory_dist"] = trajectory_dist
            if self.deviation_penalty > 0.0:
                deviation_penalty = -self.deviation_penalty * max(0.0, trajectory_dist - self.deviation_margin)
                penalty += deviation_penalty
                info["deviation_penalty"] = deviation_penalty

        speed = float(data[0])
        if (self._prev_speed is not None
                and self._last_action is not None
                and self._collision_cooldown <= 0
                and self._shaping_steps >= self.collision_warmup_steps):
            speed_drop = self._prev_speed - speed
            brake = float(self._last_action[1]) if self._last_action.size > 1 else 0.0
            if (speed_drop > self.collision_speed_drop
                    and brake <= self.collision_brake_threshold
                    and self._prev_speed > self.collision_min_speed):
                collision_penalty = -min(
                    self.collision_penalty_max,
                    self.collision_penalty_per_kmh * speed_drop)
                penalty += collision_penalty
                self._collision_cooldown = self.collision_cooldown_steps
                info["collision"] = True
                info["speed_drop"] = speed_drop
                info["collision_penalty"] = collision_penalty

        if self._collision_cooldown > 0:
            self._collision_cooldown -= 1
        self._prev_speed = speed
        self._shaping_steps += 1
        return penalty, info

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        if self.save_replays and self.save_replay_interval_minutes > 0.0:
            now = time.time()
            if now - self.last_replay_save >= self.save_replay_interval_minutes * 60.0:
                try:
                    save_ghost()
                    self.last_replay_save = now
                    time.sleep(1.0)
                except Exception as e:
                    logging.warning(f"Could not save replay: {e}")
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {"race_time_s": 0.0, "end_of_track": False}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        if self.save_replays and self.save_replay_interval_minutes <= 0.0:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        shaping_penalty, shaping_info = self._compute_reward_shaping(data)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = self._build_info(data, end_of_track)
        info.update(shaping_info)
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty + shaping_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceImagesProgress(TM2020Interface):
    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        progress = np.array([0.0], dtype='float32')
        obs = [speed, gear, rpm, imgs, progress]
        self.reward_function.reset()
        return obs, {"race_time_s": 0.0, "end_of_track": False}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        shaping_penalty, shaping_info = self._compute_reward_shaping(data)
        progress = np.array([self.reward_function.cur_idx / max(1, self.reward_function.datalen)], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs, progress]
        end_of_track = bool(data[8])
        info = self._build_info(data, end_of_track)
        info.update(shaping_info)
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty + shaping_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1, ))
        return spaces.Tuple((speed, gear, rpm, img, progress))


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False, save_replay_interval_minutes: float = 0.0):
        super().__init__(img_hist_len=img_hist_len, gamepad=gamepad, save_replays=save_replays, save_replay_interval_minutes=save_replay_interval_minutes)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {"race_time_s": 0.0, "end_of_track": False}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = self._build_info(data, end_of_track)
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {"race_time_s": 0.0, "end_of_track": False}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = self._build_info(data, end_of_track)
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))


if __name__ == "__main__":
    pass
