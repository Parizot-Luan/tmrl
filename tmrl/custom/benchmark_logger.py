"""
benchmark_logger.py  ->  à copier dans  tmrl/custom/benchmark_logger.py

Enregistre dans TmrlData/benchmark/ deux fichiers CSV (en ajout, jamais écrasés) :

  steps.csv     : une ligne par pas de temps (20 Hz)
                  run, phase, session, episode, step, t, race_time_s, speed, progress, gas,
                  brake, steer, reward, terminated, truncated
  episodes.csv  : une ligne par épisode (résumé)
                  run, phase, session, episode, steps, duration_s, total_reward, max_progress,
                  final_progress, mean_speed, max_speed, mean_abs_steer, terminated, truncated,
                  finished, lap_time_s, time_source

Notes
  * Action TMRL = [gas, brake, steer] dans [-1, 1]  (3 composantes, steer = la 3e !).
  * progress est dans [0, 1] (fraction du circuit).
  * "finished" est une heuristique : épisode terminé (terminated) avec progress final >= finish_progress.
  * Un épisode qui n'est pas "terminated" est clôturé au reset suivant avec truncated=1 (limite de temps ou interruption).
  * "run" identifie le réseau (ex. RUN_NAME), "phase" distingue entraînement ("train") et évaluation ("eval"),
    "session" (horodatage de lancement) évite les collisions de numéros d'épisode entre deux lancements.
  * Les fichiers restent ouverts et sont vidés (flush) à chaque fin d'épisode : pas d'ouverture/fermeture
    à chaque pas (trop lent pour du temps réel à 20 Hz). Ne les ouvre pas dans Excel pendant l'entraînement
    (copie-les d'abord). Toute erreur d'écriture désactive le logger SANS faire planter l'environnement.
"""

import csv
import os
import time

import numpy as np

STEP_COLUMNS = ["run", "phase", "session", "episode", "step", "t", "race_time_s", "speed", "progress",
                "gas", "brake", "steer", "reward", "terminated", "truncated",
                "time_penalty", "collision_penalty", "collision", "speed_drop",
                "trajectory_dist", "deviation_penalty"]
EPISODE_COLUMNS = ["run", "phase", "session", "episode", "steps", "duration_s", "total_reward",
                   "max_progress", "final_progress", "mean_speed", "max_speed", "mean_abs_steer",
                   "terminated", "truncated", "finished", "lap_time_s", "time_source",
                   "total_time_penalty", "total_collision_penalty", "collisions", "mean_trajectory_dist"]


def _scalar(x):
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


class EpisodeRecorder:
    def __init__(self, folder, run_name, phase="train", time_step=0.05, finish_progress=0.97,
                 flush_every=200, progress_index=1):
        self.folder = str(folder)
        self.run_name = str(run_name)
        self.phase = str(phase)
        self.time_step = float(time_step)
        self.finish_progress = float(finish_progress)
        self.flush_every = int(flush_every)
        self.progress_index = int(progress_index)
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.episode = 0
        self._steps_f = self._steps_w = self._eps_f = self._eps_w = None
        self._disabled = False
        self._since_flush = 0
        self._reset_stats()

    # ------------------------------------------------------------------ internes
    def _reset_stats(self):
        self.n = 0
        self.total_reward = 0.0
        self.max_progress = 0.0
        self.last_progress = 0.0
        self.sum_speed = 0.0
        self.max_speed = 0.0
        self.sum_abs_steer = 0.0
        self.last_race_time = None
        self.last_end_of_track = False
        self.last_time_source = ""
        self.sum_time_penalty = 0.0
        self.sum_collision_penalty = 0.0
        self.collisions = 0
        self.sum_trajectory_dist = 0.0

    @staticmethod
    def _open_csv(path, columns):
        is_new = (not os.path.exists(path)) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if is_new:
            w.writerow(columns)
            f.flush()
        return f, w

    def _open(self):
        os.makedirs(self.folder, exist_ok=True)
        self._steps_f, self._steps_w = self._open_csv(os.path.join(self.folder, "steps.csv"), STEP_COLUMNS)
        self._eps_f, self._eps_w = self._open_csv(os.path.join(self.folder, "episodes.csv"), EPISODE_COLUMNS)
        print(f"[BenchmarkLogger] run='{self.run_name}' phase='{self.phase}' -> {self.folder}", flush=True)

    def _flush(self):
        self._steps_f.flush()
        self._eps_f.flush()
        self._since_flush = 0

    def _close_episode(self, terminated, truncated):
        if self.n == 0:
            return
        finished = bool(terminated) and (self.last_end_of_track or self.last_progress >= self.finish_progress)
        duration = self.n * self.time_step
        lap_time = None
        time_source = ""
        if finished:
            if self.last_end_of_track and self.last_race_time is not None and self.last_race_time > 0.0:
                lap_time = self.last_race_time
                time_source = "race"
            else:
                lap_time = duration
                time_source = "steps"
        self._eps_w.writerow([
            self.run_name, self.phase, self.session, self.episode, self.n, f"{duration:.3f}",
            f"{self.total_reward:.4f}", f"{self.max_progress:.5f}", f"{self.last_progress:.5f}",
            f"{self.sum_speed / self.n:.3f}", f"{self.max_speed:.3f}", f"{self.sum_abs_steer / self.n:.4f}",
            int(bool(terminated)), int(bool(truncated)), int(finished),
            f"{lap_time:.3f}" if lap_time is not None else "", time_source,
            f"{self.sum_time_penalty:.4f}", f"{self.sum_collision_penalty:.4f}",
            self.collisions, f"{self.sum_trajectory_dist / self.n:.3f}"])
        self._flush()
        self._reset_stats()

    # ------------------------------------------------------------------ API publique
    def on_reset(self):
        """À appeler au début de chaque épisode (clôt proprement un épisode interrompu)."""
        if self._disabled:
            return
        try:
            if self._eps_w is not None:
                # épisode non terminé, clôturé par le reset suivant : limite de temps (tronqué) ou interrompu
                self._close_episode(terminated=False, truncated=True)
        except Exception as e:  # ne jamais casser la boucle temps réel
            self._disable(e)
        self.episode += 1
        self._reset_stats()

    def on_step(self, obs, action, reward, terminated, truncated, info=None):
        if self._disabled:
            return
        try:
            if self._steps_f is None:
                self._open()
            speed = _scalar(obs[0])
            progress = _scalar(obs[self.progress_index])
            race_time = None
            end_of_track = False
            time_penalty = 0.0
            collision_penalty = 0.0
            collision = False
            speed_drop = 0.0
            trajectory_dist = 0.0
            deviation_penalty = 0.0
            if isinstance(info, dict):
                race_time = info.get("race_time_s")
                end_of_track = bool(info.get("end_of_track", False))
                time_penalty = float(info.get("time_penalty", 0.0))
                collision_penalty = float(info.get("collision_penalty", 0.0))
                collision = bool(info.get("collision", False))
                speed_drop = float(info.get("speed_drop", 0.0))
                trajectory_dist = float(info.get("trajectory_dist", 0.0))
                deviation_penalty = float(info.get("deviation_penalty", 0.0))
            if race_time is not None:
                self.last_race_time = float(race_time)
            if end_of_track:
                self.last_end_of_track = True
            a = np.full(3, np.nan) if action is None else np.asarray(action, dtype=np.float64).reshape(-1)
            gas, brake, steer = (list(a[:3]) + [float("nan")] * 3)[:3]
            reward = float(reward)
            self.n += 1
            self.total_reward += reward
            self.sum_time_penalty += time_penalty
            self.sum_collision_penalty += collision_penalty
            self.collisions += int(collision)
            self.sum_trajectory_dist += trajectory_dist
            self.max_progress = max(self.max_progress, progress)
            self.last_progress = progress
            self.sum_speed += speed
            self.max_speed = max(self.max_speed, speed)
            self.sum_abs_steer += abs(steer) if not np.isnan(steer) else 0.0
            self._steps_w.writerow([
                self.run_name, self.phase, self.session, self.episode, self.n, f"{self.n * self.time_step:.2f}",
                f"{race_time:.3f}" if race_time is not None else "",
                f"{speed:.3f}", f"{progress:.5f}", f"{gas:.4f}", f"{brake:.4f}", f"{steer:.4f}",
                f"{reward:.4f}", int(bool(terminated)), int(bool(truncated)),
                f"{time_penalty:.4f}", f"{collision_penalty:.4f}", int(collision),
                f"{speed_drop:.3f}", f"{trajectory_dist:.3f}", f"{deviation_penalty:.4f}"])
            self._since_flush += 1
            if terminated or truncated:
                self._close_episode(terminated, truncated)
            elif self._since_flush >= self.flush_every:
                self._flush()
        except Exception as e:
            self._disable(e)

    def _disable(self, err):
        self._disabled = True
        print(f"[BenchmarkLogger] ERREUR, logger désactivé (l'entraînement continue) : {err!r}", flush=True)

    def close(self):
        try:
            if self._eps_w is not None and not self._disabled:
                self._close_episode(terminated=False, truncated=True)
        except Exception:
            pass
        for f in (self._steps_f, self._eps_f):
            try:
                if f is not None:
                    f.flush()
                    f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
        