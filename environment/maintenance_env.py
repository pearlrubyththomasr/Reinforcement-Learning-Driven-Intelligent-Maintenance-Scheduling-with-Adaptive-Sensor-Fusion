
import gym
from gym import spaces
import numpy as np

class MaintenanceEnv(gym.Env):
    """
    Multi-sensor predictive maintenance environment.
    Uses precomputed true degradation from data file.
    State: [vibration_rms_norm, temp_mean_norm, acoustic_peak_norm, degradation]
    Action: 0 = operate, 1 = maintenance
    Reward: based on operating cost, maintenance cost, and failure penalty.
    """
    def __init__(self, data_path, window_size=256, failure_threshold=0.8,
                 op_cost=1.0, maint_cost=20.0, failure_cost=100.0):
        super(MaintenanceEnv, self).__init__()
        # Load multi-sensor data including true degradation
        data = np.load(data_path)
        self.vib = data['vibration']
        self.temp = data['temperature']
        self.aco = data['acoustic']
        self.true_degradation = data['degradation']   # precomputed true health
        self.num_samples = len(self.vib)

        self.window_size = window_size
        self.failure_threshold = failure_threshold
        self.op_cost = op_cost
        self.maint_cost = maint_cost
        self.failure_cost = failure_cost

        self.current_step = window_size
        self.maintenance_performed = False

        # Precompute feature normalization bounds (from entire dataset)
        # For faster init, use a subset
        sample_win = min(window_size, 10000)
        vib_rms_vals = []
        temp_means = []
        aco_peaks = []
        for i in range(sample_win, min(100000, self.num_samples)):
            win_start = i - sample_win
            vib_win = self.vib[win_start:i]
            temp_win = self.temp[win_start:i]
            aco_win = self.aco[win_start:i]
            vib_rms_vals.append(np.sqrt(np.mean(vib_win**2)))
            temp_means.append(np.mean(temp_win))
            aco_peaks.append(np.max(aco_win))
        self.vib_max = max(vib_rms_vals) if vib_rms_vals else 1.0
        self.temp_min = min(temp_means) if temp_means else 20
        self.temp_max = max(temp_means) if temp_means else 60
        self.aco_max = max(aco_peaks) if aco_peaks else 0.5

        # Action space: 0=operate, 1=maintain
        self.action_space = spaces.Discrete(2)
        # Observation: [norm_vib_rms, norm_temp_mean, norm_aco_peak, degradation]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.reset()

    def _extract_features(self):
        start = self.current_step - self.window_size
        vib_win = self.vib[start:self.current_step]
        temp_win = self.temp[start:self.current_step]
        aco_win = self.aco[start:self.current_step]

        vib_rms = np.sqrt(np.mean(vib_win**2))
        temp_mean = np.mean(temp_win)
        aco_peak = np.max(aco_win)

        vib_norm = np.clip(vib_rms / self.vib_max, 0, 1)
        temp_norm = np.clip((temp_mean - self.temp_min) / (self.temp_max - self.temp_min), 0, 1)
        aco_norm = np.clip(aco_peak / self.aco_max, 0, 1)

        return np.array([vib_norm, temp_norm, aco_norm], dtype=np.float32)

    def step(self, action):
        cost = 0.0
        done = False

        # Current true degradation
        current_deg = self.true_degradation[self.current_step]

        if action == 1:  # maintenance
            cost -= self.maint_cost
            # Reduce degradation temporarily (simulate repair)
            current_deg = max(0, current_deg - 0.5)
            self.maintenance_performed = True
        else:  # operate
            cost -= self.op_cost

        # Failure if degradation exceeds threshold and no recent maintenance
        if current_deg >= self.failure_threshold and not self.maintenance_performed:
            cost -= self.failure_cost
            done = True

        # Move time step
        self.current_step += 1
        if self.current_step >= self.num_samples - self.window_size:
            done = True

        # Next observation
        features = self._extract_features()
        next_degradation = self.true_degradation[self.current_step] if self.current_step < self.num_samples else 1.0
        obs = np.append(features, next_degradation).astype(np.float32)

        info = {"degradation": current_deg, "action_taken": action}
        return obs, cost, done, info

    def reset(self):
        self.current_step = self.window_size
        self.maintenance_performed = False
        features = self._extract_features()
        init_degradation = self.true_degradation[self.current_step]
        obs = np.append(features, init_degradation).astype(np.float32)
        return obs

    def render(self, mode='human'):
        pass
