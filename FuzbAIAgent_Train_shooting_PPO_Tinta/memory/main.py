import numpy as np
import torch

class PPOBuffer:
    """
    A simple buffer to store trajectories for PPO.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # večja lambda pomeni večjo varianco in bolj dolgotrajen trening
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)      # Observations
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)      # actions 
        self.adv_buf = np.zeros(size, dtype=np.float32)                 # Advantages
        self.rew_buf = np.zeros(size, dtype=np.float32)                 # Rewards 
        self.ret_buf = np.zeros(size, dtype=np.float32)                 # Returns
        self.val_buf = np.zeros(size, dtype=np.float32)                 # Value estimates
        self.logp_buf = np.zeros(size, dtype=np.float32)                # log probs

        self.gamma = gamma  # Discount
        self.lam = lam      # GAE
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Store one step of interaction."""
        if self.ptr >= self.max_size:
            # Signal that buffer is full - training should happen
            print(f"[PPOBuffer] Buffer full at {self.ptr} steps")
            return False  # Indicate buffer is full
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        return True  # Indicate successful storage

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory (episode).
        Computes advantage/returns for the path.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # Compute GAE-Lambda advantage
        adv = 0
        for i in reversed(range(len(rews) - 1)):
            delta = rews[i] + self.gamma * vals[i+1] - vals[i]
            adv = delta + self.gamma * self.lam * adv
            self.adv_buf[path_slice][i] = adv

        # Compute returns
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        """
            Gets the collected data from the buffer into tensor for training, normalizes the advantages
            and resets the buffer pointers.
            Call this at the end of an epoch to get all of the data from the buffer, with advantages normalized
            to mean 0 and std 1.
            Also resets some pointers in the buffer.
        """
        
        actual_size = self.ptr
        if actual_size == 0:
            return None
        
        indices = np.arange(actual_size)

        # Normalize the advantages
        if actual_size > 1:
            adv_mean = np.mean(self.adv_buf[indices])
            adv_std  = np.std(self.adv_buf[indices])
            if adv_std > 1e-8:
                self.adv_buf[indices] = (self.adv_buf[indices] - adv_mean) / (adv_std + 1e-8)


        data = dict(obs=self.obs_buf[indices],
                    act=self.act_buf[indices],
                    ret=self.ret_buf[indices],
                    adv=self.adv_buf[indices],
                    logp=self.logp_buf[indices]
                    )

        # Resetira pointer na začetek (pobriše buffer)
        self.ptr, self.path_start_idx = 0, 0  
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

