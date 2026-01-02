import numpy as np
from numba import njit, prange
from simulators import Model
from utils.simulator_utils import as_1d

@njit
def sample_lba_trial(
    v: np.ndarray,
    A: float,
    b: float,
    tau: float,
    s: float = 1.0,
    max_time: int = 10,
) -> np.ndarray:

    n_choices = v.shape[0]
    threshold = A + b
    rt = -1
    choice = -1

    while True:
        min_t = 1e25
        winner = -1
        for c in range(n_choices):
            # 1. Sample Start Point
            start_point = np.random.random() * A
            # 2. Sample Drift Rate
            drift_rate = np.random.normal(v[c], s)
            # 3. Calculate Time
            if drift_rate > 0:
                t_accum = (threshold - start_point) / drift_rate
                if t_accum < min_t:
                    min_t = t_accum
                    winner = c
            # We only record and break if we found a winner (winner != -1)
        if winner != -1:
            break 
    if min_t < max_time:
        return np.array([min_t+tau, winner], dtype=np.float32)
    else:
        return np.array([-1.0, -1.0], dtype=np.float32)

@njit(parallel=True)
def simulate_lba(
    v: np.ndarray,
    A: np.ndarray,
    tau: np.ndarray,
    b: np.ndarray,
    sigma: float = 1.0,
    max_time: int = 10
):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = sample_lba_trial(
            v=v[i],
            A=A[i],
            tau=tau[i],
            b=b[i],
            s=sigma,
            max_time=max_time
        )
        sim_data[i] = sim_trial

    return sim_data


def sample_lba_prior():
    return {
        "v":        {"intercept": np.random.gamma(2.5, 0.5),
                     "slope": 0.0},
        "v_diff":    {"intercept": np.random.lognormal(0, 0.5),
                     "slope":  0.0},
        "tau":      {"intercept": np.random.gamma(2.0, 0.2),
                     "slope": 0.0},
        "b":      {"intercept": np.random.gamma(0.5, 0.2),
                     "slope": 0.0},
        "A":    {"intercept": np.random.gamma(1.0, 0.4),
                     "slope": 0.0}
    }

class LBA(Model):

    def __init__(self, max_time: int = 10):
        self.max_time = max_time

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build per-trial K-way drift from scalars; no dtype control here.
        Expects in `params`: v, optional v_diff, b, tau, A (len == num_obs or scalars).
        Expects in `context`: 'correct_idx' (len == num_obs), optional 'num_alternatives'.
        """
        context = context or {}

        if "correct_idx" not in context:
            raise ValueError("LBA requires context['correct_idx'].")
        correct_idx = np.asarray(context["correct_idx"]).reshape(-1)
        if correct_idx.shape[0] != num_obs:
            raise ValueError(f"correct_idx length {correct_idx.shape[0]} != num_obs {num_obs}")

        num_alternatives = int(context.get("num_alternatives", int(correct_idx.max()) + 1))
        if num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1.")

        v_base = as_1d(params["v"]["intercept"], num_obs)
        v_diff = as_1d(params["v_diff"]["intercept"], num_obs)
        A      = as_1d(params["A"]["intercept"], num_obs)
        tau    = as_1d(params["tau"]["intercept"], num_obs)
        b  = as_1d(params["b"]["intercept"], num_obs)


        # Build per-trial K-vector drift
        v_correct   = v_base + 0.5 * v_diff
        v_incorrect = v_base - 0.5 * v_diff
        v = np.full((num_obs, num_alternatives), 0.0)
        for i in range(num_obs):
            v[i, :] = v_incorrect[i]
            v[i, correct_idx[i]] = v_correct[i]

        return {"v": v, "A": A, "tau": tau, "b": b}

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int) -> dict[str, np.ndarray]:
        correct_idx = np.random.randint(0, num_alternatives, size=num_obs)
        return {"correct_idx": correct_idx, "num_alternatives": num_alternatives}

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        num_alternatives = np.random.randint(2, 5)  # Randomly pick 2-4 alternatives per batch
        return LBA.build_context(num_obs, num_alternatives)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        results = simulate_lba(**params, max_time=self.max_time)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        return {"rts": rts, "choices": choices}
    
    def sample(self, batch_size: int | tuple, num_obs: int = 500, num_alternatives: int = 3, context=None):

        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]
            
        if context == None:
            context = self.build_context(num_obs, num_alternatives)

        parameters = []
        rts = []
        choices = []

        for i in range(batch_size):
            prior_draw = sample_lba_prior()
            prepared_params = self.prepare_params(params=prior_draw, num_obs=num_obs, context=context)
            results = self.simulate(params=prepared_params, context=context)
            rts.append(results["rts"])
            choices.append(results["choices"])
            parameters.append(prior_draw)

        prior_draws = {}

        for cycle_dict in parameters:
            for key, subdict in cycle_dict.items():
                if key not in prior_draws:
                    prior_draws[key] = {}
                for subkey, value in subdict.items():
                    prior_draws[key].setdefault(subkey, []).append(value)
        
        sim_data = {"rts": np.array(rts), "choices": np.array(choices)}
        return prior_draws | sim_data
