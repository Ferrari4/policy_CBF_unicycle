import numpy as np

class noise_train_sampler:
    def __init__(self, nd=1, rng=None):
        self.nd = nd
        self.rng = rng if rng is not None else np.random.default_rng()

    def bangbang_uniform_train(self, n_samples=50, n_samples_uniform=25, horizon=10, interval_size=2, scale=1.0):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if interval_size <= 0:
            raise ValueError("interval_size must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.nd <= 0:
            raise ValueError("nd must be positive")
        if not 0 <= n_samples_uniform <= n_samples:
            raise ValueError("n_samples_uniform must be between 0 and n_samples")

        n_intervals = (horizon + interval_size - 1) // interval_size
        uniform_disturbances = self.rng.uniform(low=-scale, high=scale, size=(n_samples_uniform, n_intervals, self.nd),)
        n_bangbang = n_samples - n_samples_uniform
        bangbang_disturbances = self.rng.choice([-scale, scale], size=(n_bangbang, n_intervals, self.nd),)
        interval_disturbances = np.concatenate([uniform_disturbances, bangbang_disturbances], axis=0)
        full_disturbances = np.repeat(interval_disturbances, repeats=interval_size, axis=1,)[:, :horizon, :]

        return full_disturbances, interval_disturbances

    def uniform_train(self, n_samples=50, horizon=10, interval_size=1, scale=1.0):
        n_intervals = (horizon + interval_size - 1) // interval_size
        interval = self.rng.uniform(low=-scale, high=scale, size=(n_samples, n_intervals, self.nd))
        full = np.repeat(interval, interval_size, axis=1)[:, :horizon, :]
        return full, interval

    def zero_train(self, n_samples=1, horizon=10, interval_size=1):
        assert n_samples == 1, "zero_train has only one distinct sequence; n_samples must be 1"
        n_intervals = (horizon + interval_size - 1) // interval_size
        interval = np.zeros((1, n_intervals, self.nd))
        full = np.repeat(interval, interval_size, axis=1)[:, :horizon, :]
        return full, interval

class noise_test_sampler:
    def __init__(self, nd=1, rng=None):
        self.nd = nd
        self._held = None
        self._interval = None
        self._worst = None
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self):
        self._held = None
        self._interval = None

    def zero_test(self, rng, x, k):
        return np.zeros(self.nd)

    def uniform_test(self, rng, x, k, scale=1.0):
        return rng.uniform(-scale, scale, size=(self.nd,))

    def bangbang_test(self, rng, x, k, interval_size=1, scale=1.0):
        interval = k // interval_size
        if interval != self._interval:
            self._held = rng.choice([-scale, scale], size=(self.nd,))
            self._interval = interval
        return self._held

    def set_worst(self, d0):
        self._worst = np.asarray(d0, dtype=float).reshape(self.nd)

    def worst_case(self, rng, x, k):
        if self._worst is None:
            return np.zeros(self.nd)
        return self._worst

if __name__ == "__main__":
    sampler = noise_test_sampler(nd=2)
    uniform_noise = sampler.uniform_test(rng=sampler.rng, x=None, k=0, scale=1.0)
    bangbang_noise = sampler.bangbang_test(rng=sampler.rng, x=None, k=0, interval_size=1, scale=1.0)
    zero_test_noise = sampler.zero_test(rng=sampler.rng, x=None, k=0)
    sampler.set_worst(d0=[-1.0, 3.0])
    worst_noise = sampler.worst_case(rng=sampler.rng, x=None, k=0)
    print("-------------TEST NOISE SAMPLER-------------")
    print("Uniform noise:\n", uniform_noise)
    print("Uniform noise shape:", uniform_noise.shape)
    print("Bang-bang noise:\n", bangbang_noise)
    print("Bang-bang noise shape:", bangbang_noise.shape)
    print("Worst-case noise:\n", worst_noise)
    print("Worst-case noise shape:", worst_noise.shape) 
    print("Zero test noise:\n", zero_test_noise)
    print("Zero test noise shape:", zero_test_noise.shape)
    print("-------------TEST NOISE SAMPLER END-------------")

    sampler_train = noise_train_sampler(nd=2)
    full_disturbances_bbu, interval_disturbances_bbu = sampler_train.bangbang_uniform_train(n_samples=10, n_samples_uniform=5, horizon=6, interval_size=4,scale=1.0)
    full_disturbances_uniform, interval_disturbances_uniform = sampler_train.uniform_train(n_samples=10, horizon=6, interval_size=4,scale=1.0)
    full_disturbances_zero, interval_disturbances_zero = sampler_train.zero_train(horizon=6, interval_size=4)
    print("-------------TRAIN NOISE SAMPLER-------------")  
    print("Full disturbances:\n", full_disturbances_bbu)
    print("Full disturbances shape:", full_disturbances_bbu.shape)
    print("Interval disturbances:\n", interval_disturbances_bbu)
    print("Interval disturbances shape:", interval_disturbances_bbu.shape)
    print("Full disturbances (uniform):\n", full_disturbances_uniform)
    print("Full disturbances (uniform) shape:", full_disturbances_uniform.shape)
    print("Interval disturbances (uniform):\n", interval_disturbances_uniform)      
    print("Interval disturbances (uniform) shape:", interval_disturbances_uniform.shape)
    print("Full disturbances (zero):\n", full_disturbances_zero)
    print("Full disturbances (zero) shape:", full_disturbances_zero.shape)
    print("Interval disturbances (zero):\n", interval_disturbances_zero)
    print("Interval disturbances (zero) shape:", interval_disturbances_zero.shape)
    print("-------------TRAIN NOISE SAMPLER END-------------")
