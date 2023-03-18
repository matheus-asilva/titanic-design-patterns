from sklearn.model_selection import train_test_split


class Sampler:
    def __init__(self, data, target, test_size=0.2, random_state=2023):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        return train_test_split(
            self.data,
            self.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
