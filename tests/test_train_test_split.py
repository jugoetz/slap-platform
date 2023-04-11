import unittest

import numpy as np

from src.train_test_split import GroupShuffleSplit2D


class TestGroupShuffleSplit2D(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.arange(10000, dtype="int")
        self.groups = np.random.randint(10, size=(10000, 2), dtype="int")
        self.splitter = GroupShuffleSplit2D(
            n_splits=10, test_size=0.001, random_state=42
        )

    def test_test_groups_not_in_train(self):
        for i, (train, test) in enumerate(
            self.splitter.split(self.X, groups=self.groups)
        ):
            test_groups = self.groups[test, :]
            train_groups = self.groups[train, :]
            for test_group in test_groups:
                with self.subTest(test_group=test_group, fold=i):
                    self.assertFalse((train_groups == test_group).all(axis=1).any())

    def test_no_overlap_between_train_and_test(self):
        for i, (train, test) in enumerate(
            self.splitter.split(self.X, groups=self.groups)
        ):
            with self.subTest(fold=i):
                self.assertFalse(set(train).intersection(set(test)))


if __name__ == "__main__":
    unittest.main()
