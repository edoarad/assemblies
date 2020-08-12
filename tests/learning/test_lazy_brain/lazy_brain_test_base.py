import unittest

from tests.learning.brain_test_utils import BrainTestUtils


class LazyBrainTestBase(unittest.TestCase):

    def setUp(self) -> None:
        self.utils = BrainTestUtils(lazy=True)
