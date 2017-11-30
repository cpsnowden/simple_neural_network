import unittest

class EmptyTest(unittest.TestCase):


    def test_nothing(self):
        raise Exception()


if __name__ == "__main__":
    unittest.main()