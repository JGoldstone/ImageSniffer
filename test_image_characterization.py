import unittest

from sys import float_info
from image_characterization import lerp, LogBin, ImageCharacterization

EXR_IMAGE_PATH = '/tmp/green_negative_sprinkle_at_x0174_y_0980.exr'

# imagine if you will LogBin(-4, 2, 7)
TEST_LERP_DOMAIN_LOW = -4
TEST_LERP_DOMAIN_HIGH = 2
TEST_LERP_RANGE_LOW = 0
TEST_LERP_RANGE_HIGH = 6


class MyTestCase(unittest.TestCase):

    # Interestingly for this particular set of domain and range values, one needs to be more than 3X epsilon out of
    # domain to have the lerp go outside of range
    def test_lerp_extrapolate_below_lowest_range(self):
        ix_too_low = lerp(TEST_LERP_DOMAIN_LOW - 3*float_info.epsilon, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH,
                          TEST_LERP_RANGE_LOW, TEST_LERP_RANGE_HIGH)
        self.assertLess(ix_too_low, TEST_LERP_RANGE_LOW)

    def test_lerp_low_bound(self):
        ix_low_bound = lerp(TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH, TEST_LERP_RANGE_LOW,
                            TEST_LERP_RANGE_HIGH)
        self.assertAlmostEqual(TEST_LERP_RANGE_LOW, ix_low_bound, delta=2 * float_info.epsilon)

    def test_lerp_high_bound(self):
        ix_high_bound = lerp(TEST_LERP_DOMAIN_HIGH, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH, TEST_LERP_RANGE_LOW, TEST_LERP_RANGE_HIGH)
        self.assertAlmostEqual(TEST_LERP_RANGE_HIGH, ix_high_bound, delta=float_info.epsilon)

    def test_lerp_extrapolate_above_highest_range(self):
        ix_too_high = lerp(TEST_LERP_DOMAIN_HIGH + 3*float_info.epsilon, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH, TEST_LERP_RANGE_LOW,
                           TEST_LERP_RANGE_HIGH)
        self.assertGreater(ix_too_high, TEST_LERP_RANGE_HIGH)

    def test_bin_ctor(self):
        test_bin = LogBin(-5, 2, 30)
        self.assertEqual(-5, test_bin.min)
        self.assertEqual(2, test_bin.max)
        self.assertEqual(0, test_bin.num_overflowed)
        self.assertEqual(0, test_bin.num_underflowed)
        # n.b. assertLess suceeds for x+2*e < x+4*e, fails for x+3*e < x+4*e
        smaller = float_info.epsilon * 2
        same = float_info.epsilon * 4
        # n.b. assertLess suceeds for x+6*e > x+4*e, fails for x+5*e > x+4*e
        # moral: when nudging, always nudge at least 2*float_info.epsilon
        bigger = float_info.epsilon * 6
        self.assertLess(4 + smaller, 4 + test_bin._epsilon)
        self.assertEqual(4 + same, 4 + test_bin._epsilon)
        self.assertGreater(4 + bigger, 4 + test_bin._epsilon)
        self.assertEqual(30, len(test_bin._bins))

    # binning theory (for xxxBin(-4,2,7)
    # the bins:
    # underflow if entry <= 10e-5
    # add to _bins[0] if 10e-5 < entry <= 10e-4
    # add to _bins[1] if 10e-4 < entry <= 10e-3
    # add to _bins[2] if 10e-3 < entry <= 10e-2
    # add to _bins[3] if 10e-2 < entry <= 10e-1
    # add to _bins[4] if 10e-1 < entry <= 0
    # add to _bins[5] if 0 < entry <= 1
    # add to _bins[6] if 1 < entry <= 2
    # overflow if entry > 2

    def test_binning_underflow(self):
        test_bin = LogBin(-4, 2, 7)
        test_bin.add_entry(1e-5)
        # make sure this doesn't get counted as underflow
        test_bin.add_entry(1e-5 + 2 * float_info.epsilon)
        self.assertEqual(2, test_bin.num_underflowed)

    # def test_binning_overflow(self):
    #     test_bin = LogBin(-4, 2, 7)
    #     test_bin.add_entry(99)
    #     test_bin.add_entry(100 - 2 * float_info.epsilon)
    #     test_bin.add_entry(100)
    #     test_bin.add_entry(100 + 2 * float_info.epsilon)
    #     test_bin.add_entry(101)
    #     self.assertEqual(2, test_bin.num_overflowed)

    # def test_something(self):
    #     c18n = ImageCharacterization(EXR_IMAGE_PATH)


if __name__ == '__main__':
    unittest.main()
