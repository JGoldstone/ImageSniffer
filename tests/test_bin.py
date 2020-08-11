import unittest

from sys import float_info
from bins import lerp, LogBins

EXR_IMAGE_PATH = '/tmp/green_negative_sprinkle_at_x0174_y_0980.exr'

# imagine if you will LogBins(-4, 2, 7)
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
        ix_high_bound = lerp(TEST_LERP_DOMAIN_HIGH, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH, TEST_LERP_RANGE_LOW,
                             TEST_LERP_RANGE_HIGH)
        self.assertAlmostEqual(TEST_LERP_RANGE_HIGH, ix_high_bound, delta=float_info.epsilon)

    def test_lerp_extrapolate_above_highest_range(self):
        ix_too_high = lerp(TEST_LERP_DOMAIN_HIGH + 3 * float_info.epsilon, TEST_LERP_DOMAIN_LOW, TEST_LERP_DOMAIN_HIGH,
                           TEST_LERP_RANGE_LOW,
                           TEST_LERP_RANGE_HIGH)
        self.assertGreater(ix_too_high, TEST_LERP_RANGE_HIGH)

    def test_bin_ctor(self):
        test_bin = LogBins(-5, 2, 30)
        self.assertEqual(-5, test_bin.log_max)
        self.assertEqual(2, test_bin.log_min)
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

    # the bins:
    # test_bin = LogBins(2, -4, 8)
    # negate the exposure value and take its base 10 log
    # overflow if l_ev > 2
    # add to _bins[0] if 10e+1 < l_ev <= 10e+2 [+2, +1)
    # add to _bins[1] if 10e+0 < l_ev <= 10e+1 [+1,  0)
    # add to _bins[2] if 10e-1 < l_ev <= 10e+0 [ 0, -1)
    # add to _bins[3] if 10e-2 < l_ev <= 10e-1 [-1, -2)
    # add to _bins[4] if 10e-3 < l_ev <= 10e-1 [-2, -3)
    # add to _bins[5] if 10e-4 < l_ev <= 10e-2 [-3, -4)
    # add to _bins[6] if 10e-5 < l_ev <= 10e-3 [-4, -5)
    # add to _bins[7] if 10e-6 < l_ev <= 10e-4 [-5, -6)
    # underflow if entry <= 10e-6

    def test_binning(self):
        our_epsilon = 0.000000001
        test_bin = LogBins(2, -6, 8)
        test_bin.add_entry(100.0+our_epsilon)
        self.assertEqual(1, test_bin.num_overflowed)
        # 3 for bin 0, 10e2 >= x > 10e1
        test_bin.add_entry(100.0)
        test_bin.add_entry(10.0 + 1*our_epsilon)
        test_bin.add_entry(10.0 + 2*our_epsilon)
        self.assertEqual(3, test_bin._bins[0])
        # 2 for bin 1, 10e1 >= x > 10e0
        test_bin.add_entry(10.0)
        test_bin.add_entry(1.0 + 1*our_epsilon)
        self.assertEqual(2, test_bin._bins[1])
        # 3 for bin 2, 10e0 >= x > 10e-1
        test_bin.add_entry(1.0)
        test_bin.add_entry(0.5)
        test_bin.add_entry(0.1 + 1*our_epsilon)
        self.assertEqual(3, test_bin._bins[2])
        # 4 for bin 3, 10e-1 >= x > 10e-2
        test_bin.add_entry(0.1)
        test_bin.add_entry(0.01 + 3*our_epsilon)
        test_bin.add_entry(0.01 + 2*our_epsilon)
        test_bin.add_entry(0.01 + 1*our_epsilon)
        self.assertEqual(4, test_bin._bins[3])
        # 5 for bin 4, 10e-2 >= x > 10e-3
        test_bin.add_entry(0.01)
        test_bin.add_entry(0.001 + 4*our_epsilon)
        test_bin.add_entry(0.001 + 3*our_epsilon)
        test_bin.add_entry(0.001 + 2*our_epsilon)
        test_bin.add_entry(0.001 + 1*our_epsilon)
        self.assertEqual(5, test_bin._bins[4])
        # 4 for bin 5, 10e-3 >= x > 10e-4
        test_bin.add_entry(0.001)
        test_bin.add_entry(0.0001 + 3*our_epsilon)
        test_bin.add_entry(0.0001 + 2*our_epsilon)
        test_bin.add_entry(0.0001 + 1*our_epsilon)
        self.assertEqual(4, test_bin._bins[5])
        # 3 for bin 6, 10e-4 >= x > 10e-5
        test_bin.add_entry(0.0001)
        test_bin.add_entry(0.00001 + 2*our_epsilon)
        test_bin.add_entry(0.00001 + 1*our_epsilon)
        self.assertEqual(3, test_bin._bins[6])
        # 2 for bin 7, 10e-5 >= x > 10e-6
        test_bin.add_entry(0.00001)
        test_bin.add_entry(0.000001 + 1*our_epsilon)
        self.assertEqual(2, test_bin._bins[7])
        # 3 underflows
        test_bin.add_entry(0.000001)
        test_bin.add_entry(0.000001 - 1*our_epsilon)
        test_bin.add_entry(0.000001 - 2*our_epsilon)
        self.assertEqual(3, test_bin.num_underflowed)


if __name__ == '__main__':
    unittest.main()
