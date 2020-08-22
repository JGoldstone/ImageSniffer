from pathlib import Path
import unittest
import numpy as np

from registers import is_black_pixel, is_negative_clip_component, is_zero_component, is_positive_clip_component, \
    strictly_negative_but_not_clipped_inverse_mask, strictly_positive_but_not_clipped_inverse_mask, \
    biggest_strictly_negative_non_clipping_value, \
    tiniest_strictly_negative_non_clipping_value, tiniest_strictly_positive_non_clipping_value, \
    biggest_strictly_positive_non_clipping_value, Counter, Latch
from frame_c18n import FrameC18n

EXR_IMAGE_PATH = '/tmp/green_negative_sprinkle_at_x0174_y_0980.exr'

# imagine if you will LogBins(-4, 2, 7)
TEST_LERP_DOMAIN_LOW = -4
TEST_LERP_DOMAIN_HIGH = 2
TEST_LERP_RANGE_LOW = 0
TEST_LERP_RANGE_HIGH = 6


class MyTestCase(unittest.TestCase):

    def test_black_pixel_finder(self):
        img_array = np.array(np.arange(12)).reshape([2, 2, 3])
        ref_no_blackness = [[False, False],
                            [False, False]]
        blackness = is_black_pixel(img_array)
        self.assertTrue(np.all(ref_no_blackness == blackness))
        img_array[1][1] = [0, 0, 0]
        ref_blackness = [[False, False],
                         [False, True]]
        blackness = is_black_pixel(img_array)
        self.assertTrue(np.all(ref_blackness == blackness))

    def test_negative_clip_component(self):
        fluff = np.array([-2, -1, 0, 1, 2])
        img_array = fluff
        self.assertFalse(np.any(is_negative_clip_component(img_array)))
        neg_clip = np.finfo(np.half).min
        self.assertTrue(np.any(is_negative_clip_component(np.hstack([np.array([neg_clip]), fluff]))))

    def test_is_zero_component(self):
        fluff = np.array([-2, -1, 1, 2])
        img_array = fluff
        self.assertFalse(np.any(is_zero_component(img_array)))
        self.assertTrue(np.any(is_zero_component(np.hstack([np.array([0]), fluff]))))

    def test_positive_clip_component(self):
        fluff = np.array([-2, -1, 0, 1, 2])
        img_array = fluff
        self.assertFalse(np.any(is_positive_clip_component(img_array)))
        pos_clip = np.finfo(np.half).max
        self.assertTrue(np.any(is_positive_clip_component(np.hstack([np.array([pos_clip]), fluff]))))

    @staticmethod
    def create_test_img_array():
        neg_clip = np.finfo(np.half).min
        neg_tiniest = -4 * np.finfo(np.half).tiny
        zero = 0.0
        pos_tiniest = 4 * np.finfo(np.half).tiny
        pos_clip = np.finfo(np.half).max
        img_array = np.array([
            [[zero, pos_clip, zero],
             [neg_clip, neg_tiniest, pos_clip]],
            [[neg_clip, pos_tiniest, pos_clip],
             [zero, zero, zero]],
            [[pos_clip, zero, zero],
             [neg_tiniest, pos_clip, neg_clip]],
            [[pos_tiniest, pos_clip, neg_clip],
             [zero, zero, zero]],
            [[zero, zero, pos_clip],
             [pos_clip, neg_clip, neg_tiniest]],
            [[pos_clip, neg_clip, pos_tiniest],
             [zero, zero, zero]]
        ])
        return img_array

    def test_strictly_negative_but_not_clipped_inverse_mask_creation(self):
        ref_img_array = self.create_test_img_array()
        inv_mask = strictly_negative_but_not_clipped_inverse_mask(ref_img_array)
        ref_inv_mask = np.array([
            [[False, False, False],
             [False, True, False]],
            [[False, False, False],
             [False, False, False]],
            [[False, False, False],
             [True, False, False]],
            [[False, False, False],
             [False, False, False]],
            [[False, False, False],
             [False, False, True]],
            [[False, False, False],
             [False, False, False]]
        ])
        match = np.array(ref_inv_mask == inv_mask)
        self.assertTrue(match.all())

    def test_strictly_positive_but_not_clipped_inverse_mask_creation(self):
        ref_img_array = self.create_test_img_array()
        inv_mask = strictly_positive_but_not_clipped_inverse_mask(ref_img_array)
        ref_inv_mask = np.array([
            [[False, False, False],
             [False, False, False]],
            [[False, True, False],
             [False, False, False]],
            [[False, False, False],
             [False, False, False]],
            [[True, False, False],
             [False, False, False]],
            [[False, False, False],
             [False, False, False]],
            [[False, False, True],
             [False, False, False]]
        ])
        match = np.array(ref_inv_mask == inv_mask)
        self.assertTrue(match.all())

    def test_nonclipping_neg_biggest(self):
        array = np.array([np.finfo(np.half).min, -12, -3, 0, 1.1, 6, np.finfo(np.half).max])
        biggest_neg_non_clipping = biggest_strictly_negative_non_clipping_value(array)
        self.assertEqual(-12, biggest_neg_non_clipping)

    def test_nonclipping_neg_tiniest(self):
        array = np.array([np.finfo(np.half).min, -12, -3, 0, 1.1, 6, np.finfo(np.half).max])
        tiniest_neg_non_clipping = tiniest_strictly_negative_non_clipping_value(array)
        self.assertEqual(-3, tiniest_neg_non_clipping)

    def test_nonclipping_pos_min(self):
        array = np.array([np.finfo(np.half).min, -12, -3, 0, 1.1, 6, np.finfo(np.half).max])
        tiniest_pos_non_clipping = tiniest_strictly_positive_non_clipping_value(array)
        self.assertEqual(1.1, tiniest_pos_non_clipping)

    def test_nonclipping_neg_max(self):
        array = np.array([np.finfo(np.half).min, -12, -3, 0, 1.1, 6, np.finfo(np.half).max])
        biggest_pos_non_clipping = biggest_strictly_positive_non_clipping_value(array)
        self.assertEqual(6, biggest_pos_non_clipping)

    def test_counter_pixel_tally_no_masking(self):
        img_array = np.array(np.arange(12)).reshape([2, 2, 3])
        inv_mask = np.full(img_array.shape[:2], True)
        desc = 'black pixels'
        counter = Counter(desc, is_black_pixel)
        counter.tally_pixels(img_array, inv_mask)
        self.assertEqual(desc, counter.desc)
        self.assertEqual(0, counter.count)
        img_array[1][1] = [0, 0, 0]
        counter = Counter(desc, is_black_pixel)
        counter.tally_pixels(img_array, inv_mask)
        self.assertEqual(1, counter.count)
        actual = counter.summarize(indent_level=0)
        no_indent_summary = counter.summarize(indent_level=0)
        self.assertEqual('black pixels: 1', no_indent_summary.rstrip('\n'))
        single_indent_summary = counter.summarize(indent_level=1)
        self.assertEqual('  black pixels: 1', single_indent_summary.rstrip('\n'))

    def test_counter_channel_tally(self):
        desc = 'negative clip channel values'
        for channel in range(3):
            img = np.array(np.arange(12)).reshape([2, 2, 3])
            inv_mask = np.full(img.shape[:2], True)
            counter = Counter(desc, is_negative_clip_component, channel)
            counter.tally_channel_values(img, inv_mask)
            self.assertEqual(desc, counter.desc)
            self.assertEqual(0, counter.count)
            img[1][1][channel] = np.finfo(np.half).min
            counter = Counter(desc, is_negative_clip_component, channel)
            counter.tally_channel_values(img, inv_mask)
            self.assertEqual(1, counter.count)

    def test_latch(self):
        desc = 'negative clip channel values'
        for channel in range(3):
            img_array = np.array([
                [[-8, 6, -3],
                 [20, 3, 10]],
                [[38, 2, 1],
                 [-12, 4, 1]]])
            inv_mask = np.full(img_array.shape[:2], True)
            biggest_non_clipping_neg_latch = Latch('bigneg', biggest_strictly_negative_non_clipping_value, channel)
            tiniest_non_clipping_neg_latch = Latch('tinyneg', tiniest_strictly_negative_non_clipping_value, channel)
            tiniest_non_clipping_pos_latch = Latch('tinypos', tiniest_strictly_positive_non_clipping_value, channel)
            biggest_non_clipping_pos_latch = Latch('bigpos', biggest_strictly_positive_non_clipping_value, channel)
            biggest_non_clipping_neg_latch.latch_max_channel_value(img_array, inv_mask)
            tiniest_non_clipping_neg_latch.latch_max_channel_value(img_array, inv_mask)
            tiniest_non_clipping_pos_latch.latch_max_channel_value(img_array, inv_mask)
            biggest_non_clipping_pos_latch.latch_max_channel_value(img_array, inv_mask)
            self.assertEqual('bigneg', biggest_non_clipping_neg_latch.desc)
            biggest_neg_ref = [-12, None, -3]
            tiniest_neg_ref = [-8, None, -3]
            tiniest_pos_ref = [20, 2, 1]
            biggest_pos_ref = [38, 6, 10]
            self.assertEqual(biggest_neg_ref[channel], biggest_non_clipping_neg_latch.latched_value)
            self.assertEqual(tiniest_neg_ref[channel], tiniest_non_clipping_neg_latch.latched_value)
            self.assertEqual(tiniest_pos_ref[channel], tiniest_non_clipping_pos_latch.latched_value)
            self.assertEqual(biggest_pos_ref[channel], biggest_non_clipping_pos_latch.latched_value)

    def test_frame_c18n(self):
        images_dir = Path("../images")
        print(f"current working directory is {Path.cwd()}")
        test_image_name = "cg_factory_B091C011_161004_R2XF.645.exr"
        frame_c18n = FrameC18n(images_dir / test_image_name)
        frame_c18n.tally()
        print(frame_c18n)
        print(frame_c18n.summarize())


if __name__ == '__main__':
    unittest.main()
