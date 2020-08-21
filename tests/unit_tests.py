from pathlib import Path
import unittest
import numpy as np
import OpenImageIO as oiio

from registers import is_black_pixel, is_negative_clip_component, is_zero_component, is_positive_clip_component, \
    strictly_negative_but_not_clipped_masked_array, \
    strictly_positive_but_not_clipped_masked_array, biggest_strictly_negative_non_clipping_value, \
    tiniest_strictly_negative_non_clipping_value, tiniest_strictly_positive_non_clipping_value, \
    biggest_strictly_positive_non_clipping_value
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
        img_array_shape = img_array.shape
        return img_array

    def test_strictly_negative_but_not_clipped_masked_array_creation(self):
        ref_img_array = self.create_test_img_array()
        masked_array = strictly_negative_but_not_clipped_masked_array(ref_img_array).mask
        ref_array = np.array([
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
        ref_masked_array = np.logical_not(ref_array)
        match = np.array(ref_masked_array == masked_array)
        self.assertTrue(match.all())

    def test_strictly_positive_but_not_clipped_masked_array_creation(self):
        ref_img_array = self.create_test_img_array()
        masked_array = strictly_positive_but_not_clipped_masked_array(ref_img_array).mask
        ref_array = np.array([
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
        ref_masked_array = np.logical_not(ref_array)
        match = np.array(ref_masked_array == masked_array)
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

    # def test_register_ctor(self):
    #     ref_img_array, ref_img_spec = self.create_test_img_array()
    #     reference_neg_clip = np.array([
    #          [[False, False, False],
    #           [False, True, False]],
    #          [[False, False, False],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [True, False, False]],
    #          [[False, False, False],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [False, False, True]],
    #          [[False, False, False],
    #           [False, False, False]]
    #     ])
    #
    #     reference_neg = np.array([
    #          [[False, False, False],
    #           [True, True, False]],
    #          [[True, False, False],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [True, False, True]],
    #          [[False, False, True],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [False, True, True]],
    #          [[False, True, False],
    #           [False, False, False]]
    #          ])
    #
    # #     reference_zero = np.array([
    #          [[True, False, True],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [True, True, True]],
    #          [[False, True, True],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [True, True, True]],
    #          [[True, True, False],
    #           [False, False, False]],
    #          [[False, False, False],
    #           [True, True, True]]
    #     ])
    #
    #     reference_black = np.array([
    #          [False,
    #           False],
    #          [False,
    #           True],
    #          [False,
    #           False],
    #          [False,
    #           True],
    #          [False,
    #           False],
    #          [False,
    #           True]
    #     ])
    #
    #     reference_pos = np.array([
    #          [[False, True, False],
    #           [False, False, True]],
    #          [[False, True, True],
    #           [False, False, False]],
    #          [[True, False, False],
    #           [False, True, False]],
    #          [[True, True, False],
    #           [False, False, False]],
    #          [[False, False, True],
    #           [True, False, False]],
    #          [[True, False, True],
    #           [False, False, False]]
    #     ])
    #
    #     reference_pos_clip = np.array([
    #          [[False, True, False],
    #           [False, False, True]],
    #          [[False, False, True],
    #           [False, False, False]],
    #          [[True, False, False],
    #           [False, True, False]],
    #          [[False, True, False],
    #           [False, False, False]],
    #          [[False, False, True],
    #           [True, False, False]],
    #          [[True, False, False],
    #           [False, False, False]]
    #     ])
    #     ixs = Registers._image_indices(ref_img_array)
    #     registers = Registers(ref_img_spec)
    #     registers.tally(ref_img_array, ixs)
    #     # neg_clips_match = (reference_neg_clip == ixs['neg_clip']).all()
    #     # self.assertTrue(neg_clips_match)
    #     self.assertTrue((reference_neg_clip == ixs['neg_clip']).all())
    #     self.assertTrue((reference_neg      == ixs['neg'     ]).all())
    #     self.assertTrue((reference_zero     == ixs['zero'    ]).all())
    #     self.assertTrue((reference_black    == ixs['black'   ]).all())
    #     self.assertTrue((reference_pos      == ixs['pos'     ]).all())
    #     self.assertTrue((reference_pos_clip == ixs['pos_clip']).all())

    def test_frame_c18n(self):
        images_dir = Path("../images")
        print(f"current working directory is {Path.cwd()}")
        test_image_name = "cg_factory_B091C011_161004_R2XF.645.exr"
        frame_c18n = FrameC18n(images_dir / test_image_name)
        frame_c18n.tally()
        print(frame_c18n)


if __name__ == '__main__':
    unittest.main()
