from sys import float_info
from pathlib import Path
import unittest
import numpy as np
import OpenImageIO as oiio

from registers import Registers, is_black_pixel, strictly_negative_but_not_clipped_masked_array, strictly_positive_but_not_clipped_masked_array
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
        img_array[1][1] = [0, 0, 0]
        ref_blackness = [[ False, False],
                         [False, True]]
        blackness = is_black_pixel(img_array)
        self.assertTrue(np.all(ref_blackness == blackness))


    def create_test_img_array(self):
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
        img_spec = oiio.ImageSpec(img_array_shape[1], img_array_shape[0], img_array_shape[2], oiio.TypeHalf)
        buf = oiio.ImageBuf(img_spec)
        return img_array, img_spec

    def test_strictly_negative_but_not_clipped_masked_array_creation(self):
        ref_img_array, _ = self.create_test_img_array()
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
        ref_img_array, _ = self.create_test_img_array()
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
