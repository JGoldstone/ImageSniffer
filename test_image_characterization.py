import unittest

from image_characterization import ImageCharacterization

EXR_IMAGE_PATH = '/tmp/green_negative_sprinkle_at_x0174_y_0980.exr'

class MyTestCase(unittest.TestCase):
    def test_something(self):
        c18n = ImageCharacterization(EXR_IMAGE_PATH)


if __name__ == '__main__':
    unittest.main()
