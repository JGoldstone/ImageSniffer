import unittest
from pathlib import Path

from catalog import Catalog


class MyTestCase(unittest.TestCase):

    def test_traversal(self):
        cat = Catalog("/tmp/jgoldstone000_image_catalog.csv")
        cat.register_content(Path("/Volumes/jgoldstone000/cust/shows/the_goldfinch/AFTER_CALIBRATION/MINI"))
        cat.save()


if __name__ == '__main__':
    unittest.main()
