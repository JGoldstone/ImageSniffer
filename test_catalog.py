import unittest
# import uuid
# import pandas as pd
# from pathlib import Path

from catalog import Catalog

class MyTestCase(unittest.TestCase):

    def test_traversal(self):
        cat = Catalog("/tmp/jgoldstone000_image_catalog.csv")
        cat.register_content("/Volumes/jgoldstone000")
        cat.save()

if __name__ == '__main__':
    unittest.main()
