import unittest
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import generate_captions, rank_captions, modify_caption

class TestModelFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_image_path = "unit_test/test_image.jpg"
        cls.test_image = Image.new("RGB", (224, 224))  # Create a blank image
        cls.test_image.save(cls.test_image_path)

    def test_generate_captions(self):
        captions = generate_captions(self.test_image)
        self.assertIsInstance(captions, list)
        self.assertEqual(len(captions), 5)
        self.assertTrue(all(isinstance(c, str) for c in captions))

    def test_rank_captions(self):
        dummy_captions = [
            "A dog playing in the park.",
            "A person walking a dog.",
            "A beautiful sunset.",
            "A group of people at the beach.",
            "A car driving on a highway."
        ]
        ranked_captions = rank_captions(self.test_image, dummy_captions)

        self.assertIsInstance(ranked_captions, list)
        self.assertEqual(len(ranked_captions), 5)
        self.assertTrue(all(isinstance(c, str) for c in ranked_captions))

    def test_modify_caption(self):
        # Sample inputs
        selected_caption = "A dog running through the park."
        user_instruction = "It is really a cat!"

        # Modify the caption
        modified_caption = modify_caption(selected_caption, user_instruction)

        # Check if the modified caption contains expected words
        self.assertIsInstance(modified_caption, str)  # Ensure it's a string
        self.assertGreater(len(modified_caption), 0)  # Ensure the modified caption isn't empty

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)

if __name__ == "__main__":
    unittest.main()
