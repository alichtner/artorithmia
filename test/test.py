# A set of scripts that will test my classes and functions
import unittest

from src.Art_class import Art

class TestArt(unittest.TestCase):
    def test_create_Art_class(self):
        """
        Build the art class
        Load an image
        Load metadata
        """
        art = Art()
        art.load_image('images/bird.jpg')
        self.assertEqual(len(art.hue_bins), 50)
        self.assertEqual(art.short_name, 'bird')

    def test_create_ClusterArt_class(self):
        """
        Build a collection of Art objects
        """
        pass

    def test_get_art_info(self):
        """
        Returns report of art characteristics
        """
        pass

    def test_get_cluster_center_images(self):
        pass

    def test_get_similar_images(self):
        pass

    def test_get_user_recommendations(self):
        pass

    def test_art_from_taste_space(self):
        """
        Return art using a 'taste' vector
        INPUT: Taste Vector
        OUTPUT: Indices of artwork

        ex. taste_vec = np.array([0.0, 1.3, 0.5])
            where items represent [redness, aspect_ratio, blurriness]
        """

if __name__=='__main__':
    test_create_Art_class()
