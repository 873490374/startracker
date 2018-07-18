from program.planar_triangle import PlanarTriangleImage
from program.star import StarUV
from program.tracker.image_processor import ImageProcessor
from program.tracker.orientation_finder import OrientationFinder
from program.tracker.star_identifier import StarIdentifier


class StarTracker:

    def __init__(
            self,
            image_processor: ImageProcessor,
            star_identifier: StarIdentifier,
            orientation_finder: OrientationFinder):
        self.image_processor = image_processor
        self.star_identifier = star_identifier
        self.orientation_finder = orientation_finder

    def calculate_orientation(self, previous_orientation=None):
        image_stars = self.get_image_stars()
        if previous_orientation:
            # tracking mode
            frame = self.compare_frames(image_stars, previous_orientation)
            self.finalize(frame)
        else:
            # LIS mode
            frame = self.find_orientation(image_stars)
            self.finalize(frame)

    def finalize(self, frame):
        if frame:
            print(frame.orientation)
            self.calculate_orientation(frame)
        else:
            print('Orientation lost')
            self.calculate_orientation()

    def get_image_stars(self) -> [StarUV]:
        return self.image_processor.get_image_star_vectors()

    def compare_frames(self, image_stars, previous_orientation):
        pass

    def find_orientation(self, image_stars: [StarUV]):
        identified_stars = self.star_identifier.identify_stars(image_stars)
        orientation = self.orientation_finder.find_orientation(
            identified_stars)
        return orientation
