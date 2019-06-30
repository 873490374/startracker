import numpy as np

from program.tracker.image_processor import ImageProcessor
from program.tracker.attitude_finder import AttitudeFinder
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker


class StarTracker:

    def __init__(
            self,
            image_processor: ImageProcessor,
            star_identifier: StarIdentifier,
            attitude_finder: AttitudeFinder,
            tracker: Tracker,
            tracking_mode_enabled: bool):
        self.image_processor = image_processor
        self.star_identifier = star_identifier
        self.attitude_finder = attitude_finder
        self.tracker = tracker
        self.tracking_mode_enabled = tracking_mode_enabled

    def run(self):
        if self.tracking_mode_enabled:
            # tracking mode
            identified_stars = []
            while True:
                image_stars = self.get_image_stars()
                if (not identified_stars or len((set(
                        [int(star[1]) for star in identified_stars]))) < 3):
                    identified_stars = self.identify_stars(image_stars)
                else:
                    identified_stars = self.tracker.track(
                        image_stars, identified_stars)
                attitude = self.find_attitude(identified_stars)
                yield identified_stars, attitude
        else:
            while True:
                # LIS mode
                image_stars = self.get_image_stars()
                identified_stars = self.identify_stars(image_stars)
                attitude = self.find_attitude(identified_stars)
                yield identified_stars, attitude

    def get_image_stars(self) -> np.ndarray:
        return self.image_processor.get_image_star_vectors()

    def identify_stars(self, image_stars: np.ndarray):
        return self.star_identifier.identify_stars(image_stars)

    def find_attitude(self, identified_stars: np.ndarray):

        attitude = self.attitude_finder.find_attitude(
            identified_stars)
        return attitude
