import numpy as np
import matplotlib.pyplot as plt

from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor
from program.tracker.main_program import StarTracker
from program.tracker.orientation_finder import OrientationFinder
from program.tracker.quest import QuestCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker


res_x = 900
res_y = 900


class TestStartrackerJetson:

    def test_full_startracker_brightness(
            self, pixel_size, focal_length, a_roi, c_roi, star_mag_pix,
            principal_point, planar_triangle_calculator,
            triangle_catalog, star_catalog):
        i_threshold = 150
        mag_threshold = 160

        centroid_calculator = CentroidCalculator(
            pixel_size,
            focal_length,
            a_roi,
            c_roi,
            i_threshold,
            mag_threshold,
            star_mag_pix,
            principal_point
        )
        image_processor = ImageProcessor(
            CameraConnector(), centroid_calculator)

        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calculator,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            orientation_finder=OrientationFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calculator),
            tracking_mode_enabled=False,
        )
        for i in range(5):
            sg = st.run()
            stars, q = next(sg)
            plot_result(stars, res_x, res_y)


def plot_result(stars, res_x_, res_y_):
    stars = np.array(stars)
    txt = stars[:, 1]
    txt = txt.astype(int)
    y = stars[:, 5]
    x = stars[:, 6]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(xmin=0, xmax=res_x_)
    ax.set_ylim(ymin=0, ymax=res_y_)

    for i, txt in enumerate(txt):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
