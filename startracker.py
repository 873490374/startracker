__author__ = 'Szymon Michalski'


print('Author: ', __author__)


"""
Processing steps:
Image correction:
- Dark frame subtraction
- Lens correction
Star detection (acquisition of xy coordinates):
- Image thresholding
- Formation of star clusters
- Image centroiding (possible techniques:
    - Weighted Sum,
    - Maximum Likelihood Estimator)
"""

"""
Star tracking and attitude computing
Relative motion tracking                    Star catalogue search
- operating on xy coordinates               - operating on celestial
                                            coordinates
                                            (requires transformation
                                            from xy coordinates)

                    - Pattern matching (possible techniques:
                        Angle Matching,
                        Spherical Triangle Pattern Matching,
                        Planar Triangle Pattern Matching,
                        Rate Matching
                    )

- compares patterns between two frames      - preforms search in on-board
                                            reduced star catalog
                                            (k-vector technique)

- requires preliminary knowledge            - no preliminary knowledge required
of attitude
- error propagation

                    - Determining the attitude quaternion (possible techniques:
                        The Predictive Attitude Determination,
                        QUEST,
                        TRIAD,
                        The Singular Value Decomposition,
                        The Fast Optimal
                        Attitude Matrix
                    )
"""