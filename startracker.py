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

"""
1. Star detection
- Image thresholding: Otsu's method             <---
- Formation of star clusters: ???
- Image centroiding (possible techniques:
    - Weighted Sum,
    - Maximum Likelihood Estimator)             <---
2. Celestial to xy                              <---
3. Pattern Matching
    - Angle Matching,
    - Spherical Triangle Pattern Matching,
    - Planar Triangle Pattern Matching,         <---
    - Rate Matching
4. k-vector technique                           <---
5. Attitude calculation - quaternion
    - The Predictive Attitude Determination,
    - QUEST,                                    <---
    - TRIAD,
    - The Singular Value Decomposition,
    - The Fast Optimal
    - Attitude Matrix
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('1.jpg', 0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

thresh = ['img', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(eval(thresh[i]), 'gray')
    plt.title(thresh[i])

img2 = plt.show()
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')
plt.savefig(pp, format='pdf')

cv2.imwrite('result.jpeg', thresh1)
