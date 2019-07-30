import math
import os

import numpy as np
import matplotlib.pyplot as plt

# noinspection PyPackageRequirements
import mock
import pytest
from PIL import Image
from Quaternion import Quat
from astropy import units as u
from astropy.coordinates import Angle

from program.const import MAIN_PATH
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor
from program.tracker.main_program import StarTracker
from program.tracker.attitude_finder import AttitudeFinder
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.quest import QuestCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker


images_path = os.path.join(MAIN_PATH, 'tests/images/')

expected_full = [
    # 0
    [{'id_cat': 30883, 'x': 567.1247112628691, 'y': 86.93461070322651},
     {'id_cat': 29650, 'x': 846.1869785987435, 'y': 231.4688043035078},  # mag>5
     {'id_cat': -1, 'x': 394.28125773268596, 'y': 455.0937834813715},  # Mars
     {'id_cat': 32246, 'x': 37.00010760593619, 'y': 191.95356365565536},
     {'id_cat': 30343, 'x': 511.4222424791687, 'y': 322.0785843996516},
     {'id_cat': 29696, 'x': 171.1255822197625, 'y': 872.4853767254992},
     {'id_cat': 29655, 'x': 629.5706706087175, 'y': 441.50053550658515},
     {'id_cat': 28734, 'x': 734.0314874862601, 'y': 649.4847448529001}],
    # 1
    [{'id_cat': 30883, 'x': 450.1890666096323, 'y': 33.498286003660006},
     {'id_cat': 29650, 'x': 731.5313206453219, 'y': 172.46862278262836},  # mag>5
     {'id_cat': 28716, 'x': 831.8127906505101, 'y': 399.50007674736503},
     {'id_cat': -1, 'x': 283.5937149261077, 'y': 404.2500142135894},  # Mars
     {'id_cat': 30343, 'x': 398.45352341293994, 'y': 269.2504501331041},
     {'id_cat': 29696, 'x': 68.56282699033468, 'y': 825.6564321649621},
     {'id_cat': 29655, 'x': 518.6171989110517, 'y': 386.4146671990824},
     {'id_cat': 28734, 'x': 627.0157261044701, 'y': 592.3754575851508}],
    # 2
    [{'id_cat': 29650, 'x': 535.3751040725068, 'y': 129.49992106301943},  # mag>5
     {'id_cat': 28716, 'x': 642.5626047595929, 'y': 353.1566097160898},
     {'id_cat': -1, 'x': 94.96874522550722, 'y': 375.5000103530815},  # Mars
     {'id_cat': 27468, 'x': 582.5002603637881, 'y': 854.6257768178182},
     {'id_cat': 30343, 'x': 205.40667781549266, 'y': 236.90688439361583},
     {'id_cat': 27913, 'x': 778.0006366446078, 'y': 503.1259701689598},
     {'id_cat': 29655, 'x': 329.40666488256215, 'y': 350.1571618716604},
     {'id_cat': 28734, 'x': 444.28910303286835, 'y': 552.2816071426246}],
    # 3
    [{'id_cat': 29650, 'x': 239.50020062556234, 'y': 819.7501238225412},  # mag>5
     {'id_cat': 27830, 'x': 228.18737413210405, 'y': 764.4369588074944},
     {'id_cat': 28716, 'x': 531.5625397012493, 'y': 125.78140709147112},
     {'id_cat': 26451, 'x': 865.7500222040394, 'y': 576.2501023546354},
     {'id_cat': 27468, 'x': 481.31304153566555, 'y': 628.2822357541728},
     {'id_cat': 27913, 'x': 669.5469223548134, 'y': 273.2506229407772},
     {'id_cat': 29655, 'x': 218.05470872613841, 'y': 128.742808694011},
     {'id_cat': 28734, 'x': 337.1172664320985, 'y': 328.98508831351336}],
    # 4
    [{'id_cat': 27629, 'x': 144.75010774710057, 'y': 600.5002603070665},
     {'id_cat': 27830, 'x': 132.31163621758986, 'y': 545.2491271837619},
     {'id_cat': 26451, 'x': 765.7187445567174, 'y': 344.3125613838372},
     {'id_cat': 27468, 'x': 382.53170573110845, 'y': 403.7819666539195},
     {'id_cat': 27913, 'x': 564.1256377650882, 'y': 45.48566671612761},
     {'id_cat': 28734, 'x': 232.289140596343, 'y': 107.28148973761174}],
    # 5
    [{'id_cat': 27629, 'x': 25.499828340051, 'y': 393.0},
     {'id_cat': 25945, 'x': 881.5000003358543, 'y': 32.249973155059166},
     {'id_cat': 27830, 'x': 11.999403301367025, 'y': 338.43669851908993},
     {'id_cat': 26451, 'x': 640.3437552477546, 'y': 121.37517278012822},
     {'id_cat': 27468, 'x': 258.7194066330189, 'y': 190.3132398525765},
     {'id_cat': 24822, 'x': 858.0622129261697, 'y': 449.4068908991457},
     {'id_cat': 25428, 'x': 345.961020130853, 'y': 767.2656479913894}],
    # 6
    [{'id_cat': -1, 'x': 865.0001931322183, 'y': 615.0000772528873},
     {'id_cat': -1, 'x': 319.1252534744547, 'y': 256.8752534744547},
     {'id_cat': -1, 'x': 692.1265514894797, 'y': 488.8737242947999},
     {'id_cat': -1, 'x': 655.4846653768201, 'y': 188.50055933817458},
     {'id_cat': -1, 'x': 158.72661666609645, 'y': 530.6566871751542}],
    # 7
    [{'id_cat': 23835, 'x': 675.8124430940038, 'y': 35.375567004507474},
     {'id_cat': 24822, 'x': 288.2190143899462, 'y': 127.00067694335095},
     {'id_cat': 23497, 'x': 582.0627013439513, 'y': 299.6876135121891},
     {'id_cat': 21881, 'x': 836.0468404510973, 'y': 672.7816978026173}],
    # 8
    [{'id_cat': 22565, 'x': 628.8125717655751, 'y': 134.12510427758554},
     {'id_cat': 23497, 'x': 283.4373612198232, 'y': 192.75010456271622},
     {'id_cat': 21881, 'x': 564.3593641933396, 'y': 546.2040280606743},
     {'id_cat': 20711, 'x': 834.0157496834609, 'y': 737.093890157238},
     {'id_cat': 20636, 'x': 880.1176654037401, 'y': 710.0399078320156}],
    # 9
    [{'id_cat': 20542, 'x': 892.9993996962646, 'y': 410.4997880545319},
     {'id_cat': -1, 'x': 336.6875208939059, 'y': 198.87521455748643},  # mag>5
     {'id_cat': -1, 'x': 864.875211144048, 'y': 201.1251700093903},  # mag>5
     {'id_cat': -1, 'x': 876.0615573625471, 'y': 203.1253266380703},  # mag>5
     {'id_cat': 21273, 'x': 848.9067019069573, 'y': 99.25044144456999},
     {'id_cat': 20877, 'x': 871.2498737979292, 'y': 277.7187023282484},
     {'id_cat': 20648, 'x': 845.1877151262182, 'y': 430.5319462474838},
     {'id_cat': 19990, 'x': 855.032052916954, 'y': 726.4062824751252},
     {'id_cat': 21683, 'x': 699.1719443720049, 'y': 113.49968598582294},
     {'id_cat': 21029, 'x': 841.0156629317739, 'y': 238.4221812711265},
     {'id_cat': 20889, 'x': 726.7810684857418, 'y': 487.0786794372741},
     {'id_cat': 21881, 'x': 301.9533827609574, 'y': 614.3760079507216},
     {'id_cat': 20711, 'x': 584.546842022391, 'y': 785.1878434014532},
     {'id_cat': 20885, 'x': 889.47601759162, 'y': 242.96141434645682},
     {'id_cat': 20635, 'x': 628.6178080661252, 'y': 754.7821295718782},
     {'id_cat': -1, 'x': 777.1410357097658, 'y': 846.7432292605831},  # Mercury
     {'id_cat': 21421, 'x': 729.1134635583808, 'y': 200.4106536021432}],
    # 10
    [{'id_cat': -1, 'x': 834.9999320675249, 'y': 252.99999999999997},  # mag>5
     {'id_cat': 20484, 'x': 647.0, 'y': 428.99993204675184},
     {'id_cat': -1, 'x': 646.7499501232335, 'y': 113.50018894132498},  # mag>5
     {'id_cat': -1, 'x': 775.3749302206297, 'y': 328.4999742389376},  # mag>5
     {'id_cat': -1, 'x': 780.374999155866, 'y': 803.4999709287642},  # mag>5
     {'id_cat': 20901, 'x': 711.0626763666653, 'y': 79.62537410881141},
     {'id_cat': -1, 'x': 567.405564029769, 'y': 267.96878808513065},  # mag>5
     {'id_cat': -1, 'x': 33.12515587409143, 'y': 303.6879360496431},  # mag>5
     {'id_cat': 20542, 'x': 604.8126827201727, 'y': 474.12505112994336},
     {'id_cat': 21273, 'x': 536.4064493789799, 'y': 167.06309658415913},
     {'id_cat': 20732, 'x': 679.4377504194704, 'y': 233.78124739236245},
     {'id_cat': 20713, 'x': 644.0000421093443, 'y': 307.62554255757993},
     {'id_cat': 20877, 'x': 571.5937777980502, 'y': 343.2190684989964},
     {'id_cat': 20648, 'x': 557.0315080002736, 'y': 497.28203551504055},
     {'id_cat': 19990, 'x': 589.0322078865099, 'y': 791.2816946746901},
     {'id_cat': 21683, 'x': 388.51601274123425, 'y': 192.15603303229892},
     {'id_cat': 21029, 'x': 538.7975039402643, 'y': 306.2189972888662},
     {'id_cat': 20205, 'x': 766.7033933802122, 'y': 379.1566861958828},
     {'id_cat': 20455, 'x': 621.7970910253413, 'y': 494.5630875893638},
     {'id_cat': 20889, 'x': 443.4848205179912, 'y': 562.1723884513838},
     {'id_cat': 21881, 'x': 28.6090006474521, 'y': 720.6560933151602},
     {'id_cat': 20711, 'x': 323.7342880925874, 'y': 870.2816369301927},
     {'id_cat': 20635, 'x': 365.3834608590777, 'y': 836.6023360073452},
     {'id_cat': 21421, 'x': 424.29304354156307, 'y': 276.58652416154985},
     {'id_cat': -1, 'x': 587.9144809035329, 'y': 306.8558384144857}],
    # 20894 or 20885, stars are too close to each other
    # 11
    [{'id_cat': -1, 'x': 544.9999323158144, 'y': 567.0},  # mag>5
     {'id_cat': 20484, 'x': 370.00003576026324, 'y': 756.0000357602632},
     {'id_cat': -1, 'x': 119.00040065561828, 'y': 132.49908266350715},  # mag>5
     {'id_cat': -1, 'x': 172.37509409831608, 'y': 101.5000212725224},  # mag>5
     {'id_cat': 19860, 'x': 837.6250059038106, 'y': 195.0000437092759},
     {'id_cat': -1, 'x': 348.500187957711, 'y': 440.50002208322206},  # mag>5
     {'id_cat': -1, 'x': 490.8751029084925, 'y': 646.7500776155109},  # mag>5
     {'id_cat': 20901, 'x': 409.6250992558571, 'y': 402.68775650059393},
     {'id_cat': 20885, 'x': 279.43698658111816, 'y': 600.3125818477126},
     {'id_cat': 20542, 'x': 330.2498512033075, 'y': 803.5625978098354},
     {'id_cat': 21589, 'x': 245.6876376287473, 'y': 271.50021397463445},
     {'id_cat': 21273, 'x': 241.37509169398146, 'y': 501.50010364368364},
     {'id_cat': 20732, 'x': 388.625187293107, 'y': 558.7817939652551},
     {'id_cat': 20713, 'x': 358.34389740904413, 'y': 634.5942757474443},
     {'id_cat': 20877, 'x': 288.56255430053795, 'y': 674.7187282732788},
     {'id_cat': 20648, 'x': 284.4378354259019, 'y': 829.7505649194422},
     {'id_cat': 21402, 'x': 385.8128239634126, 'y': 103.43792236423269},
     {'id_cat': 19740, 'x': 854.5313534960917, 'y': 240.6564269237767},
     {'id_cat': 21683, 'x': 95.51583854679765, 'y': 536.4529219275155},
     {'id_cat': 21027, 'x': 253.1566320265141, 'y': 640.2659589808871},
     {'id_cat': 20205, 'x': 485.62531422533993, 'y': 697.7980867648475},
     {'id_cat': 20455, 'x': 348.87521230493576, 'y': 822.8136136659203},
     {'id_cat': -1, 'x': 302.1720524035215, 'y': 637.5550010302663}],
    # 12
    [{'id_cat': -1, 'x': 77.00014751265044, 'y': 228.0000815180589},  # mag>5
     {'id_cat': 19860, 'x': 527.500000237823, 'y': 511.5000005431974},
     {'id_cat': -1, 'x': 54.50003159547557, 'y': 787.5001777811494},  # mag>5
     {'id_cat': -1, 'x': 789.7503407394258, 'y': 554.0000465988617},  # mag>5
     {'id_cat': 20901, 'x': 113.62518716827203, 'y': 745.6878761813817},
     {'id_cat': 18907, 'x': 883.9065234546742, 'y': 378.2502448209549},
     {'id_cat': -1, 'x': 609.2501559879146, 'y': 430.56238905409896},  # mag>5
     {'id_cat': 19860, 'x': 547.531767511817, 'y': 555.7813859254959},
     {'id_cat': 21402, 'x': 71.46925565590425, 'y': 448.28227448262237}],
    # 13
    [{'id_cat': 19860, 'x': 338.87503500728104, 'y': 828.7500151105805},
     {'id_cat': -1, 'x': 602.5000565725608, 'y': 861.6253370484962},   # mag>5
     {'id_cat': -1, 'x': 769.5000038950031, 'y': 407.5000375263471},  # mag>5
     {'id_cat': 18907, 'x': 690.0314170851098, 'y': 682.2814960834111},
     {'id_cat': -1, 'x': 417.43738444773055, 'y': 744.7497054423002},  # mag>5
     {'id_cat': 19740, 'x': 360.43776128157333, 'y': 872.4691373314445}],
]

# noinspection LongLine
expected_moon = [
    {'id_cat': -1, 'x': 774.0019202131829, 'y': 404.0024296574967},  # mag>5
    {'id_cat': -1, 'x': 836.8755084490103, 'y': 226.7509400284472},  # mag>5
    {'id_cat': 104019, 'x': 93.62508738403614, 'y': 829.6880011950361},
    {'id_cat': -1, 'x': 790.0625022588831, 'y': 794.3127642906418},  # mag>5
    {'id_cat': 102485, 'x': 52.32835870205922, 'y': 212.50068546862224},
    {'id_cat': 101027, 'x': 774.0004874433298, 'y': 452.2816490779578}]  # mag>5

expected_sun = [
    {'id_cat': -1, 'x': 458.2500286063857, 'y': 524.5002661895369},  # mag>5
    {'id_cat': -1, 'x': 329.0001380663201, 'y': 183.00013788723206},  # mag>5
    {'id_cat': -1, 'x': 461.5, 'y': 834.25},  # mag>5
    {'id_cat': -1, 'x': 256.6250309014855, 'y': 593.6879848324827},  # mag>5
    {'id_cat': -1, 'x': 49.749261678641886, 'y': 736.0619835197087},  # mag>5
    {'id_cat': 19990, 'x': 137.31260448872354, 'y': 195.2812751852752},
    {'id_cat': 19038, 'x': 324.4690777224997, 'y': 424.2192721924408},
    {'id_cat': -1, 'x': 613.4843304047495, 'y': 778.1101952812508},  # mag>5
    {'id_cat': -1, 'x': 86.04716870540567, 'y': 328.8522409209291},  # Mercury
    {'id_cat': 17847, 'x': 541.0787177433388, 'y': 722.4846130227927},
    {'id_cat': 17608, 'x': 599.1334595933151, 'y': 736.9855523016442},
    {'id_cat': 17499, 'x': 618.8831652074186, 'y': 764.0945192028254},
    {'id_cat': -1, 'x': 423.9109240446152, 'y': 342.78972296645617},  # sun
    {'id_cat': 17702, 'x': 571.6291970560633, 'y': 740.403262362156},
    {'id_cat': 17573, 'x': 594.2016537835002, 'y': 784.9067972144085},
    {'id_cat': -1, 'x': 408.413548764507, 'y': 312.6115968387289},  # sun
    {'id_cat': -1, 'x': 425.10231479807044, 'y': 316.1713094219349},  # sun
    {'id_cat': -1, 'x': 399.0926957735787, 'y': 328.71873809133115},  # sun
    {'id_cat': -1, 'x': 412.75, 'y': 328.28125},  # sun
    {'id_cat': -1, 'x': 428.563288254241, 'y': 330.13677453060563},  # sun
    {'id_cat': -1, 'x': 409.4272278589646, 'y': 344.9511391113605}]  # sun

expected_brightness = [
    {'id_cat': 26563, 'x': 658.4063000612621, 'y': 90.93732311236997},
    {'id_cat': 25737, 'x': 554.9688613530328, 'y': 669.500036171366},
    {'id_cat': 26885, 'x': 190.65669744783236, 'y': 722.5316498113112},
    {'id_cat': 25282, 'x': 647.3750014678801, 'y': 745.5002337741047},
    {'id_cat': 28413, 'x': 60.109686504791526, 'y': 168.4379982858984},
    {'id_cat': 25044, 'x': 679.8747257817444, 'y': 816.0788298503406},
    {'id_cat': 26237, 'x': 615.2501235404593, 'y': 312.9695689462995},
    {'id_cat': 26549, 'x': 449.5471155815289, 'y': 450.007751873414},
    {'id_cat': 26311, 'x': 434.3203427944378, 'y': 587.2893505388877},
    {'id_cat': 25281, 'x': 716.4220491072787, 'y': 628.8596025147667},
    {'id_cat': 26199, 'x': 669.9686006746604, 'y': 230.8760841094297},
    {'id_cat': 26727, 'x': 380.5548941859445, 'y': 478.434290454249},
    {'id_cat': 25930, 'x': 474.5508576938088, 'y': 704.9885650257854},
    {'id_cat': 26220, 'x': 646.46393525763, 'y': 275.17387541828714}]


@pytest.fixture
def camera_fov():
    return 10


@pytest.fixture
def cos_camera_fov(camera_fov):
    return np.cos(np.deg2rad(camera_fov))


@pytest.fixture
def focal_length_normalized(camera_fov):
    return 0.5 / np.tan(np.deg2rad(camera_fov) / 2)


@pytest.fixture
def sensor_variance():
    return 270e-6 / 10


@pytest.fixture
def res_x():
    return 900


@pytest.fixture
def res_y():
    return 900


@pytest.fixture
def pixel_size():
    return 1


@pytest.fixture
def focal_length(focal_length_normalized, res_x):
    return focal_length_normalized * res_x


@pytest.fixture
def a_roi():
    return 5


@pytest.fixture
def c_roi():
    return 10


@pytest.fixture
def i_threshold():
    return 150


@pytest.fixture
def mag_threshold():
    return 160


@pytest.fixture
def star_mag_pix():
    return 14


@pytest.fixture
def principal_point(res_x, res_y):
    return 0.5 * res_x, 0.5 * res_y


@pytest.fixture
def centroid_calculator(
        pixel_size, focal_length, a_roi, c_roi,
        i_threshold, mag_threshold, star_mag_pix, principal_point):
    return CentroidCalculator(
        pixel_size,
        focal_length,
        a_roi,
        c_roi,
        i_threshold,
        mag_threshold,
        star_mag_pix,
        principal_point
    )


@pytest.fixture
def image_processor(centroid_calculator):
    return ImageProcessor(CameraConnector(), centroid_calculator)


@pytest.fixture
def triangle_catalog():
    catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
    filename_triangle = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    with open(filename_triangle, 'rb') as f:
        triangle_catalog = np.genfromtxt(
            f, dtype=np.float64, delimiter=',')
    return triangle_catalog


@pytest.fixture
def star_catalog():
    filename_star = os.path.join(
        MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
    with open(filename_star, 'rb') as f:
        star_catalog = np.genfromtxt(
            f, dtype=np.float64, delimiter=',')
    return star_catalog


@pytest.fixture
def planar_triangle_calculator(sensor_variance):
    return PlanarTriangleCalculator(
        sensor_variance=sensor_variance
    )


@pytest.fixture
def star_tracker(
        image_processor, planar_triangle_calculator,
        triangle_catalog, star_catalog):
    return StarTracker(
        image_processor=image_processor,
        star_identifier=StarIdentifier(
            planar_triangle_calculator=planar_triangle_calculator,
            triangle_catalog=triangle_catalog,
            star_catalog=star_catalog,
        ),
        attitude_finder=AttitudeFinder(
            quest_calculator=QuestCalculator(),
            star_catalog=star_catalog,
        ),
        tracker=Tracker(
            planar_triangle_calculator=planar_triangle_calculator),
        tracking_mode_enabled=False,
    )

@pytest.mark.cuda
class TestFullStarTracker:

    def test_full_startracker(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        for i in range(14):
            img_path = os.path.join(
                images_path, 'test_full_{}.png'.format(i))
            img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('L')
            with mock.patch.object(
                    image_processor, 'get_image', return_value=img):

                stars, q = next(sg)
                a, g, b, n, att = validate(stars, q, expected_full[i])
                all_ += a
                good += g
                bad += b
                not_recognized += n
                attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_moon(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        img_path = os.path.join(images_path, 'test_full_moon.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_moon)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_sun(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        img_path = os.path.join(images_path, 'test_full_sun.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_sun)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_brightness(
            self, pixel_size, focal_length, a_roi, c_roi, star_mag_pix,
            principal_point, planar_triangle_calculator,
            triangle_catalog, star_catalog):
        i_threshold = 160
        mag_threshold = 180

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
            attitude_finder=AttitudeFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calculator),
            tracking_mode_enabled=False,
        )
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = st.run()
        img_path = os.path.join(images_path, 'test_full_brightness.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(
                image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_brightness)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))


def validate(stars, q, expected):
    all_ = 0
    good = 0
    bad = 0
    not_recognized = 0
    attitude_not_found = 0

    if not stars or q is None:
        attitude_not_found += 1
    if not stars:
        not_recognized_in_scene = sum(
            [1 for s in expected if s['id_cat'] != -1])
        all_ += not_recognized_in_scene
        not_recognized += not_recognized_in_scene
    else:
        for es in expected:
            if es['id_cat'] != -1:
                for s in stars:
                    if (
                            np.isclose(s[5], es['x'], atol=0.00001) and
                            np.isclose(s[6], es['y'], atol=0.00001)):
                        all_ += 1
                        if s[1] == es['id_cat']:
                            good += 1
                        elif s[1] == -1:
                            not_recognized += 1
                        else:
                            bad += 1

        print('')
        print('Quaternion =', q)
        if q is not None:
            q = Quat(q)
            xx = (360 - q.ra) + 90
            print(xx)
            print(Angle('{}d'.format(xx)).to_string(unit=u.hour))
            print(q.dec)
            print(360-q.roll)

        # plot_result(stars, res_x(), res_y())
    return all_, good, bad, not_recognized, attitude_not_found


def plot_result(stars, res_x_, res_y_):
    stars = np.array(stars)
    txt = stars[:, 1]
    txt = txt.astype(int)
    x = stars[:, 5]
    y = stars[:, 6]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(xmin=0, xmax=res_x_)
    ax.set_ylim(ymin=0, ymax=res_y_)

    for i, txt in enumerate(txt):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
