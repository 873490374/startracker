import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
# noinspection PyPackageRequirements
import mock
import numpy as np
# noinspection PyPackageRequirements
import pytest
from PIL import Image

from program.const import MAIN_PATH
from program.tracker.attitude_finder import AttitudeFinder
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor
from program.tracker.main_program import StarTracker
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.quest import QuestCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker

images_path = os.path.join(MAIN_PATH, 'tests/images/')

expected_tracking = [
    # 0
    [{'id_cat': 24244.0, 'x': 560.7497329104594, 'y': 480.000195583081},
     {'id_cat': 24845.0, 'x': 498.687709233178, 'y': 301.1877779562518},
     {'id_cat': 24674.0, 'x': 303.62507020498754, 'y': 869.1875601227285},
     {'id_cat': 24305.0, 'x': 674.7500722159853, 'y': 106.25017421675594},
     {'id_cat': 24327.0, 'x': 573.4372387253776, 'y': 381.78131212541507},
     {'id_cat': 23231.0, 'x': 839.0315796800486, 'y': 509.2818722820722},
     {'id_cat': -1.0, 'x': 776.9689315903972, 'y': 704.3130829870277},
     {'id_cat': 23972.0, 'x': 537.9068525336747, 'y': 768.0628980018041},
     {'id_cat': 25247.0, 'x': 198.26569490601509, 'y': 740.954408742312},
     {'id_cat': 24436.0, 'x': 408.171967537126, 'y': 777.295668375114}],
    # 1
    [{'id_cat': 24244.0, 'x': 570.7497640605487, 'y': 452.00019533047794},
     {'id_cat': 24845.0, 'x': 508.0625418008655, 'y': 273.12526272827193},
     {'id_cat': 24674.0, 'x': 314.250064642502, 'y': 841.5625646987792},
     {'id_cat': 24305.0, 'x': 683.8750491316587, 'y': 77.562559979196},
     {'id_cat': 24327.0, 'x': 583.4385780365824, 'y': 353.78205932938323},
     {'id_cat': 23231.0, 'x': 848.9065917582869, 'y': 480.37498968517014},
     {'id_cat': -1.0, 'x': 787.0625628794024, 'y': 675.8129999150642},
     {'id_cat': 23972.0, 'x': 548.2191752565805, 'y': 739.9066308193715},
     {'id_cat': 25247.0, 'x': 208.7972419891475, 'y': 713.5636548726754},
     {'id_cat': 24436.0, 'x': 418.62902548680177, 'y': 749.4343367983734}],
    # 2
    [{'id_cat': 24244.0, 'x': 581.7497633785649, 'y': 429.0002152555769},
     {'id_cat': 24845.0, 'x': 518.6875612129152, 'y': 250.18767986629658},
     {'id_cat': 24674.0, 'x': 325.8749627034399, 'y': 818.8124674473596},
     {'id_cat': 24305.0, 'x': 694.0937966631273, 'y': 54.343826118861266},
     {'id_cat': 24327.0, 'x': 593.9063061482052, 'y': 330.65677159577194},
     {'id_cat': 23231.0, 'x': 860.0629977218637, 'y': 457.15715471658854},
     {'id_cat': -1.0, 'x': 798.5625008872748, 'y': 652.3755123033696},
     {'id_cat': 23972.0, 'x': 559.781578498931, 'y': 716.9379077064091},
     {'id_cat': 25247.0, 'x': 220.34411467521107, 'y': 691.3605804009424},
     {'id_cat': 24436.0, 'x': 430.26961791023, 'y': 726.7077667780636}],
    # 3
    [{'id_cat': 24244.0, 'x': 630.4996550179766, 'y': 386.5000418397547},
     {'id_cat': 24845.0, 'x': 565.8749745270225, 'y': 208.50019611720188},
     {'id_cat': 24674.0, 'x': 378.5624563894873, 'y': 778.9999942932393},
     {'id_cat': 24305.0, 'x': 739.1875524207064, 'y': 11.031377225511903},
     {'id_cat': 24327.0, 'x': 641.7808657048606, 'y': 288.5013146419351},
     {'id_cat': -1.0, 'x': 849.7186818392206, 'y': 607.9066742880561},
     {'id_cat': 23972.0, 'x': 611.5002375455949, 'y': 674.8125969181136},
     {'id_cat': 23364.0, 'x': 731.500249215219, 'y': 862.9378758843908},
     {'id_cat': 25247.0, 'x': 272.00036821956263, 'y': 652.5007851453136},
     {'id_cat': 24436.0, 'x': 481.9981134257662, 'y': 685.8072919544418}],
    # 4
    [{'id_cat': 24244.0, 'x': 648.3748088242443, 'y': 350.00024483418383},
     {'id_cat': -1.0, 'x': 7.254201571718317, 'y': 692.376115786707},
     {'id_cat': 24845.0, 'x': 582.6878017973704, 'y': 171.87537444530085},
     {'id_cat': 24674.0, 'x': 397.5625421890073, 'y': 743.000010743411},
     {'id_cat': 24327.0, 'x': 658.9689606802176, 'y': 251.5638192298765},
     {'id_cat': -1.0, 'x': 867.8125834463526, 'y': 570.375547164391},
     {'id_cat': 23972.0, 'x': 629.9067462367183, 'y': 638.0627222790822},
     {'id_cat': 23364.0, 'x': 750.1869592182076, 'y': 825.4683200274073},
     {'id_cat': 25247.0, 'x': 290.1875571529568, 'y': 616.7820640377918},
     {'id_cat': 24436.0, 'x': 500.4610056710869, 'y': 649.5280869391875}],
    # 5
    [{'id_cat': 24244.0, 'x': 668.12482588473, 'y': 316.12520009674154},
     {'id_cat': 24845.0, 'x': 601.687459438955, 'y': 138.2502064776711},
     {'id_cat': 24674.0, 'x': 418.9374977857674, 'y': 710.5001246668762},
     {'id_cat': 24327.0, 'x': 678.5941985446392, 'y': 217.50084064168183},
     {'id_cat': 23972.0, 'x': 650.9067992395244, 'y': 604.5322445094721},
     {'id_cat': 23364.0, 'x': 772.1880619098898, 'y': 791.4689413194926},
     {'id_cat': 25247.0, 'x': 311.1565020575132, 'y': 584.5009926864514},
     {'id_cat': -1.0, 'x': 8.239962468622018, 'y': 712.8288994613927},
     {'id_cat': -1.0, 'x': 18.566260817537135, 'y': 664.0518669243886},
     {'id_cat': 24436.0, 'x': 521.4961352907278, 'y': 616.2897868737793}],
    # 6
    [{'id_cat': 24244.0, 'x': 713.2499041195008, 'y': 279.50011627456297},
     {'id_cat': 24845.0, 'x': 645.0624533068595, 'y': 102.12494485900382},
     {'id_cat': 24674.0, 'x': 467.5625754230881, 'y': 676.0000320256611},
     {'id_cat': 24327.0, 'x': 722.5311537370499, 'y': 180.7813763645426},
     {'id_cat': 23972.0, 'x': 698.406840738332, 'y': 568.1564005919932},
     {'id_cat': 23364.0, 'x': 821.218884887006, 'y': 754.2817930554702},
     {'id_cat': 26563.0, 'x': 25.046862459492377, 'y': 499.7815026704044},
     {'id_cat': 25247.0, 'x': 358.67237482962867, 'y': 551.2667757145115},
     {'id_cat': 23875.0, 'x': 629.1250181072681, 'y': 891.1868318594047},
     {'id_cat': 26237.0, 'x': 31.1175598280938, 'y': 725.9699317175423},
     {'id_cat': 26241.0, 'x': 66.91013381877053, 'y': 633.4779713414133},
     {'id_cat': 24436.0, 'x': 569.2364236577579, 'y': 581.2077762957019},
     {'id_cat': 26220.0, 'x': 50.6995046421666, 'y': 680.236335996171}],
    # 7
    [{'id_cat': 24244.0, 'x': 734.8746561523451, 'y': 249.50017835783916},
     {'id_cat': 26241.0, 'x': 666.4378678131183, 'y': 72.6877278899453},
     {'id_cat': 24674.0, 'x': 490.7499432580164, 'y': 646.9999849782132},
     {'id_cat': 24327.0, 'x': 744.3447666705771, 'y': 150.87541184226365},
     {'id_cat': 23972.0, 'x': 721.3752539645872, 'y': 538.4068581347166},
     {'id_cat': 23364.0, 'x': 845.2194651312482, 'y': 723.8754376195491},
     {'id_cat': 26563.0, 'x': 47.875081618670286, 'y': 472.7502021290062},
     {'id_cat': 25247.0, 'x': 381.68805453127896, 'y': 522.5942112603525},
     {'id_cat': 23875.0, 'x': 653.6408705932133, 'y': 862.3596752771252},
     {'id_cat': 26237.0, 'x': 55.03951461240012, 'y': 698.9464729270026},
     {'id_cat': 24845.0, 'x': 90.31246709940183, 'y': 606.1768817262486},
     {'id_cat': 26220.0, 'x': 74.0959351636412, 'y': 653.246229893035},
     {'id_cat': 24436.0, 'x': 592.0273228393078, 'y': 551.7311326100147}],
    # 8
    [{'id_cat': 24244.0, 'x': 787.3748993510527, 'y': 208.00024516647392},
     {'id_cat': 24845.0, 'x': 716.5625870026774, 'y': 31.50040091401999},
     {'id_cat': 24674.0, 'x': 546.9374772059662, 'y': 607.6249905575917},
     {'id_cat': 24327.0, 'x': 795.4384225230993, 'y': 108.78136039004752},
     {'id_cat': 23972.0, 'x': 776.6261770512231, 'y': 496.7820510207331},
     {'id_cat': -1.0, 'x': 676.8127229618755, 'y': 870.6577033011363},
     {'id_cat': 26563.0, 'x': 102.43795256483621, 'y': 438.1412189834599},
     {'id_cat': 25247.0, 'x': 436.4849435679225, 'y': 484.51633894724637},
     {'id_cat': 23875.0, 'x': 711.7344894241915, 'y': 821.4377853710743},
     {'id_cat': 26237.0, 'x': 111.80514702154315, 'y': 664.0090057830523},
     {'id_cat': 26241.0, 'x': 145.94521263815267, 'y': 570.9774805676012},
     {'id_cat': 24436.0, 'x': 647.3613949315313, 'y': 511.49680951778635},
     {'id_cat': 26220.0, 'x': 130.5314373262238, 'y': 617.945365373721}],
    # 9
    [{'id_cat': 24244.0, 'x': 819.4998662982036, 'y': 161.50008798326095},
     {'id_cat': 24674.0, 'x': 581.4999536911218, 'y': 563.250050529679},
     {'id_cat': 24327.0, 'x': 826.9065476697131, 'y': 62.656730173569684},
     {'id_cat': 23972.0, 'x': 810.6261465209557, 'y': 450.7818825494905},
     {'id_cat': -1.0, 'x': 713.4387638404266, 'y': 825.3763402817408},
     {'id_cat': 27366.0, 'x': 12.046914020537884, 'y': 126.93773992550265},
     {'id_cat': 26563.0, 'x': 136.09424593111012, 'y': 396.5632819923817},
     {'id_cat': 25247.0, 'x': 470.3127083853823, 'y': 440.62579808088486},
     {'id_cat': 23875.0, 'x': 747.5154760955297, 'y': 775.969113927302},
     {'id_cat': 26237.0, 'x': 146.99269762484596, 'y': 622.2592047453132},
     {'id_cat': 26549.0, 'x': 18.656719029792626, 'y': 795.4777628608736},
     {'id_cat': 26241.0, 'x': 180.41004268042792, 'y': 528.8955247785898},
     {'id_cat': 26220.0, 'x': 165.26783743317446, 'y': 576.1700186375983},
     {'id_cat': 24436.0, 'x': 681.4297264087265, 'y': 466.4050740068394}],
    # 10
    [{'id_cat': 24244.0, 'x': 845.7498262384021, 'y': 115.000155686445},
     {'id_cat': 24674.0, 'x': 609.8124549383824, 'y': 517.9375027660511},
     {'id_cat': 24327.0, 'x': 852.7187836990104, 'y': 16.28205973361579},
     {'id_cat': 23972.0, 'x': 838.0626106715629, 'y': 404.6883737641006},
     {'id_cat': -1.0, 'x': 742.6567779785036, 'y': 779.4384664852745},
     {'id_cat': 27366.0, 'x': 38.046902608129116, 'y': 84.73461776607692},
     {'id_cat': 26563.0, 'x': 163.48458548765325, 'y': 353.50052996556036},
     {'id_cat': 25247.0, 'x': 498.06304469369707, 'y': 396.1731046041459},
     {'id_cat': 23875.0, 'x': 777.0627901181151, 'y': 729.8283854359177},
     {'id_cat': 26237.0, 'x': 175.5786390575352, 'y': 579.1182432386147},
     {'id_cat': 26549.0, 'x': 48.10166028016776, 'y': 752.7581996452477},
     {'id_cat': 25281.0, 'x': 351.4694073374153, 'y': 860.6570098456679},
     {'id_cat': 26311.0, 'x': 66.95294648346072, 'y': 889.0466371118565},
     {'id_cat': 26241.0, 'x': 208.51553946644546, 'y': 485.8408763534091},
     {'id_cat': 24436.0, 'x': 709.2070981723646, 'y': 420.67066125508353},
     {'id_cat': 26220.0, 'x': 193.7151011574502, 'y': 532.9981185615674}],
    # 11
    [{'id_cat': 24244.0, 'x': 865.8747156331601, 'y': 95.50019037756523},
     {'id_cat': 24674.0, 'x': 631.5000116219412, 'y': 499.50008548515586},
     {'id_cat': 23972.0, 'x': 859.219039161916, 'y': 384.9068673284276},
     {'id_cat': -1.0, 'x': 765.4388025107144, 'y': 760.376454162296},
     {'id_cat': 27366.0, 'x': 57.87504874074648, 'y': 68.21894374332604},
     {'id_cat': 26563.0, 'x': 184.48490249294392, 'y': 336.5005527231134},
     {'id_cat': 25247.0, 'x': 519.0159446417738, 'y': 377.7351600237193},
     {'id_cat': 23875.0, 'x': 799.2657133193679, 'y': 710.7349460002522},
     {'id_cat': 26237.0, 'x': 197.3208706553389, 'y': 561.9852854254091},
     {'id_cat': 26549.0, 'x': 70.67999530232234, 'y': 736.4464995046798},
     {'id_cat': 25281.0, 'x': 374.19595279748967, 'y': 842.9148818857439},
     {'id_cat': 26311.0, 'x': 90.04675661203106, 'y': 872.6249933300032},
     {'id_cat': 26241.0, 'x': 229.95684503861227, 'y': 468.54768279635437},
     {'id_cat': 26220.0, 'x': 215.2052070019596, 'y': 516.0568708979513},
     {'id_cat': 26727.0, 'x': 11.544247873009962, 'y': 780.0197904737504},
     {'id_cat': 24436.0, 'x': 730.4219439753028, 'y': 401.5280473924669}],
    # 12
    [{'id_cat': 23972.0, 'x': 892.7455824898504, 'y': 342.1250400561581},
     {'id_cat': 24674.0, 'x': 667.4999686914579, 'y': 458.25007093733683},
     {'id_cat': 25737.0, 'x': 266.75004698805174, 'y': 883.9999997220762},
     {'id_cat': -1.0, 'x': 802.8127973334681, 'y': 718.3762677329971},
     {'id_cat': 27366.0, 'x': 91.14068758505502, 'y': 31.3129049274274},
     {'id_cat': 26563.0, 'x': 219.5160066387328, 'y': 298.29755677522894},
     {'id_cat': 25247.0, 'x': 554.4850141822639, 'y': 337.5168855994444},
     {'id_cat': 23875.0, 'x': 836.6250001441206, 'y': 668.6098913330501},
     {'id_cat': 26237.0, 'x': 233.953699782575, 'y': 523.7667401307176},
     {'id_cat': 26549.0, 'x': 108.39089427623108, 'y': 698.6332746815527},
     {'id_cat': 25281.0, 'x': 412.45351074656304, 'y': 803.0781075591191},
     {'id_cat': 26311.0, 'x': 128.76562663839394, 'y': 834.742148799586},
     {'id_cat': 26241.0, 'x': 265.90226252711506, 'y': 430.1222273025802},
     {'id_cat': 26727.0, 'x': 48.425736762692736, 'y': 743.5358363457506},
     {'id_cat': 24436.0, 'x': 765.9454061925522, 'y': 359.7624130063536},
     {'id_cat': 26220.0, 'x': 251.65460034828624, 'y': 477.429779811739}],
    # 13
    [{'id_cat': 24674.0, 'x': 703.3125493681036, 'y': 430.68746525544196},
     {'id_cat': 25737.0, 'x': 304.81245027932266, 'y': 858.999989431259},
     {'id_cat': -1.0, 'x': 840.1259392488598, 'y': 690.1262127195719},
     {'id_cat': 26563.0, 'x': 254.04729401378148, 'y': 273.78210520468343},
     {'id_cat': 25247.0, 'x': 589.1721109687486, 'y': 310.6258897057281},
     {'id_cat': 23875.0, 'x': 873.4530343490441, 'y': 640.0159121943001},
     {'id_cat': 26237.0, 'x': 269.8989217896918, 'y': 498.96980154821676},
     {'id_cat': 26549.0, 'x': 145.51607770160751, 'y': 674.7115557462052},
     {'id_cat': 25281.0, 'x': 449.9296552679145, 'y': 777.2736218479886},
     {'id_cat': 26311.0, 'x': 166.79692086005622, 'y': 810.6171467731754},
     {'id_cat': 26241.0, 'x': 301.3007929453104, 'y': 405.2669054051392},
     {'id_cat': 26220.0, 'x': 287.2579832861145, 'y': 452.7425055501035},
     {'id_cat': 26727.0, 'x': 85.82798596181935, 'y': 720.0048429542211},
     {'id_cat': 24436.0, 'x': 800.921972831753, 'y': 331.75265316931007}],
    # 14
    [{'id_cat': 27658.0, 'x': 29.875333616641562, 'y': 146.87624886394724},
     {'id_cat': 24674.0, 'x': 741.812507237124, 'y': 401.68764599699847},
     {'id_cat': 25737.0, 'x': 346.5625923006037, 'y': 832.5001805650678},
     {'id_cat': -1.0, 'x': 880.4380323049301, 'y': 659.7823147193254},
     {'id_cat': -1.0, 'x': 456.06285712976216, 'y': 880.4065965308425},
     {'id_cat': 26536.0, 'x': 291.5628557726553, 'y': 247.1722752299936},
     {'id_cat': 25247.0, 'x': 627.1569558634105, 'y': 282.01682139140144},
     {'id_cat': 26237.0, 'x': 309.0552945795048, 'y': 472.5714419428035},
     {'id_cat': 26549.0, 'x': 185.73451240529323, 'y': 648.9299256084903},
     {'id_cat': 25281.0, 'x': 490.9768091041886, 'y': 749.5862893507044},
     {'id_cat': 26311.0, 'x': 207.87496086056714, 'y': 784.7343622680818},
     {'id_cat': 26241.0, 'x': 339.72260509801595, 'y': 378.485351378761},
     {'id_cat': 26220.0, 'x': 326.06475938015876, 'y': 425.94532774713764},
     {'id_cat': 26727.0, 'x': 126.4725967137184, 'y': 694.6140168026177},
     {'id_cat': 25930.0, 'x': 278.42191221070345, 'y': 887.6479610767468},
     {'id_cat': -1.0, 'x': 838.9219793105101, 'y': 301.7175452730256}],
    # 15
    [{'id_cat': -1.0, 'x': 70.43858329296444, 'y': 123.68913238149342},
     {'id_cat': 24674.0, 'x': 784.1875435422653, 'y': 372.5624631445595},
     {'id_cat': 25737.0, 'x': 391.62501856291726, 'y': 806.687725599616},
     {'id_cat': -1.0, 'x': 501.844418425148, 'y': 854.0321521742829},
     {'id_cat': 26563.0, 'x': 332.60961064491835, 'y': 221.65602472203662},
     {'id_cat': 25247.0, 'x': 668.3597106040495, 'y': 254.2353849309543},
     {'id_cat': 26311.0, 'x': 252.92180728311138, 'y': 759.9687262592638},
     {'id_cat': 26237.0, 'x': 351.7426232912968, 'y': 447.0637766125444},
     {'id_cat': 26549.0, 'x': 229.8672945463657, 'y': 624.5164491184236},
     {'id_cat': 25281.0, 'x': 535.5859560485667, 'y': 722.9225890114826},
     {'id_cat': 26241.0, 'x': 381.89066575274995, 'y': 352.7078636027089},
     {'id_cat': 26727.0, 'x': 170.97260038744554, 'y': 670.4187086325192},
     {'id_cat': 25930.0, 'x': 324.1562222855025, 'y': 862.899039871149},
     {'id_cat': 24436.0, 'x': 880.6329224209189, 'y': 272.3816173839548},
     {'id_cat': 26220.0, 'x': 368.34969971541534, 'y': 400.47469706766447}],
    # 16
    [{'id_cat': -1.0, 'x': 103.62591457845555, 'y': 93.68824456958083},
     {'id_cat': 24674.0, 'x': 818.9374772167782, 'y': 338.6249904990109},
     {'id_cat': 25737.0, 'x': 429.12494967766554, 'y': 774.812618980636},
     {'id_cat': -1.0, 'x': 539.343624589063, 'y': 821.4062942195546},
     {'id_cat': 26563.0, 'x': 366.56296067823337, 'y': 190.65679314841339},
     {'id_cat': 25247.0, 'x': 702.5316039478304, 'y': 220.7037914129499},
     {'id_cat': 25044.0, 'x': 590.3435226658248, 'y': 880.3297238058656},
     {'id_cat': 26237.0, 'x': 386.984840073149, 'y': 415.7042541963973},
     {'id_cat': 26549.0, 'x': 266.164047816647, 'y': 593.6880764797145},
     {'id_cat': 25281.0, 'x': 572.5470270414248, 'y': 690.1954413381285},
     {'id_cat': 26311.0, 'x': 290.0467461978132, 'y': 728.9530004209794},
     {'id_cat': 26199.0, 'x': 416.4922292714791, 'y': 321.30192904422523},
     {'id_cat': 26727.0, 'x': 207.70323310473225, 'y': 640.2080096580414},
     {'id_cat': 25930.0, 'x': 361.96097052054563, 'y': 831.4064318506571},
     {'id_cat': 26235.0, 'x': 403.3614830495351, 'y': 368.84573477783556}],
    # 17
    [{'id_cat': -1.0, 'x': 147.68805509655766, 'y': 63.75136616205252},
     {'id_cat': 24674.0, 'x': 865.5000629284999, 'y': 302.68752676844076},
     {'id_cat': 25737.0, 'x': 478.6874802315093, 'y': 741.8751401019778},
     {'id_cat': -1.0, 'x': 589.1870368631253, 'y': 787.6251215724775},
     {'id_cat': 26563.0, 'x': 411.5627767913961, 'y': 158.17234246523063},
     {'id_cat': 25247.0, 'x': 748.0162006344063, 'y': 185.67269323146925},
     {'id_cat': 25044.0, 'x': 640.5931358849616, 'y': 845.969583262361},
     {'id_cat': 26237.0, 'x': 433.93798290883905, 'y': 383.0558728367272},
     {'id_cat': 26549.0, 'x': 314.5470034581084, 'y': 562.0869751891893},
     {'id_cat': 25281.0, 'x': 621.6487561924125, 'y': 656.1799475440426},
     {'id_cat': 26311.0, 'x': 339.5781450377753, 'y': 697.0154610144318},
     {'id_cat': 26199.0, 'x': 462.7110413837666, 'y': 288.4425190875802},
     {'id_cat': 26235.0, 'x': 449.9650566513254, 'y': 336.1367671002709},
     {'id_cat': 26727.0, 'x': 256.47679736587736, 'y': 608.9109438234785},
     {'id_cat': 25930.0, 'x': 412.1485071307293, 'y': 799.0552383172648}],
    # 18
    [     {'id_cat': -1.0, 'x': 38.0, 'y': 262.0},
     {'id_cat': -1.0, 'x': 190.12565321869482, 'y': 30.75085908302605},
     {'id_cat': 25737.0, 'x': 526.5000900378247, 'y': 706.6252460145342},
     {'id_cat': -1.0, 'x': 637.1874580314468, 'y': 751.4693066057208},
     {'id_cat': 26885.0, 'x': 193.59388731229475, 'y': 864.6570231178224},
     {'id_cat': 26563.0, 'x': 454.67213368907744, 'y': 123.17227325784162},
     {'id_cat': 25247.0, 'x': 791.2188631611527, 'y': 148.15721215839864},
     {'id_cat': 25044.0, 'x': 688.9215122214671, 'y': 809.43904306113},
     {'id_cat': 26237.0, 'x': 478.6252206674759, 'y': 347.7742773799162},
     {'id_cat': 26549.0, 'x': 360.71118447028005, 'y': 527.7271840575746},
     {'id_cat': 25281.0, 'x': 668.6097928179396, 'y': 619.7661992691903},
     {'id_cat': 26311.0, 'x': 386.7812596622611, 'y': 662.6249318771422},
     {'id_cat': 26199.0, 'x': 506.74618909164883, 'y': 253.08317154203192},
     {'id_cat': 26727.0, 'x': 302.90226501575773, 'y': 575.1572394477711},
     {'id_cat': 25930.0, 'x': 460.1682383221284, 'y': 763.9770495410094},
     {'id_cat': 26235.0, 'x': 494.44352356039553, 'y': 300.88284855211884}],
    # 19
    [{'id_cat': -1.0, 'x': 68.0, 'y': 245.00000000000003},
     {'id_cat': 25737.0, 'x': 559.0626894698066, 'y': 686.6251392531383},
     {'id_cat': -1.0, 'x': 670.0316536531061, 'y': 731.2824107814746},
     {'id_cat': 26885.0, 'x': 227.0628690402857, 'y': 846.688560574462},
     {'id_cat': 26563.0, 'x': 484.1093375535521, 'y': 103.64085262121773},
     {'id_cat': 25247.0, 'x': 821.1410812583998, 'y': 126.9229198927681},
     {'id_cat': 25044.0, 'x': 722.1873512592902, 'y': 788.5945612892012},
     {'id_cat': 26237.0, 'x': 509.5160028688298, 'y': 328.43890038839123},
     {'id_cat': 26549.0, 'x': 392.4534812697399, 'y': 508.8131653374681},
     {'id_cat': 25281.0, 'x': 700.5234536774844, 'y': 599.0860099614675},
     {'id_cat': 26311.0, 'x': 419.0389829735203, 'y': 643.6171499349919},
     {'id_cat': 26199.0, 'x': 537.0507602405794, 'y': 233.28226174251768},
     {'id_cat': 26727.0, 'x': 334.8242457667121, 'y': 556.3444673331757},
     {'id_cat': 25930.0, 'x': 492.95338641159583, 'y': 744.492570464782},
     {'id_cat': 26235.0, 'x': 525.0003757310326, 'y': 281.240269370178}],
]


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
            planar_triangle_calculator=planar_triangle_calculator,
        ),
        tracking_mode_enabled=False,
    )


@pytest.mark.cuda
class TestTracking:

    def test_tracking(self, star_tracker, image_processor):
        t = 0
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        star_tracker.tracking_mode_enabled = True

        sg = star_tracker.run()
        for i in range(0, 20):
            img_path = os.path.join(
                images_path, 'test_tracking_{}.png'.format(i))
            img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('L')
            with mock.patch.object(
                    image_processor, 'get_image', return_value=img):
                start = timer()
                stars, q = next(sg)
                t += timer() - start
                a, g, b, n, att = validate(stars, q, expected_tracking[i])
                all_ += a
                good += g
                bad += b
                not_recognized += n
                attitude_not_found += att

        print('Average time: {}'.format(t/20))
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

        # plot_result(stars, 900, 900)
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