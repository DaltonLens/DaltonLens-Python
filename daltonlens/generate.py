from collections import namedtuple
import itertools

import numpy as np
from pathlib import Path
import sys
import random

from daltonlens import convert, simulate, utils

def rgb_span(width, height):
    """Generate an image that spans the full RGB range.
    
    The underlying image is always 27x27. It corresponds to a 2D view of
    a 9x9x9 image spanning the range of the blue, red and green channels.

    Parameters
    ----------
    width, height : int
        Size of the output image. The 27x27 image will be scaled to match
        that size, repeating the pixel values (nearest neighbor).

    Returns
    -------
    im : array of shape (height, width, 3)
        Color image.
    """

    # True image is always 27x27
    # Larger dimensions are obtained by transforming into that 27x27 image
    steps_per_channel = 8
    steps = [*range(0,256,256//steps_per_channel), 255]
    n = len(steps)
    # same as 27x27
    n_pixels = 9*9*9

    im = np.zeros((height,width,3), dtype=np.uint8)
    for r_fullRes in range(0, height):
        for c_fullRes in range(0, width):
            r = (27 * r_fullRes) // height
            c = (27 * c_fullRes) // width
            r_idx = r%9
            g_idx = c%9
            b_idx = (c//9) + (r//9)*3
            im[r_fullRes,c_fullRes,:] = (steps[r_idx], steps[g_idx], steps[b_idx])
    return im

def ishihara_image(fg_color, bg_color, mask):
    """Generate an Ishihara-like image of small circles

    Template downloaded from the generator made by Francisco Couzo
    https://franciscouzo.github.io/ishihara/

    Parameters
    ----------
    fg_color : Tuple(int)
        sRGB color for the foreground circles, e.g. (255,0,0).
    bg_color : Tuple)int)
        sRGB color for the background circles.
    mask : array of shape (N,N,1)
         Values > 127 indicate foreground.

    Returns
    -------
    im : array of shape (1738, 1738, 3)
        Color image. The color of each circle depends on the mask.
    """

    try: import cv2
    except ImportError:
        sys.stderr.write("OpenCV is required for ishihara_image: `pip install opencv-python'\n")
        sys.exit (1)

    # These circles were generated with Francisco Couzo generator
    # https://franciscouzo.github.io/ishihara/
    # An image of black circles was saved as SVG (resources/ishihara_circles.svg) and parsed
    # to generate the array below. I've lost the code :facepalm: , but it was pretty trivial.
    circle_image_width = 1738
    circle_image_height = 1738
    circles = [(1307, 1217, 26), (702, 1166, 15), (1482, 1019, 25), (900, 1281, 30), (1059, 964, 15), (816, 938, 30), (898, 567, 22), (843, 828, 31), (848, 883, 22), (960, 778, 29), (710, 860, 33), (317, 1032, 26), (280, 848, 24), (794, 1320, 28), (73, 1017, 14), (514, 762, 13), (585, 764, 16), (946, 117, 29), (960, 1146, 27), (968, 338, 24), (1273, 974, 22), (1055, 754, 26), (1034, 920, 32), (1053, 855, 29), (673, 330, 13), (1104, 950, 25), (184, 1072, 27), (1041, 676, 33), (1040, 447, 29), (1030, 613, 26), (914, 982, 27), (898, 736, 19), (300, 918, 20), (1141, 315, 14), (554, 1554, 32), (1125, 809, 23), (886, 1050, 31), (406, 781, 21), (1473, 396, 28), (1096, 1256, 25), (719, 1319, 13), (1301, 937, 23), (1650, 766, 21), (1259, 481, 24), (806, 295, 27), (177, 1005, 32), (840, 1248, 32), (132, 662, 29), (1555, 832, 28), (936, 1058, 13), (599, 1306, 32), (896, 812, 23), (1222, 851, 28), (1263, 1086, 23), (833, 573, 25), (907, 1207, 23), (456, 264, 15), (1160, 956, 28), (666, 787, 15), (570, 906, 22), (967, 1259, 19), (744, 648, 27), (726, 1009, 23), (760, 409, 21), (139, 1055, 19), (937, 1529, 30), (781, 877, 17), (210, 1345, 20), (345, 986, 15), (1065, 1402, 28), (1146, 426, 29), (481, 1311, 21), (485, 1036, 20), (386, 689, 31), (1589, 1170, 20), (324, 1326, 29), (1124, 1064, 22), (587, 846, 24), (833, 1102, 25), (832, 1477, 30), (227, 780, 22), (397, 1151, 17), (1508, 1300, 22), (613, 509, 19), (698, 416, 26), (688, 1108, 31), (1209, 1028, 26), (614, 1604, 20), (723, 691, 17), (669, 1299, 30), (932, 702, 26), (956, 1018, 13), (557, 1481, 15), (878, 477, 26), (966, 870, 29), (289, 354, 32), (1004, 732, 21), (1105, 613, 18), (585, 1129, 26), (418, 886, 30), (451, 1445, 23), (1087, 1557, 33), (217, 589, 26), (736, 527, 14), (1318, 688, 12), (66, 763, 24), (800, 466, 26), (325, 514, 22), (386, 1363, 15), (1252, 174, 32), (524, 395, 19), (1194, 1200, 30), (764, 593, 14), (1086, 535, 28), (1598, 937, 29), (341, 1122, 23), (1008, 786, 15), (447, 948, 30), (433, 1138, 14), (1111, 684, 27), (960, 948, 29), (648, 613, 16), (1422, 465, 19), (1260, 765, 32), (908, 58, 17), (635, 729, 32), (1342, 866, 27), (1019, 1476, 33), (1256, 1417, 18), (284, 1419, 21), (807, 686, 33), (1431, 1249, 31), (601, 301, 16), (1037, 365, 18), (1670, 831, 30), (703, 1580, 24), (1045, 1291, 25), (174, 830, 18), (1273, 550, 20), (657, 534, 20), (422, 1104, 15), (372, 480, 29), (568, 601, 29), (395, 829, 23), (812, 531, 13), (1486, 941, 19), (479, 1372, 32), (1168, 581, 30), (644, 1564, 26), (547, 552, 16), (1402, 556, 18), (1452, 314, 14), (378, 1283, 16), (499, 626, 23), (1089, 456, 14), (1540, 1111, 28), (1018, 1337, 18), (590, 111, 21), (1522, 592, 24), (202, 650, 31), (445, 1227, 21), (914, 500, 14), (948, 1307, 13), (100, 721, 25), (1022, 145, 13), (1439, 909, 20), (892, 1424, 18), (936, 1230, 13), (1336, 561, 26), (1606, 609, 29), (789, 768, 13), (467, 400, 21), (55, 918, 13), (525, 812, 26), (1309, 434, 20), (602, 234, 26), (1077, 1139, 26), (637, 1050, 17), (123, 1179, 16), (905, 864, 28), (1347, 1001, 19), (574, 357, 27), (1144, 865, 22), (1560, 744, 18), (1159, 1441, 18), (1267, 1529, 28), (496, 477, 19), (447, 463, 28), (400, 567, 22), (976, 1416, 20), (1008, 1555, 29), (1306, 895, 13), (982, 486, 25), (733, 268, 17), (262, 754, 14), (1268, 1460, 24), (1206, 1502, 23), (757, 919, 24), (1105, 345, 21), (1214, 625, 19), (264, 1257, 19), (202, 497, 22), (1484, 756, 25), (869, 1351, 29), (399, 396, 23), (989, 1088, 25), (1012, 1132, 17), (1025, 1651, 33), (962, 431, 31), (146, 521, 31), (629, 959, 31), (716, 784, 31), (92, 839, 19), (755, 1495, 19), (750, 1185, 32), (851, 764, 19), (540, 735, 17), (844, 999, 23), (1399, 1322, 31), (206, 929, 29), (1293, 1488, 12), (1021, 827, 13), (867, 210, 18), (299, 988, 16), (1157, 508, 30), (678, 653, 18), (594, 555, 15), (764, 266, 13), (556, 1408, 29), (1225, 1357, 22), (1662, 991, 22), (981, 647, 15), (1436, 583, 24), (782, 794, 14), (1262, 433, 21), (1353, 687, 19), (213, 847, 18), (1001, 538, 27), (470, 844, 18), (790, 1101, 15), (330, 385, 19), (200, 535, 13), (1412, 868, 17), (411, 1015, 31), (1334, 640, 25), (1355, 497, 32), (471, 1520, 19), (919, 255, 17), (950, 1581, 16), (689, 1511, 16), (573, 1207, 27), (1342, 1364, 19), (1375, 307, 25), (642, 1407, 28), (1187, 1131, 20), (440, 1056, 18), (1158, 1302, 16), (877, 275, 24), (1593, 1068, 21), (1339, 1307, 28), (404, 240, 30), (1147, 1610, 15), (1013, 978, 17), (363, 786, 13), (772, 536, 15), (1303, 796, 13), (1446, 783, 20), (717, 370, 20), (734, 564, 18), (1116, 1358, 24), (477, 218, 14), (905, 1106, 16), (147, 731, 20), (1411, 1097, 26), (537, 677, 15), (839, 616, 14), (867, 141, 25), (315, 825, 16), (689, 457, 14), (1424, 1181, 16), (1537, 1017, 18), (867, 1594, 23), (1397, 1413, 18), (1158, 143, 23), (1445, 521, 30), (960, 1351, 24), (729, 1355, 16), (878, 1553, 16), (554, 446, 24), (953, 299, 13), (1027, 197, 16), (692, 502, 13), (1544, 1184, 13), (624, 563, 13), (1140, 1503, 24), (1208, 301, 17), (1656, 919, 14), (233, 454, 13), (314, 576, 30), (1481, 616, 13), (887, 627, 25), (879, 402, 22), (1272, 230, 17), (847, 1185, 20), (1353, 1505, 23), (789, 1603, 22), (994, 90, 25), (231, 1105, 20), (1318, 295, 18), (626, 468, 22), (1560, 452, 17), (1183, 759, 30), (661, 997, 15), (1169, 358, 16), (1590, 1216, 23), (1544, 530, 32), (1421, 654, 32), (100, 924, 24), (1397, 802, 27), (599, 1637, 16), (1191, 1074, 16), (1174, 207, 21), (757, 112, 17), (659, 127, 29), (932, 195, 12), (585, 1059, 21), (653, 1210, 25), (1591, 507, 16), (1369, 1449, 18), (1466, 1131, 14), (519, 578, 14), (687, 954, 21), (1173, 683, 31), (675, 227, 24), (406, 620, 30), (448, 324, 24), (764, 829, 22), (1300, 1021, 29), (302, 696, 14), (489, 909, 18), (1090, 1617, 25), (1120, 242, 26), (725, 1429, 23), (239, 1158, 29), (891, 927, 29), (487, 715, 15), (898, 1476, 14), (153, 1208, 13), (645, 848, 16), (637, 1119, 19), (758, 752, 15), (692, 720, 16), (524, 1239, 28), (441, 666, 18), (1108, 1431, 20), (293, 1070, 18), (801, 233, 30), (1037, 1027, 30), (1087, 1095, 18), (560, 157, 33), (1239, 1198, 15), (562, 1164, 12), (142, 877, 23), (514, 1095, 15), (1226, 933, 16), (1610, 1008, 24), (1454, 707, 28), (1118, 166, 22), (264, 796, 13), (771, 349, 23), (806, 1168, 13), (564, 281, 23), (778, 1427, 24), (184, 440, 16), (1454, 1319, 19), (354, 737, 23), (647, 1484, 31), (992, 1032, 13), (1345, 1069, 23), (490, 1130, 25), (351, 1485, 16), (958, 1207, 12), (1654, 660, 13), (636, 211, 13), (954, 557, 18), (766, 1658, 21), (1408, 1040, 21), (1042, 1250, 13), (934, 635, 22), (1248, 632, 14), (883, 701, 14), (674, 292, 14), (1404, 372, 29), (858, 1440, 12), (544, 500, 30), (571, 694, 23), (1307, 1382, 18), (918, 1378, 22), (1084, 189, 16), (1076, 809, 17), (545, 1023, 17), (1108, 757, 21), (554, 963, 15), (811, 1025, 15), (733, 1062, 15), (740, 473, 18), (1363, 919, 23), (957, 240, 20), (338, 1395, 25), (859, 723, 13), (264, 953, 28), (657, 1352, 15), (286, 419, 27), (916, 343, 24), (789, 1540, 20), (1329, 737, 23), (189, 1285, 17), (1566, 889, 14), (332, 1179, 28), (937, 592, 17), (1054, 172, 14), (163, 588, 20), (467, 518, 27), (1389, 426, 16), (1171, 1362, 25), (1461, 1079, 25), (747, 1263, 26), (1348, 261, 25), (678, 572, 16), (99, 1050, 14), (220, 703, 24), (1276, 1166, 25), (1466, 1394, 21), (115, 1115, 30), (873, 533, 14), (830, 436, 13), (164, 470, 17), (694, 178, 23), (1271, 1321, 13), (1638, 729, 16), (295, 625, 21), (1216, 568, 13), (417, 1330, 16), (1107, 876, 14), (518, 1283, 12), (1297, 852, 16), (1103, 293, 18), (642, 270, 13), (1274, 672, 31), (256, 563, 19), (379, 343, 22), (702, 96, 16), (153, 1242, 16), (1235, 1136, 23), (228, 996, 14), (752, 1131, 16), (1248, 898, 17), (117, 784, 23), (986, 583, 14), (572, 1260, 16), (565, 1623, 17), (305, 1103, 16), (385, 1218, 26), (235, 375, 20), (1530, 670, 24), (1494, 1208, 18), (299, 1226, 17), (857, 1413, 13), (688, 608, 16), (334, 1444, 22), (988, 1181, 13), (909, 1144, 20), (910, 1663, 22), (1007, 282, 18), (1033, 508, 15), (841, 339, 18), (1258, 1238, 25), (1158, 1014, 17), (376, 977, 13), (1496, 855, 23), (726, 328, 14), (969, 825, 13), (1418, 748, 17), (606, 657, 20), (1308, 1132, 19), (834, 70, 15), (396, 1500, 17), (633, 817, 15), (1509, 966, 14), (1139, 1205, 14), (1234, 263, 14), (360, 410, 13), (1288, 723, 15), (798, 625, 24), (1303, 381, 20), (175, 1162, 23), (789, 84, 18), (1052, 139, 15), (333, 1250, 17), (1059, 398, 13), (831, 1049, 14), (1208, 1446, 16), (1394, 961, 20), (1254, 292, 19), (1275, 337, 17), (1097, 129, 18), (1242, 1302, 15), (1095, 1028, 23), (265, 491, 27), (1127, 1150, 20), (626, 892, 15), (1195, 1321, 17), (1190, 906, 14), (746, 1535, 18), (381, 921, 13), (1032, 1199, 12), (585, 1471, 14), (1215, 372, 28), (1174, 838, 16), (1285, 585, 13), (782, 1370, 16), (115, 972, 12), (994, 681, 12), (203, 1221, 16), (1171, 247, 14), (431, 418, 14), (1521, 462, 21), (686, 1627, 12), (930, 380, 13), (256, 723, 16), (1432, 1401, 13), (1106, 485, 13), (669, 694, 14), (921, 412, 13), (367, 275, 14), (761, 177, 16), (1068, 1194, 16), (502, 163, 24), (472, 783, 32), (801, 981, 12), (1426, 1440, 20), (419, 1548, 15), (615, 787, 19), (545, 1338, 24), (441, 704, 13), (458, 730, 13), (982, 173, 32), (1482, 906, 15), (1360, 449, 14), (799, 816, 13), (1209, 1597, 28), (796, 1136, 18), (319, 871, 19), (279, 1188, 17), (306, 777, 20), (855, 656, 16), (1086, 911, 16), (706, 1244, 17), (393, 1436, 13), (1067, 1343, 24), (1561, 1276, 15), (1388, 246, 17), (1457, 1199, 13), (352, 631, 20), (770, 721, 13), (801, 1221, 14), (236, 1298, 23), (525, 886, 22), (1481, 579, 17), (499, 1484, 18), (1041, 545, 13), (359, 870, 18), (1383, 719, 21), (1199, 173, 14), (141, 618, 13), (837, 1686, 13), (685, 1387, 16), (1254, 1049, 12), (1220, 466, 14), (768, 1002, 14), (388, 435, 16), (1425, 993, 21), (680, 812, 13), (1523, 902, 23), (481, 1575, 18), (485, 976, 16), (353, 1078, 15), (942, 1091, 19), (1571, 989, 13), (1471, 472, 24), (1604, 863, 24), (1599, 760, 23), (264, 892, 19), (421, 1275, 17), (578, 804, 18), (1308, 1093, 18), (1235, 520, 17), (1053, 276, 14), (1138, 1116, 14), (1604, 690, 15), (498, 292, 30), (503, 1540, 17), (666, 899, 24), (473, 572, 27), (336, 438, 12), (539, 232, 27), (1454, 843, 19), (520, 1047, 13), (1505, 497, 15), (1399, 1471, 18), (960, 1638, 20), (1087, 1487, 30), (856, 1146, 18), (408, 307, 14), (1604, 1127, 22), (755, 961, 16), (1376, 587, 16), (950, 63, 15), (752, 63, 14), (1162, 1089, 15), (1346, 1173, 31), (498, 674, 18), (1472, 651, 18), (300, 294, 26), (1319, 1434, 31), (849, 1645, 24), (1205, 1278, 22), (792, 1052, 13), (97, 873, 13), (265, 1358, 14), (623, 370, 17), (744, 1601, 15), (1123, 1397, 13), (1236, 717, 14), (745, 228, 22), (377, 1100, 15), (1367, 758, 14), (939, 1474, 15), (322, 656, 13), (1400, 1141, 16), (1263, 818, 18), (1068, 90, 19), (540, 1068, 14), (1141, 633, 17), (824, 114, 14), (994, 1220, 22), (372, 1323, 13), (1148, 1555, 17), (1058, 325, 13), (823, 1431, 14), (1352, 378, 15), (614, 1364, 16), (1500, 1164, 17), (1178, 319, 14), (1324, 1261, 18), (1342, 783, 14), (1642, 1101, 19), (645, 656, 13), (837, 1387, 16), (1547, 1241, 18), (367, 591, 13), (1176, 1241, 12), (670, 1247, 13), (438, 188, 27), (818, 387, 13), (684, 1030, 21), (525, 979, 14), (522, 1596, 18), (935, 822, 15), (1357, 817, 15), (502, 1195, 17), (466, 688, 14), (923, 301, 14), (1351, 409, 15), (565, 655, 12), (804, 1683, 15), (1061, 219, 13), (823, 1560, 16), (91, 631, 18), (641, 1641, 15), (591, 1022, 14), (415, 1378, 15), (67, 975, 24), (251, 1045, 24), (639, 415, 14), (1005, 1272, 15), (470, 1265, 22), (603, 436, 17), (935, 908, 13), (160, 780, 13), (1570, 686, 16), (1642, 1036, 15), (1368, 1266, 16), (1484, 1247, 14), (912, 446, 13), (438, 1173, 13), (521, 938, 18), (872, 1511, 17), (694, 1205, 15), (59, 805, 15), (156, 933, 20), (1001, 371, 14), (1368, 1108, 17), (724, 153, 14), (1015, 244, 14), (1277, 627, 14), (1341, 964, 17), (1201, 535, 13), (658, 1162, 12), (1373, 1220, 13), (106, 596, 15), (369, 1048, 18), (1420, 1366, 14), (253, 674, 19), (1515, 710, 18), (1098, 1174, 13), (428, 753, 13), (892, 104, 15), (1614, 817, 14), (1228, 970, 19), (1417, 315, 12), (736, 724, 17), (888, 237, 13), (626, 1257, 20), (759, 689, 14), (1550, 1050, 15), (233, 533, 18), (444, 1490, 16), (998, 842, 12), (382, 1395, 15), (468, 1079, 15), (540, 312, 13), (1459, 975, 16), (829, 735, 13), (978, 998, 16), (1022, 1424, 13), (822, 783, 13), (493, 366, 20), (348, 312, 18), (771, 502, 16), (1098, 388, 18), (1257, 1382, 14), (1515, 1066, 19), (750, 1087, 13), (1623, 564, 13), (1669, 718, 15), (349, 250, 14), (486, 437, 13), (1322, 221, 12), (1097, 844, 13), (1149, 1233, 14), (893, 671, 17), (263, 603, 13), (799, 849, 13), (1265, 390, 13), (1484, 1355, 15), (982, 1520, 14), (1520, 388, 15), (445, 1559, 13), (319, 960, 13), (826, 1609, 14), (1242, 592, 13), (1188, 1409, 15), (193, 880, 17), (638, 326, 13), (724, 606, 18), (989, 1607, 15), (548, 771, 18), (929, 1420, 19), (294, 731, 16), (1014, 325, 12), (1524, 770, 15), (614, 1176, 15), (511, 545, 15), (535, 1146, 16), (665, 388, 15), (534, 643, 15), (458, 1007, 16), (956, 1682, 13), (1211, 225, 18), (54, 885, 18), (918, 1615, 23), (811, 891, 14), (1035, 1090, 15), (407, 1066, 14), (818, 170, 21), (872, 970, 15), (1376, 1381, 13), (388, 528, 14), (146, 813, 12), (1103, 1320, 13), (726, 1384, 13), (1120, 580, 16), (577, 414, 15), (784, 147, 18), (1316, 822, 14), (1211, 803, 19), (857, 369, 12), (621, 159, 17), (1132, 905, 15), (1127, 1634, 14), (1036, 1160, 13), (1205, 503, 13), (683, 1445, 15), (1156, 1265, 14), (258, 1002, 15), (1429, 951, 15), (124, 572, 14), (746, 1312, 13), (299, 1143, 17), (1423, 414, 12), (1122, 998, 15), (477, 875, 13), (511, 1428, 17), (918, 1328, 20), (1201, 433, 21), (1551, 941, 15), (1310, 1524, 13), (316, 464, 14), (1423, 277, 18), (1082, 583, 14), (1178, 631, 15), (1308, 520, 14), (1547, 1301, 13), (344, 943, 16), (192, 750, 15), (468, 1170, 14), (874, 341, 13), (710, 910, 13), (1659, 951, 15), (443, 1403, 14), (887, 769, 14), (916, 165, 16), (578, 987, 15), (696, 536, 13), (850, 1534, 12), (246, 1222, 14), (1182, 876, 14), (1465, 1280, 13), (1628, 668, 13), (726, 1659, 15), (703, 65, 13), (1088, 722, 13), (1225, 1556, 15), (1162, 900, 13), (975, 705, 14), (1025, 400, 15), (1169, 1530, 14), (1473, 352, 16), (907, 214, 14), (1075, 1062, 15), (702, 283, 13), (1274, 878, 14), (603, 1531, 21), (1303, 253, 19), (1650, 881, 15), (1684, 887, 17), (298, 1381, 13), (860, 439, 14), (1141, 726, 21), (1070, 630, 13), (82, 677, 21), (1112, 89, 16), (991, 1300, 12), (633, 1009, 13), (1254, 1579, 18), (409, 509, 13), (1137, 376, 17), (1361, 218, 13), (1444, 1154, 15), (359, 1016, 14), (1194, 119, 14), (758, 1039, 12), (1642, 698, 14), (677, 1655, 13), (917, 764, 14), (987, 911, 13), (1574, 788, 13), (1519, 422, 15), (1005, 1392, 15), (974, 275, 14), (1233, 331, 15), (509, 1006, 14), (927, 534, 17), (1282, 1356, 16), (433, 808, 13), (1269, 1280, 16), (271, 1113, 16), (1309, 489, 12), (876, 1097, 14), (668, 359, 13), (662, 443, 13), (969, 1465, 14), (842, 1300, 14), (1095, 424, 16), (276, 1320, 13), (1325, 331, 17), (1223, 666, 17), (673, 758, 12), (222, 1250, 14), (666, 76, 14), (1296, 204, 17), (1494, 811, 18), (1065, 487, 16), (143, 967, 14), (187, 1123, 14), (1031, 73, 14), (1437, 340, 13), (1123, 1297, 14), (494, 247, 13), (1382, 999, 14), (1566, 637, 12), (960, 522, 16), (428, 359, 13), (353, 836, 15), (1384, 843, 15), (1361, 1410, 13), (1248, 1002, 13), (1638, 1147, 14), (243, 1395, 14), (843, 520, 15), (1591, 553, 17), (1193, 988, 16), (801, 741, 14), (1072, 363, 13), (657, 1608, 16), (181, 693, 13), (614, 618, 14), (750, 1398, 13), (1634, 1066, 13), (860, 91, 13), (747, 306, 13), (400, 739, 14), (116, 1017, 17), (936, 478, 14), (1539, 1339, 14), (979, 614, 15), (416, 1462, 15), (823, 1359, 13), (1175, 462, 16), (391, 948, 15), (526, 353, 13), (1032, 114, 15), (1095, 1205, 12), (537, 1505, 13), (334, 685, 16), (1505, 640, 12), (1160, 1165, 13), (1380, 619, 14), (580, 1357, 14), (544, 1115, 13), (794, 1267, 16), (619, 1081, 17), (905, 1571, 14), (506, 849, 14), (364, 222, 13), (1139, 283, 14), (1464, 879, 14), (196, 1193, 12), (1373, 654, 13), (1493, 540, 16), (1052, 578, 14), (584, 730, 14), (711, 1484, 14), (1076, 994, 13), (721, 940, 14), (1243, 1491, 13), (1538, 980, 14), (880, 1180, 13), (962, 1048, 13), (1230, 132, 13), (1397, 1201, 14), (1133, 1583, 13), (1040, 1591, 16), (607, 1452, 12), (454, 613, 15), (290, 536, 13), (798, 420, 14), (1665, 1056, 13), (221, 416, 22), (1177, 1625, 13), (374, 1462, 13), (565, 1089, 14), (1010, 867, 13), (617, 88, 13), (1223, 1092, 14), (645, 298, 13), (185, 788, 12), (1649, 621, 14), (994, 813, 13), (882, 180, 15), (762, 1572, 15), (249, 819, 13), (802, 1646, 13), (1590, 659, 15), (1064, 1228, 14), (437, 549, 13), (294, 661, 13), (62, 851, 13), (355, 1359, 13), (1160, 1053, 13), (1576, 1022, 12), (787, 577, 12), (1371, 1035, 15), (1503, 362, 14), (597, 1392, 13), (545, 1291, 15), (938, 1185, 13), (297, 1287, 15), (1312, 601, 15), (611, 404, 12), (650, 182, 13), (837, 262, 15), (1185, 280, 13), (1561, 600, 13), (764, 447, 15), (863, 62, 14), (1506, 1384, 13), (346, 903, 17), (661, 488, 16), (725, 1146, 12), (545, 853, 14), (1217, 1395, 15), (1552, 484, 13), (994, 218, 13), (253, 1333, 12), (738, 1466, 13), (1232, 414, 14), (811, 353, 14), (227, 887, 15), (720, 1515, 13), (717, 1625, 17), (1202, 255, 12), (1061, 1449, 14), (1588, 717, 14), (1453, 436, 13), (433, 837, 13), (872, 1678, 15), (611, 917, 13), (515, 430, 13), (986, 1680, 13), (961, 670, 13), (802, 1393, 14), (604, 1571, 13), (246, 629, 16), (520, 703, 15), (130, 842, 13), (580, 945, 14), (1040, 801, 17), (780, 1466, 14), (689, 1339, 13), (356, 557, 13), (342, 1283, 15), (1206, 718, 12), (714, 1290, 15), (1043, 1518, 13), (853, 182, 13), (1418, 837, 13), (1398, 523, 13), (704, 1539, 14), (1519, 1258, 14), (1620, 901, 12), (191, 1252, 14), (783, 54, 12), (459, 362, 13), (1147, 103, 13), ]

    sx = mask.shape[1] / float(circle_image_width)
    sy = mask.shape[0] / float(circle_image_height)
    im = np.zeros((circle_image_height, circle_image_width, 3), dtype=np.uint8)
    for circle in circles:
        cx, cy, r = circle
        color = fg_color if mask[int(cy*sy),int(cx*sx)] > 127 else bg_color
        cv2.circle(im, (int(cx),int(cy)), int(r), color, -1, cv2.LINE_AA)
    return im

def ishihara_plate_dichromacy(deficiency: simulate.Deficiency, 
                              label: str = None,
                              lms_model: convert.LMSModel = convert.LMSModel_sRGB_SmithPokorny75()):
    """Generate an image "plate" with several Ishihara-like images on in.

    The algorithm samples colors in the LMS space and generate confusion lines
    for it. Then for each line, it picks the two extreme colors on it and
    generates an Ishihara-like circle image with one color as the foreground (a
    number) and the other one as the background. This allows to evaluate the CVD
    kind, by checking on which plate the numbers get harder to read, if any.
    """
    try: import cv2
    except ImportError:
        sys.stderr.write("OpenCV is required for ishihara_image: `pip install opencv-python'\n")
        sys.exit (1)

    from daltonlens import geometry

    lms_yellow = lms_model.LMS_from_linearRGB @ np.array([1,1,0])
    lms_blue = lms_model.LMS_from_linearRGB @ np.array([0,0,1])
    U = lms_yellow # - lms_black == 0
    V = lms_blue # - lms_black == 0
    
    mask_images_path = Path(__file__).parent.absolute() / "data"
    assert mask_images_path.exists()

    random.seed()

    width = 128
    height = 128

    # Generate a set of confusion lines by walking along the diagonal plane
    images = []
    lms_seeds = itertools.product([0.1,0.25,0.5,0.75,0.9], [0.1,0.25,0.5,0.75,0.9])
    for uv_plane in lms_seeds:
        lms_color = uv_plane @ np.array([U,V])
        segment = geometry.lms_confusion_segment(lms_color, lms_model, deficiency)
        p1, p2 = segment
        # p1 and p2 are the extremes of the segment. A CVD with full severity would
        # not distinguish these two colors. A CVD with half severity would distinguish
        # them, but not the two colors situated half-way along the segment, etc.
        # So the severity is simulated by scaling the distance between the extreme points.
        # |********| -> severity of 1,   LMS distance between the points = 8
        # **|****|** -> severity of 0.5, LMS distance between the points = 4
        # ****||**** -> severity of 0,   LMS distance between the points = 0
        norm = np.linalg.norm(p2-p1)
        d = utils.normalized(p2-p1)
        dist = np.linalg.norm(p2-p1)
        srgb_1,srgb_2 = convert.sRGB_from_linearRGB(convert.apply_color_matrix(np.array([p1, p2]), lms_model.linearRGB_from_LMS))
        # Leave 0 and 4 out, we don't have nice masks for them
        n = random.choice([1,2,3, 5,6,7,8,9])
        mask_file = str(mask_images_path / "mask") + f"_{n}.png"
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) if n > 0 else np.zeros((256,256,1))
        if mask is None:
            print(f"ERROR: could not load the mask from {mask_file}, does the file exist?")
            return None
        im = ishihara_image(srgb_1*255.0, srgb_2*255.0, mask)
        im = cv2.resize(im, (width,height))
        images.append(im)

    # 5*5 total
    n_per_row = 5
    hstacked_images = []
    for i in range(0, len(images), n_per_row):
        hstacked_images.append(np.hstack(images[i:i+n_per_row]))
    plate_image = np.vstack(hstacked_images)

    if label:
        text_im = np.zeros((64, plate_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(text_im, label, (16,32), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220,220,220), 1, cv2.LINE_AA)
        plate_image = np.vstack([text_im, plate_image])

    return plate_image

def simulator_ishihara_plate(simulator: simulate.Simulator,
                             deficiency: simulate.Deficiency, 
                             severity: float = 1.0,
                             label: str = None,
                             lms_model: convert.LMSModel = convert.LMSModel_sRGB_SmithPokorny75()):
    """Generate an image "plate" with several Ishihara-like images on in.

    This version can handle various severities and evaluate a specific
    simulator. It works by first applying the CVD simulation of all the possible
    RGB values. Then for a set of reference colors, it checks what are the most
    different color that projected to the same color after applying the
    simulation. Then it uses these most different color pairs to generate some
    Ishihara-like images. The severity of the observer is given by the first
    value where he can't read any number. The kind of CVD is given by the
    deficiency where the severity is highest.

    For example, someone who:
        - Cannot see any number with PROTAN and severity < 0.5
        - Cannot see any number with DEUTAN and severity < 0.7
        - Cannot see any number with TRITAN and severity < 0.1
    is probably a mild-deutan according to that simulator, with severity 0.7.
    """
    try: import cv2
    except ImportError:
        sys.stderr.write("OpenCV is required for ishihara_image: `pip install opencv-python'\n")
        return None

    try: import colour
    except ImportError:
        sys.stderr.write("colour is required for simulator_ishihara_plate: `pip install colour-science'\n")
        return None

    from daltonlens import geometry

    mask_images_path = Path(__file__).parent.absolute() / "data"
    assert mask_images_path.exists()

    width = 256
    height = 256

    # Sample the entire linear RGB color space, but with only 64 values per axis instead of 256
    # This is enough to evaluate the algorithm and runs much faster.
    num_steps = 64
    same_color_threshold = 6.0 / 256.0
    axis_values = np.linspace(0.0, 1.0, num_steps)
    mesh = np.meshgrid(axis_values, axis_values, axis_values)
    grid = np.stack(mesh, axis=-1).astype(float)
    # grid.shape is (64,64,64,3): a volume with one color per cell, covering the entire RGB gamut.
    
    # Compute a set of reference colors to test. Basically 0.1 or 0.9 for each channel.
    per_axis_seed = [0.2, 0.8]
    rgb_refs = []
    for r,g,b in itertools.product(per_axis_seed, per_axis_seed, per_axis_seed):
        rgb_refs.append(grid[int(r*num_steps), int(g*num_steps), int(b*num_steps)])
    rgb_refs = np.array(rgb_refs)
    rgb_refs_cvd = simulator._simulate_cvd_linear_rgb(rgb_refs.reshape(1,-1,3), deficiency, severity=severity).reshape(-1,3)

    grid = grid.reshape((grid.shape[0], -1, 3))
    # grid is now a 2D RGB image with shape (64, 64x64=4096, 3)
    grid_cvd = simulator._simulate_cvd_linear_rgb(grid, deficiency, severity=severity)
    np.set_printoptions(precision=3, suppress=True)

    # Now switch to 1D, we don't care about having an image.
    grid_cvd = grid_cvd.reshape(-1, 3)
    grid = grid.reshape(-1, 3)

    linearRGB_to_Lab = lambda im_rgb: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(im_rgb, apply_cctf_decoding=False))

    # Manage a subset of colors in the grid.
    ColorSubset = namedtuple('ColorSubset', ['orig_rgb', 'orig_lab', 'cvd_rgb', 'cvd_lab'])
    def filterSubset (subset: ColorSubset, indices):
        orig_rgb = subset.orig_rgb[indices]
        cvd_rgb = subset.cvd_rgb[indices]
        orig_lab = subset.orig_lab[indices] if subset.orig_lab is not None else None
        cvd_lab = subset.cvd_lab[indices] if subset.cvd_lab is not None else None
        return ColorSubset(orig_rgb, orig_lab, cvd_rgb, cvd_lab)

    colorPairs = []

    for rgb_ref, rgb_ref_cvd in zip(rgb_refs, rgb_refs_cvd):
        ref_lab = linearRGB_to_Lab(rgb_ref)
        ref_cvd_lab = linearRGB_to_Lab(rgb_ref_cvd)

        # Perceptual distance between the simulated colour and the original one.
        # This will typically have the largest distance when the severity is
        # low.
        dE_refRgb_refCvd = colour.delta_E(ref_lab, ref_cvd_lab)

        current_set = ColorSubset (grid, None, grid_cvd, None)

        # We also need to check if two colors that used to be different now fall
        # on the same color. This will typically happen with full severity, when
        # two colors on each side of the projection plane collapse to a single
        # location on the plane. This makes an even bigger difference than the
        # simulated color vs original color.
        diff_abs = np.abs(current_set.cvd_rgb - rgb_ref_cvd)
        max_diff = np.max(diff_abs, axis=-1)
        indices = max_diff < same_color_threshold
        current_set = filterSubset(current_set, indices)
        
        orig_rgb = grid[indices]
        cvd_rgb = grid_cvd[indices]
        current_set = current_set._replace(orig_lab=linearRGB_to_Lab(current_set.orig_rgb))
        current_set = current_set._replace(cvd_lab=linearRGB_to_Lab(current_set.cvd_rgb))

        # Now we know that all these colors are similar once transformed with CVD.
        dE_refCvd_cvd = colour.delta_E(ref_cvd_lab, current_set.cvd_lab)
        refined_indices = dE_refCvd_cvd < 1.0
        current_set = filterSubset(current_set, refined_indices)

        dE_refRgb_orig = colour.delta_E(ref_lab, current_set.orig_lab)
        indices_that_changed = dE_refRgb_orig > 2.0
        current_set = filterSubset(current_set, indices_that_changed)
        dE_refRgb_orig = dE_refRgb_orig[indices_that_changed]
        
        # Pick the color pair that is the most different between all the source
        # colors that project to this CVD simulated colors.
        if dE_refRgb_orig.size == 0 or dE_refRgb_refCvd > np.max(dE_refRgb_orig):
            colorPairs.append((rgb_ref, rgb_ref_cvd, dE_refRgb_refCvd))
        else:
            k = np.argmax(dE_refRgb_orig)
            colorPairs.append((rgb_ref, current_set.orig_rgb[k], dE_refRgb_orig[k]))

    # Keep the top N pairs.
    colorPairs = sorted(colorPairs, key=lambda cp: -cp[2])
    colorPairs = colorPairs[0:3]
    images = []
    random.seed()
    for cp in colorPairs:
        # Don't try to render images where the distance is too low.
        # Theoretically the just noticeable distance is 1, but we take a margin
        # to avoid tests that are too hard even for a normal observer.
        if cp[2] < 2.0:
            im = np.full((height, width, 3), 127, dtype=np.uint8)
        else:
            n = random.choice([1,2,3, 5,6,7,8,9])
            mask_file = str(mask_images_path / "mask") + f"_{n}.png"
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) if n > 0 else np.zeros((256,256,1))
            if mask is None:
                print(f"ERROR: could not load the mask from {mask_file}, does the file exist?")
                return None
            srgb0,srgb1 = convert.sRGB_from_linearRGB(np.array([cp[0], cp[1]]))
            im = ishihara_image(srgb0*255.0, srgb1*255.0, mask)
            im = cv2.resize(im, (width,height))
        images.append(im)
    
    im = np.hstack(images)
    if label:
        text_im = np.zeros((64, im.shape[1], 3), dtype=np.uint8)
        cv2.putText(text_im, label, (16,32), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220,220,220), 1, cv2.LINE_AA)
        im = np.vstack([text_im, im])
    return im
