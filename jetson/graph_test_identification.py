import matplotlib.pyplot as plt
import pandas as pd


test_names = []
in_scene_good_verify = []
in_triangle_good_verify = []
exact_good_verify = []
percent_identified_stars = []
avg_time = []

# test_names.append('mag4_uv_cat_mag5')
# in_scene_good_verify.append(982 / 10)
# in_triangle_good_verify.append(982 / 10)
# exact_good_verify.append(982 / 10)
# percent_identified_stars.append(99.69)
# avg_time.append(0.009)

test_names.append('mag5_uv_cat_mag5')
in_scene_good_verify.append(1000 / 10)
in_triangle_good_verify.append(993 / 10)
exact_good_verify.append(993 / 10)
percent_identified_stars.append(99.17)
avg_time.append(0.0229)

test_names.append('mag556_uv_cat_mag5')
in_scene_good_verify.append(922 / 10)
in_triangle_good_verify.append(904 / 10)
exact_good_verify.append(904 / 10)
percent_identified_stars.append(32.2)
avg_time.append(0.0401)

test_names.append('mag556_uv_cat_mag6')
in_scene_good_verify.append(984 / 10)
in_triangle_good_verify.append(979 / 10)
exact_good_verify.append(979 / 10)
percent_identified_stars.append(95.47)
avg_time.append(2.193535611206003)

# test_names.append('mag4_xy_cat5')
# in_scene_good_verify.append(972 / 10)
# in_triangle_good_verify.append(929 / 10)
# exact_good_verify.append(929 / 10)
# percent_identified_stars.append(95.03)
# avg_time.append(0.0079)

test_names.append('mag5_xy_cat5')
in_scene_good_verify.append(996 / 10)
in_triangle_good_verify.append(961 / 10)
exact_good_verify.append(960 / 10)
percent_identified_stars.append(95.89)
avg_time.append(0.0226)

test_names.append('esa_xy_cat_mag5')
in_scene_good_verify.append(90)
in_triangle_good_verify.append(90)
exact_good_verify.append(90)
percent_identified_stars.append(44.76)
avg_time.append(0.0871)

test_names.append('esa_xy_cat_mag6')
in_scene_good_verify.append(66)
in_triangle_good_verify.append(66)
exact_good_verify.append(66)
percent_identified_stars.append(56.51)
avg_time.append(4.9)


df = pd.DataFrame(
    {
        'test_names': test_names,
        'in_scene_good_verify': in_scene_good_verify,
        'exact_good_verify': exact_good_verify,
        'percent_identified_stars': percent_identified_stars,
        'avg_time': avg_time,
    }
)
df.set_index("test_names", drop=True, inplace=True)

fig = plt.figure()  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes
ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

width = 0.1

df.in_scene_good_verify.plot(
    kind='bar', color='red', ax=ax, width=width, position=2)
df.exact_good_verify.plot(
    kind='bar', color='green', ax=ax, width=width, position=1)
df.percent_identified_stars.plot(
    kind='bar', color='blue', ax=ax, width=width, position=0)
df.avg_time.plot(
    kind='bar', color='black', ax=ax2, width=width, position=-1)

ax.set_ylabel('Percent [%]')
ax2.set_ylabel('Time [s]')

plt.show()
