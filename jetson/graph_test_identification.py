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
#
# test_names.append('mag5_uv_cat_mag5')
# in_scene_good_verify.append(1000 / 10)
# in_triangle_good_verify.append(993 / 10)
# exact_good_verify.append(993 / 10)
# percent_identified_stars.append(99.17)
# avg_time.append(0.0229)

test_names.append('mag4 cat5')
in_scene_good_verify.append(972 / 10)
in_triangle_good_verify.append(929 / 10)
exact_good_verify.append(929 / 10)
percent_identified_stars.append(95.03)
avg_time.append(0.0079)

test_names.append('mag5 cat5')
in_scene_good_verify.append(996 / 10)
in_triangle_good_verify.append(961 / 10)
exact_good_verify.append(960 / 10)
percent_identified_stars.append(95.89)
avg_time.append(0.0226)

# test_names.append('mag556_uv_cat_mag5')
test_names.append('mag5.56 cat5')
in_scene_good_verify.append(922 / 10)
in_triangle_good_verify.append(904 / 10)
exact_good_verify.append(904 / 10)
percent_identified_stars.append(32.2)
avg_time.append(0.0401)

# test_names.append('mag556_uv_cat_mag6')
test_names.append('mag5.56 cat6')
in_scene_good_verify.append(984 / 10)
in_triangle_good_verify.append(979 / 10)
exact_good_verify.append(979 / 10)
percent_identified_stars.append(95.47)
avg_time.append(2.193535611206003)

test_names.append('esa cat5')
in_scene_good_verify.append(90)
in_triangle_good_verify.append(90)
exact_good_verify.append(90)
percent_identified_stars.append(44.76)
avg_time.append(0.0871)

test_names.append('esa cat6')
in_scene_good_verify.append(66)
in_triangle_good_verify.append(66)
exact_good_verify.append(66)
percent_identified_stars.append(56.51)
avg_time.append(4.9)


df = pd.DataFrame(
    {
        'Scenario': test_names,
        '% scenes: star in scene': in_scene_good_verify,
        '% scenes: star to star': exact_good_verify,
        '% overall identified stars': percent_identified_stars,
        'Average time (right)': avg_time,
    },
)
df.set_index("Scenario", drop=True, inplace=True)

fig = plt.figure()  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes

width = 0.1

ax.set_ylabel('Percent [%]')

ax = df.plot(kind='bar', secondary_y=['Average time (right)'])
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax.right_ax.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center left')
plt.title('Scenes time computation and input stars number')
ax.right_ax.set_ylabel('Time [s]')

plt.show()
