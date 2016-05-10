__author__ = 'Szymon Michalski'
# 1.
a_roi = 5
i_threshold = 5

for x, y in image:
    if I(x, y) > i_threshold:
        x_start = x - (a_roi - 1)/2
        y_start = y - (a_roi - 1)/2
        x_end = x_start + a_roi
        y_end = y_start + a_roi
# 2.
        if x_start < 0 or y_start < 0:
            pass
# 3.
i_bottom = 0
i_top = 0
i_left = 0
i_right = 0
i_border = 0
# 4. - normalization
# I^(x, y)= I(x, y) - I_border

# 5.

b =

x_cm =

y_cm =

# 6.

# 7.

u
