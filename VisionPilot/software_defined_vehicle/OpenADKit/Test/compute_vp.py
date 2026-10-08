import numpy as np

left_fit = np.array([
0.0126787557,
-2.94695562,
208.096418
])

right_fit = np.array([
-0.00339083,
2.48173375,
-58.1098334
])

# Horizon region
y0 = 62

# lane positions
xL = left_fit[0]*y0*y0 + left_fit[1]*y0 + left_fit[2]
xR = right_fit[0]*y0*y0 + right_fit[1]*y0 + right_fit[2]

# derivatives
mL = 2*left_fit[0]*y0 + left_fit[1]
mR = 2*right_fit[0]*y0 + right_fit[1]

# tangent lines:
# x = m*y + b

bL = xL - mL*y0
bR = xR - mR*y0

y_vp = (bR - bL)/(mL - mR)
x_vp = mL*y_vp + bL

print("VP:")
print("x =", x_vp)
print("y =", y_vp)