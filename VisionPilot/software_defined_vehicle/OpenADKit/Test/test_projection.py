import numpy as np

fx = 360.8899
fy = 360.2564
cx = 418.3760
cy = 235.9731

h = 1.0
pitch = np.radians(-18.569597431099865)

# Point 10 m ahead on ground
X = 0
Z = 10

Xc = X

Yc = (
    -h*np.cos(pitch)
    - Z*np.sin(pitch)
)

Zc = (
    Z*np.cos(pitch)
    - h*np.sin(pitch)
)

u = fx*(Xc/Zc) + cx
v = fy*(Yc/Zc) + cy

print("Projected pixel:")
print("u =", u)
print("v =", v)