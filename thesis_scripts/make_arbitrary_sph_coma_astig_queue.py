from pathlib import Path
import numpy as np

from iris.data import PersistentQueue

np.random.seed(1234)


def make_mag_xy_random_values():
    fudge1 = np.random.normal(loc=1, scale=0.5, size=500)
    fudge2 = np.random.normal(loc=1, scale=0.5, size=500)
    mid_factor, high_factor = -0.5, 0.5 ** 2
    low_value = np.random.random(500) * 0.1725
    mid_value, high_value = low_value * mid_factor * fudge1, low_value * high_factor * fudge2

    angles = np.radians(np.random.random(size=500) * 180)
    x_factor, y_factor = np.cos(angles), np.sin(angles)

    return low_value, mid_value, high_value, x_factor, y_factor


low_value, mid_value, high_value, x_factor, y_factor = make_mag_xy_random_values()

xmag_low = low_value * x_factor
ymag_low = low_value * y_factor

xmag_mid = mid_value * x_factor
ymag_mid = mid_value * y_factor

xmag_high = high_value * x_factor
ymag_high = high_value * y_factor

z4 = np.zeros(500)
z5 = xmag_low
z6 = ymag_low
z12 = xmag_mid
z13 = ymag_mid
z21 = xmag_high
z22 = ymag_high

low_value, mid_value, high_value, x_factor, y_factor = make_mag_xy_random_values()
z9, z16, z25 = low_value, mid_value, high_value

low_value, mid_value, high_value, x_factor, y_factor = make_mag_xy_random_values()

xmag_low = low_value * x_factor
ymag_low = low_value * y_factor

xmag_mid = mid_value * x_factor
ymag_mid = mid_value * y_factor

xmag_high = high_value * x_factor
ymag_high = high_value * y_factor

z7 = xmag_low
z8 = ymag_low
z14 = xmag_mid
z15 = ymag_mid
z23 = xmag_high
z24 = ymag_high

# stack all truths together and split them into 5 chunks for 5 workers per datatype
alltruth = np.stack((z4, z5, z6, z7, z8, z9, z12, z13, z14, z15, z16, z21, z22, z23, z24, z25), axis=1)
truths = list(alltruth)

p = Path(__file__).parent / '..' / '..' / 'data' / 'arbitrary-coma-astigmatism'
q = PersistentQueue(p / 'queue.pkl', overwrite=True)
q.put_many(truths)
