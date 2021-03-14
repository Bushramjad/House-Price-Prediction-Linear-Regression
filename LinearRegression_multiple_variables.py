import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# input
House_Details = {'area': [2104, 1416, 1534, 852],
                 'rooms': [5, 3, 3, 2],
                 'floors': [1, 2, 2, 1],
                 'age': [45, 40, 30, 46],
                 }

one = np.array([np.ones(len(House_Details))]).T

prices = np.array([460, 232, 315, 178])

# Scaled features
area_s, age_s, floor_s, room_s = np.array([])

# feature scaling
i = 0
for k in House_Details.items():
    area = House_Details.get("area")
    area_s = np.append(area_s, (area[i] - 852) / (2104 - 852))

    age = House_Details.get("age")
    age_s = np.append(age_s, (age[i] - 40) / (46 - 40))

    rooms = House_Details.get("rooms")
    room_s = np.append(room_s, (rooms[i] - 2) / (5 - 2))

    floors = House_Details.get("floors")
    floor_s = np.append(floor_s, (floors[i] - 1) / (2 - 1))
    i = i + 1

# Features matrix
X = np.array([area_s, room_s, floor_s, age_s]).T
X = np.hstack((one, X))

Y = (prices[:, np.newaxis])

# Applying the normal equation
theta = inv(X.T.dot(X)).dot(X.T).dot(Y)

y_pred = theta[0] + \
         theta[1] * area_s + \
         theta[2] * room_s + \
         theta[3] * floor_s + \
         theta[4] * age_s

i = 0
print('---------------------------------------')
for k, v in House_Details.items():
    area, rooms, floor, age = v
    print("{:<8} {:<8} {:<8} {:<8} {:<8}".format(k, area, rooms, floor, age))

print('---------------------------------------')
print('Price', end="    ")
for k in prices:
    print("{:<8}".format(int(y_pred[i])), end=" ")
    i = i + 1
print('\n---------------------------------------')

print(np.subtract(prices, y_pred))

plt.plot(area_s, prices, color='black')
plt.plot(area_s, y_pred, color='blue', linewidth=3)
plt.show()
