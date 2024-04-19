from math import sqrt


# https://en.wikipedia.org/wiki/Euclidean_distance


plot1 = [1, 3]
plot2 = [2, 5]

#                                 1 **2 = 1                  -2 **2 = 4
euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )

print(euclidean_distance)

