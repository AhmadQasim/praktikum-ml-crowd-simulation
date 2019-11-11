import matplotlib.pyplot as plt

x = [15, 20, 25, 30]
y = [95.84111313472663, 115.96484544233599, 150.15237367987424, 180.66460723778266]

plt.plot(x, y)
plt.title('Entropy by Number of Pedestrians (True: GNM, Simulator: OSM)')
plt.xlabel('Pedestrians')
plt.ylabel('Entropy')
plt.legend()
plt.savefig('./report/pictures/task5_gnm_osm.png')