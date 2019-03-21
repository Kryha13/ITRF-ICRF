import csv
# pozyskanie potrzebnych danych z pliku .sp3

# dane satelity

with open("dane.txt") as f, open("PG06.txt") as f2:
    st = set(line.rstrip() for line in f2)
    # r = csv.reader(f, delimiter=" ")
    data = [line.rstrip() for line in f if line.rsplit()[0] in st]   #
    # data = [row for row in r if row[0] in st]   - kazdy element jako oddzielny element listy
    print(data)


with open('danegot.txt', 'w') as file_handler:
    for item in data:
        file_handler.write("{}\n".format(item))


import numpy as np

g = np.loadtxt('danegot.txt', usecols=(1, 2, 3))

# epoki


with open("dane.txt") as f, open("ep") as f2:
    st = set(line.rstrip() for line in f2)
    # r = csv.reader(f, delimiter=" ")
    data = [line.rstrip() for line in f if line.rsplit()[0] in st]   #
    # data = [row for row in r if row[0] in st]   - kazdy element jako oddzielny element listy
    print(data)


with open('epoki', 'w') as file_handler:
    for item in data:
        file_handler.write("{}\n".format(item))