from libtailor import *


tot_frame = 30
raw_height = 240
raw_weight = 416
new_height = 224
new_width = 416
raw_filename = "RaceHorses_416x240_30.yuv"
new_filename = "TAIL.yuv"

for i in range(tot_frame):
    Y, U, V = read_frame(raw_filename, i, raw_height, raw_weight)
    Y = Y[0:new_height, 0:new_width]
    U = U[0:new_height >> 1, 0:new_width >> 1]
    V = V[0:new_height >> 1, 0:new_width >> 1]
    write_frame(new_filename, i, new_height, new_width, Y, U, V)