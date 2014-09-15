import os, sys

if len(sys.argv) != 6:
    print 'Usage: ' + sys.argv[0] + ' <bin> <param-path> <num> <facelist> <nonfacelist>'
    exit(-1)

bin = os.path.abspath(sys.argv[1])
dir = os.path.abspath(sys.argv[2])
num = int(sys.argv[3])
face = os.path.abspath(sys.argv[4])
nonface = os.path.abspath(sys.argv[5])

for ii in range(1, num + 1):
    param = dir + os.sep + 'mit_cbcl_' + str(ii) + '.param'
    os.system(bin + ' ' + param + ' ' + face + ' ' + nonface)
