#coding: UTF-8

import os, sys

if len(sys.argv) != 3:
    print 'Usage: ' + sys.argv[0] + ' <dir> <output>'
    sys.exit(-1)

dir = sys.argv[1]
files = os.listdir(dir)
files.sort()

fout = open(sys.argv[2], 'w')
for file in files:
    fout.write(os.path.abspath(dir + file) + '\n')
fout.close()
