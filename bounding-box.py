#!/usr/bin/env python

import fileinput
import re


xs, ys, zs = [], [], []
for line in fileinput.input():
    m = re.match(r'^v\s+(?P<x>-?\d+(?:\.\d*)?)\s+(?P<y>-?\d+(?:\.\d*)?)\s+(?P<z>-?\d+(?:\.\d*)?)', line)
    if m:
        x, y, z = map(float, m.group('x', 'y', 'z'))
        xs.append(x)
        ys.append(y)
        zs.append(z)
if xs and ys and zs:
    min_x, max_x = min(xs), max(xs)
    range_x = max_x - min_x
    min_y, max_y = min(ys), max(ys)
    range_y = max_y - min_y
    min_z, max_z = min(zs), max(zs)
    range_z = max_z - min_z
    print 'x = %f .. %f (%f)' % (min_x, max_x, range_x)
    print 'y = %f .. %f (%f)' % (min_y, max_y, range_y)
    print 'z = %f .. %f (%f)' % (min_z, max_z, range_z)
    print 'x-y area = %f' % (range_x * range_y)
    print 'x-z area = %f' % (range_x * range_z)
    print 'y-z area = %f' % (range_y * range_z)
    print 'volume = %f' % (range_x * range_y * range_z)
