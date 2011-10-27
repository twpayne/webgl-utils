#!/usr/bin/env python
# FIXME in theory we can also generate optimized textures at multiple level of detail

from __future__ import with_statement

from collections import defaultdict
from itertools import chain, combinations, izip, tee
import logging
import math
from operator import attrgetter
from optparse import OptionParser
import os
import os.path
import re
import sys

import Image
import ImageDraw
import numpy
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon


logger = logging.getLogger(os.path.basename(sys.argv[0]) if __name__ == '__main__' else __name__)


# http://docs.python.org/library/itertools.html

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def connected_components(nodes=set(), edges=set()):
    "yields sets of connected components"
    connections = defaultdict(set)
    nodes = set(nodes)
    for left, right in edges:
        nodes.add(left)
        nodes.add(right)
        connections[left].add(right)
        connections[right].add(left)
    while nodes:
        visited = set()
        tovisit = set((nodes.pop(),))
        while tovisit:
            tovisit2 = set()
            for node in tovisit:
                tovisit2.update(connections[node])
            visited.update(tovisit)
            tovisit = tovisit2 - visited
        nodes.difference_update(visited)
        yield visited


# http://code.activestate.com/recipes/442299-pack-multiple-images-of-different-sizes-into-one-i/

class PackNode(object):
    """
    Creates an area which can recursively pack other areas of smaller sizes into itself.
    """
    def __init__(self, area):
        #if tuple contains two elements, assume they are width and height, and origin is (0,0)
        if len(area) == 2:
            area = (0,0,area[0],area[1])
        self.area = area

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, str(self.area))

    def get_width(self):
        return self.area[2] - self.area[0]
    width = property(fget=get_width)

    def get_height(self):
        return self.area[3] - self.area[1]
    height = property(fget=get_height)

    def insert(self, area):
        if hasattr(self, 'child'):
            a = self.child[0].insert(area)
            if a is None: return self.child[1].insert(area)
            return a

        area = PackNode(area)
        if area.width <= self.width and area.height <= self.height:
            self.child = [None,None]
            self.child[0] = PackNode((self.area[0]+area.width, self.area[1], self.area[2], self.area[1] + area.height))
            self.child[1] = PackNode((self.area[0], self.area[1]+area.height, self.area[2], self.area[3]))
            return PackNode((self.area[0], self.area[1], self.area[0]+area.width, self.area[1]+area.height))


class Material(object):

    def __init__(self, name):
        self.name = name
        self.image = None
        self.sub_image1s = []


class SubImage1(object):

    def __init__(self, polygon, vtis):
        self.polygon = polygon
        self.vtis = vtis


class SubImage2(object):

    def __init__(self, material, image, transform):
        self.material = material
        self.image = image
        self.transform = transform
        self.uv = None

    @property
    def area(self):
        return self.image.size[0] * self.image.size[1]


def translate(tx, ty):
    a = numpy.identity(3)
    a[0][2] = tx
    a[1][2] = ty
    return a

def scale(sx, sy):
    a = numpy.identity(3)
    a[0][0] = sx
    a[1][1] = sy
    return a


def main(argv):
    logging.basicConfig(level=logging.DEBUG)
    parser = OptionParser()
    parser.add_option('-i', '--input', metavar='FILENAME', help='input object filename')
    parser.add_option('-o', '--output', metavar='FILENAME', help='object filename')
    parser.add_option('-m', '--margin', default=2, metavar='PIXELS', type=int)
    parser.add_option('-a', '--alignment', default=1, metavar='N', type=int)
    parser.add_option('-d', '--debug', action='store_true')
    options, args = parser.parse_args(argv[1:])
    if options.input is None or options.input == '-':
        lines = list(sys.stdin)
        input_dirname = '.'
    else:
        lines = list(open(options.input))
        input_dirname = os.path.dirname(options.input)
    if options.output is None or options.output == '-':
        output_dirname = '.'
        basename = obj
    else:
        output_dirname = os.path.dirname(options.output) or '.'
        basename = os.path.splitext(os.path.basename(options.output))[0]
    fieldss = [re.split(r'\s+', line.rstrip()) if not line.startswith('#') else None for line in lines]
    # Parse texture vertices and material libraries
    vts = [None]
    materials_by_name, material = {}, None
    for line, fields in enumerate(fieldss):
        if not fields:
            continue
        if fields[0] == 'vt':
            assert len(fields) == 3
            vts.append(Point(map(float, fields[1:])))
            continue
        if fields[0] == 'mtllib':
            assert len(fields) == 2
            for mtllib_line in open(os.path.join(input_dirname, fields[1])):
                mtllib_fields = re.split(r'\s+', mtllib_line.rstrip())
                if mtllib_fields[0] == 'newmtl':
                    assert len(mtllib_fields) == 2
                    assert mtllib_fields[1] not in materials_by_name
                    material = Material(mtllib_fields[1])
                    materials_by_name[mtllib_fields[1]] = material
                    continue
                if mtllib_fields[0] == 'map_Kd':
                    assert len(mtllib_fields) == 2
                    assert material is not None
                    assert material.image is None
                    filename = os.path.join(input_dirname, mtllib_fields[1])
                    material.image = Image.open(filename).transpose(Image.FLIP_TOP_BOTTOM)
                    material.image.filename = filename
                    continue
            continue
    # Assert that all texture vertices are valid
    assert all(0.0 <= vt.x <= 1.0 and 0.0 <= vt.y <= 1 for vt in vts[1:])
    materials = materials_by_name.values()
    logger.info('found %d materials and %d texture vertices' % (len(materials), len(vts) - 1))
    if options.debug:
        # Create ImageDraw objects for each material
        for material in materials:
            if material.image:
                material.image.draw = ImageDraw.Draw(material.image)
    # Determine the material used for each texture vertex and build the list of connected texture vertices
    material = None
    materials_by_vti = defaultdict(set)
    vti_edges = set()
    for fields in fieldss:
        if not fields:
            continue
        if fields[0] == 'usemtl':
            assert len(fields) == 2
            material = materials_by_name[fields[1]]
            continue
        if fields[0] == 'f':
            assert material is not None
            vtis = list(int(vti) for vti in (field.split('/')[1] for field in fields[1:]) if vti)
            if not vtis:
                continue
            for vti in vtis:
                materials_by_vti[vti].add(material)
            vti_edges.update(pairwise(vtis))
            if options.debug:
                # Mark up the debug image with the face (outlined in yellow)
                material.image.draw.polygon(list((vts[vti].x * material.image.size[0], material.image.size[1] * vts[vti].y) for vti in vtis), outline='#ff0')
            continue
    material_by_vti = {}
    for vti in sorted(materials_by_vti.keys()):
        if len(materials_by_vti[vti]) != 1:
            logger.warn('texture vertex %d is used in multiple materials: %s' % (vti, ', '.join(sorted(material.name for material in materials_by_vti[vti]))))
        material_by_vti[vti] = min(materials_by_vti[vti], key=attrgetter('name'))
    # Find adjacent areas of used texture (SubImage1s)
    for vtis in connected_components(edges=vti_edges):
        # Assert that all texture vertices are for the same material
        material = material_by_vti[next(iter(vtis))]
        assert all(material_by_vti[vti] == material for vti in vtis)
        # Calculate the bounding box of these texture vertices
        x0, y0, x1, y1 = MultiPoint(list(vts[vti] for vti in vtis)).bounds
        polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        # Create SubImage1
        sub_image1 = SubImage1(polygon, vtis)
        material.sub_image1s.append(sub_image1)
        if options.debug:
            # Mark up the debug image with the calculated SubImage1 (outlined in green)
            u0, v0, u1, v1 = sub_image1.polygon.bounds
            i0, j0 = u0 * material.image.size[0], v0 * material.image.size[1]
            i1, j1 = u1 * material.image.size[0], v1 * material.image.size[1]
            material.image.draw.rectangle(((i0, j0), (i1, j1)), outline='#0f0')
    logger.info('found %d SubImage1s' % sum(len(material.sub_image1s) for material in materials))
    # Find overlapping or adjacent SubImage1s in each material and merge them into SubImage2s
    sub_image2s = []
    sub_image2_by_vti = [None] * len(vts)
    for material in materials:
        if not material.image:
            continue
        sub_image1_edges = []
        for sub_image1a, sub_image1b in combinations(material.sub_image1s, 2):
            if sub_image1a.polygon.intersects(sub_image1b.polygon):
                sub_image1_edges.append((sub_image1a, sub_image1b))
        for sub_image1s in connected_components(nodes=material.sub_image1s, edges=sub_image1_edges):
            # Determine bounding box in texture coordinates
            u0, v0, u1, v1 = MultiPolygon([sub_image1.polygon for sub_image1 in sub_image1s]).bounds
            # Convert texture coordinates to pixel coordinates, rounding out
            i0, j0 = int(u0 * material.image.size[0]), int(v0 * material.image.size[1])
            i1, j1 = int(math.ceil(u1 * material.image.size[0])), int(math.ceil(v1 * material.image.size[1]))
            # Add margin
            i0, j0 = max(i0 - options.margin, 0), max(j0 - options.margin, 0)
            i1, j1 = min(i1 + options.margin, material.image.size[0]), min(j1 + options.margin, material.image.size[1])
            # Create a larger image if needed, to ensure alignment
            width, height = i1 - i0, j1 - j0
            if width % options.alignment or height % options.alignment:
                width = options.alignment * ((width + options.alignment - 1) // options.alignment)
                height = options.alignment * ((height + options.alignment - 1) // options.alignment)
                image = Image.new('RGB', (width, height))
                image.paste(material.image.crop((i0, j0, i1, j1)), (0, 0, i1 - i0, j1 - j0))
            else:
                image = material.image.crop((i0, j0, i1, j1))
            # Create SubImage2
            tx = -float(i0) / material.image.size[0]
            ty = -float(j0) / material.image.size[1]
            sx = material.image.size[0] / float(width)
            sy = material.image.size[1] / float(height)
            transform = numpy.dot(scale(sx, sy), translate(tx, ty))
            sub_image2 = SubImage2(material, image, transform)
            sub_image2s.append(sub_image2)
            # Associate this SubImage2 with its vtis
            for vti in chain(*(sub_image1.vtis for sub_image1 in sub_image1s)):
                assert vti not in sub_image2_by_vti
                sub_image2_by_vti[vti] = sub_image2
            if options.debug:
                # Mark up the debug image with the calculated SubImage2 (outlined in red)
                material.image.draw.rectangle(((i0, j0), (i1, j1)), outline='#f00')
    logger.info('merged SubImage1s into %d SubImage2s' % len(sub_image2s))
    # Sort SubImage2s by height, then width to assist packing heuristic
    sub_image2s = sorted(sub_image2s, key=lambda sub_image2: (sub_image2.image.size[1], sub_image2.image.size[0]), reverse=True)
    # Calculate packing of SubImage2s
    area = sum(sub_image2.area for sub_image2 in sub_image2s)
    logger.info('texture area is %d pixels' % area)
    size = 1 << int(math.ceil(math.log(math.sqrt(area)) / math.log(2)))
    logger.info('minimum texture size is %dx%d' % (size, size))
    while True:
        tree = PackNode((size, size))
        for sub_image2 in sub_image2s:
            sub_image2.uv = tree.insert(sub_image2.image.size)
            if sub_image2.uv is None:
                break
            sx = float(sub_image2.uv.width) / size
            sy = float(sub_image2.uv.height) / size
            tx = float(sub_image2.uv.area[0]) / size
            ty = float(sub_image2.uv.area[1]) / size
            transform = numpy.dot(translate(tx, ty), scale(sx, sy))
            sub_image2.transform = numpy.dot(transform, sub_image2.transform)
        else:
            break
        size *= 2
    logger.info('final texture size is %dx%d (%.1f%% efficient)' % (size, size, 100.0 * area / (size * size)))
    # Merge all SubImage2s into a single image
    image = Image.new('RGB', (size, size))
    for sub_image2 in sub_image2s:
        image.paste(sub_image2.image, sub_image2.uv.area)
    if options.debug:
        image.draw = ImageDraw.Draw(image)
    # Prepare to write the output files
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
    # Write the Wavefront OBJ file
    obj_filename = os.path.join(output_dirname, '%s.obj' % basename)
    logger.info('creating %s' % obj_filename)
    with open(obj_filename, 'w') as obj:
        vti = 0
        for line, fields in izip(lines, fieldss):
            if not fields:
                obj.write(line)
                continue
            if fields[0] == 'mtllib':
                obj.write('mtllib %s.mtl\r\n' % basename)
                continue
            if fields[0] == 'vt':
                assert len(fields) == 3
                vti += 1
                vt = numpy.array([float(fields[1]), float(fields[2]), 1])
                vt = numpy.dot(sub_image2_by_vti[vti].transform, vt)
                obj.write('vt %f %f\r\n' % (vt[0], vt[1]))
                if options.debug:
                    # Mark up the image with a magenta rectangle at each new texture vertex
                    i, j = vt[0] * image.size[0], vt[1] * image.size[1]
                    r = 5
                    image.draw.rectangle(((i - r, j - r), (i + r, j + r)), outline='#f0f', fill='#fff')
                continue
            if fields[0] == 'usemtl':
                obj.write('usemtl mtl\r\n')
                continue
            obj.write(line)
    # Write the material library
    mtllib_filename = os.path.join(output_dirname, '%s.mtl' % basename)
    logger.info('creating %s' % mtllib_filename)
    with open(mtllib_filename, 'w') as mtllib:
        mtllib.write('newmtl mtl\r\n')
        mtllib.write('Ka 1 1 1\r\n')
        mtllib.write('Kd 1 1 1\r\n')
        mtllib.write('Ks 1 0 0\r\n')
        mtllib.write('map_Kd %s.jpg\r\n' % basename)
    # Write the texture
    image_filename = os.path.join(output_dirname, '%s.jpg' % basename)
    logger.info('creating %s' % image_filename)
    image.transpose(Image.FLIP_TOP_BOTTOM).save(image_filename)
    if options.debug:
        # Write the debug images to disk
        for material in materials:
            if not material.image:
                continue
            basename = os.path.splitext(os.path.basename(material.image.filename))
            filename = os.path.join(os.path.dirname(material.image.filename), basename[0] + '-debug' + basename[1])
            material.image.transpose(Image.FLIP_TOP_BOTTOM).save(filename)


if __name__ == '__main__':
    main(sys.argv)
