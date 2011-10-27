obj-merge-textures
==================

This script takes a Wavefront OBJ file with multiple textures as input and
writes a Wavefront OBJ with a single, WebGL-compatible texture as output.


Usage
-----

    obj-merge-textures.py --input=input.obj --output=output.obj

This will create `output.obj`, `output.mtl` and `output.jpg` containing the
single texture Wavefront OBJ object, materials library and texture impage
respectively.  The script merges all textures from `input.obj`'s material
library into a single 2^N * 2^N image for some integer N and transforms the
texture coordinates appropriately.

The `--margin=N` argument allows you to specify a margin around each texture
fragment when it is cut out of the original material.  It defaults to 2 pixels.

Passing the `--alignment=N` value will ensure that all textures are aligned on
`N`-pixel boundaries, allowing the resulting texture to be scaled `N` times
down without textures bleeding into each other.


Bugs
----

The script assumes that all faces are textured, strange things will happen to
non-textured faces.

When aligning textures, the textures should be surrounded with some calculated
color to prevent the black background bleeding into the texture when it is
scaled down.  This effect can already by mitigated by using a suitable value
for `--margin`.
