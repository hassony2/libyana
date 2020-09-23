import warnings

import numpy as np

try:
    import tinyobjloader
except ImportError:
    warnings.warn(
        "Could not laod tinyobjloader for faster_load_obj, use fast_load_obj"
    )


def faster_load_obj(obj_path):
    reader = tinyobjloader.ObjReader()
    ret = reader.ParseFromFile(obj_path)
    if ret is False:
        raise RuntimeError(f"Could not load {obj_path} with tinyobjloader")
    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()
    vert_nb = len(attrib.vertices) // 3
    vertices = np.array(attrib.vertices).reshape(vert_nb, 3)
    faces = [
        index.vertex_index for shape in shapes for index in shape.mesh.indices
    ]
    face_nb = len(faces) // 3
    faces = np.array(faces).reshape(face_nb, 3)
    colors = np.array(attrib.colors).reshape(
        vert_nb, len(attrib.colors) // vert_nb
    )
    if len(attrib.texcoords):
        tex_coords_nb = len(attrib.texcoords) // 2
        texcoords = np.array(attrib.texcoords).reshape(tex_coords_nb, 2)
    else:
        texcoords = None
    return {
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
        "texcoords": texcoords,
    }


def fast_load_obj(file_obj, **kwargs):
    """
    Reused from Trimesh package but ignores textures to be faster !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, "decode"):
        text = text.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n") + " \n"

    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current["f"]) > 0:
            # get vertices as clean numpy array
            vertices = np.array(current["v"], dtype=np.float64).reshape(
                (-1, 3)
            )
            # do the same for faces
            faces = np.array(current["f"], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (
                np.array(list(remap.keys())),
                np.array(list(remap.values())),
            )
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                "vertices": vertices[vert_order],
                "faces": face_order[faces],
                "metadata": {},
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current["g"]) > 0:
                face_groups = np.zeros(len(current["f"]) // 3, dtype=np.int64)
                for idx, start_f in current["g"]:
                    face_groups[start_f:] = idx
                loaded["metadata"]["face_groups"] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ["v"]}
    current = {k: [] for k in ["v", "f", "g"]}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == "f":
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split("/")
                    current["v"].append(attribs["v"][int(f_split[0]) - 1])
                current["f"].append(remap[f])
        elif line_split[0] == "o":
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == "g":
            # defining a new group
            group_idx += 1
            current["g"].append((group_idx, len(current["f"]) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


def fast_write_obj(vertices, faces, path):
    """
    Adapted from
    https://github.com/mikedh/trimesh

    Export a mesh as a Wavefront OBJ file
    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported
    Returns
    -----------
    export : str
      OBJ format output
    """
    # store the multiple options for formatting
    # vertex indexes for faces
    face_formats = {
        ("v",): "{}",
        ("v", "vn"): "{}//{}",
        ("v", "vt"): "{}/{}",
        ("v", "vn", "vt"): "{}/{}/{}",
    }
    # we are going to reference face_formats with this
    face_type = ["v"]

    # otherwise just export vertices
    v_blob = vertices

    # add the first vertex key and convert the array
    export = (
        "v "
        + array_to_string(v_blob, col_delim=" ", row_delim="\nv ", digits=8)
        + "\n"
    )

    # the format for a single vertex reference of a face
    face_format = face_formats[tuple(face_type)]
    faces = "f " + array_to_string(
        faces + 1, col_delim=" ", row_delim="\nf ", value_format=face_format
    )
    # add the exported faces to the export
    export += faces

    with open(path, "w") as t_f:
        t_f.write(export)


def array_to_string(
    array, col_delim=" ", row_delim="\n", digits=8, value_format="{}"
):
    """
    Convert a 1 or 2D array into a string with a specified number
    of digits and delimiter. The reason this exists is that the
    basic numpy array to string conversions are surprisingly bad.
    Parameters
    ------------
    array : (n,) or (n, d) float or int
       Data to be converted
       If shape is (n,) only column delimiter will be used
    col_delim : str
      What string should separate values in a column
    row_delim : str
      What string should separate values in a row
    digits : int
      How many digits should floating point numbers include
    value_format : str
       Format string for each value or sequence of values
       If multiple values per value_format it must divide
       into array evenly.
    Returns
    ----------
    formatted : str
       String representation of original array
    """
    # convert inputs to correct types
    array = np.asanyarray(array)
    digits = int(digits)
    row_delim = str(row_delim)
    col_delim = str(col_delim)
    value_format = str(value_format)

    # abort for non- flat arrays
    if len(array.shape) > 2:
        raise ValueError(
            "conversion only works on 1D/2D arrays not %s!", str(array.shape)
        )

    # allow a value to be repeated in a value format
    repeats = value_format.count("{}")

    if array.dtype.kind == "i":
        # integer types don't need a specified precision
        format_str = value_format + col_delim
    elif array.dtype.kind == "f":
        # add the digits formatting to floats
        format_str = (
            value_format.replace("{}", "{:." + str(digits) + "f}") + col_delim
        )
    else:
        raise ValueError("dtype %s not convertible!", array.dtype.name)

    # length of extra delimiters at the end
    end_junk = len(col_delim)
    # if we have a 2D array add a row delimiter
    if len(array.shape) == 2:
        format_str *= array.shape[1]
        # cut off the last column delimiter and add a row delimiter
        format_str = format_str[: -len(col_delim)] + row_delim
        end_junk = len(row_delim)

    # expand format string to whole array
    format_str *= len(array)

    # if an array is repeated in the value format
    # do the shaping here so we don't need to specify indexes
    shaped = np.tile(array.reshape((-1, 1)), (1, repeats)).reshape(-1)

    # run the format operation and remove the extra delimiters
    formatted = format_str.format(*shaped)[:-end_junk]

    return formatted
