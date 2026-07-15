###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################
# Shared VTK XML appended-binary helpers for MEEUUW
###################################################################################################

import sys
import base64
import numpy as np
from xml.sax.saxutils import escape


class VTKBinaryAppendedWriter:
    """
    Helper for VTK XML appended raw binary output.

    A DataArray tag contains only an offset, e.g.

        <DataArray ... Format='appended' offset='123'/>

    The actual array bytes are written later in <AppendedData>. Each block is:

        [UInt64 number of bytes][raw array data]

    By default, arrays are written in the machine's native byte order and the
    VTKFile header reports that byte order to ParaView.
    """

    def __init__(self, header_dtype=np.uint64):
        self.arrays = []
        self.offset = 0
        self.header_dtype = np.dtype(header_dtype)
        self.header_nbytes = self.header_dtype.itemsize

        if sys.byteorder == "little":
            self.byte_order = "LittleEndian"
        else:
            self.byte_order = "BigEndian"

        if self.header_dtype == np.dtype(np.uint64):
            self.header_type = "UInt64"
        elif self.header_dtype == np.dtype(np.uint32):
            self.header_type = "UInt32"
        else:
            raise ValueError("header_dtype must be np.uint64 or np.uint32")

    def add_array(self, name, array, dtype, vtk_type, number_of_components=None):
        """
        Register an array and return the corresponding VTK XML DataArray tag.
        """
        array = np.asarray(array, dtype=dtype)
        array = np.ascontiguousarray(array)

        this_offset = self.offset
        nbytes = array.nbytes

        self.arrays.append({"array": array, "nbytes": nbytes})
        self.offset += self.header_nbytes + nbytes

        safe_name = escape(str(name), {"'": "&apos;", '"': "&quot;"})

        if number_of_components is None:
            return (
                f"<DataArray type='{vtk_type}' "
                f"Name='{safe_name}' "
                f"Format='appended' "
                f"offset='{this_offset}'/>\n"
            )

        return (
            f"<DataArray type='{vtk_type}' "
            f"Name='{safe_name}' "
            f"NumberOfComponents='{number_of_components}' "
            f"Format='appended' "
            f"offset='{this_offset}'/>\n"
        )

    def add_points(self, points):
        """
        Register the VTK Points array.
        """
        points = np.asarray(points, dtype=np.float32)
        points = np.ascontiguousarray(points)

        this_offset = self.offset
        nbytes = points.nbytes

        self.arrays.append({"array": points, "nbytes": nbytes})
        self.offset += self.header_nbytes + nbytes

        return (
            f"<DataArray type='Float32' "
            f"NumberOfComponents='3' "
            f"Format='appended' "
            f"offset='{this_offset}'/>\n"
        )

    def write_appended_data(self, vtufile):
        """
        Write all registered binary arrays.
        """
        vtufile.write(b"<AppendedData encoding='base64'>\n_")

        buf = bytearray()
        for item in self.arrays:
            nbytes = np.array([item["nbytes"]], dtype=self.header_dtype)
            buf += nbytes.tobytes()
            buf += item["array"].tobytes()
            #nbytes.tofile(vtufile)
            #item["array"].tofile(vtufile)

        vtufile.write(base64.b64encode(bytes(buf)))
        vtufile.write(b"\n</AppendedData>\n")


def write_text(vtufile, text):
    """
    Write XML text to a file opened in binary mode.
    """
    vtufile.write(text.encode("utf-8"))


def vector3(x_component, z_component, scale=1.0):
    """
    Build a 3-component VTK vector from 2-D x/z data.

    The y component is zero, because MEEUUW is using x-z geometry but VTK uses
    three coordinates/components.
    """
    x_component = np.asarray(x_component) / scale
    z_component = np.asarray(z_component) / scale
    y_component = np.zeros_like(x_component, dtype=np.float32)
    return np.column_stack((x_component, y_component, z_component))


###################################################################################################
