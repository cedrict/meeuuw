###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################
# numbering of V nodes
# 3 6 2
# 7 8 5
# 0-4-1
###################################################################################################


def straighten_edges_axisymmetric(geometry, axisymmetric, straighten_edges, nel, icon_V, x_V, z_V):

    if axisymmetric and straighten_edges:
        match geometry:
            case "quarter" | "half" | "eighth":
                for iel in range(0, nel):
                    i0 = icon_V[0, iel]
                    i1 = icon_V[1, iel]
                    i2 = icon_V[2, iel]
                    i3 = icon_V[3, iel]
                    i4 = icon_V[4, iel]
                    i5 = icon_V[5, iel]
                    i6 = icon_V[6, iel]
                    i7 = icon_V[7, iel]
                    i8 = icon_V[8, iel]

                    x_V[i4] = 0.5 * (x_V[i0] + x_V[i1])
                    z_V[i4] = 0.5 * (z_V[i0] + z_V[i1])
                    x_V[i5] = 0.5 * (x_V[i1] + x_V[i2])
                    z_V[i5] = 0.5 * (z_V[i1] + z_V[i2])
                    x_V[i6] = 0.5 * (x_V[i2] + x_V[i3])
                    z_V[i6] = 0.5 * (z_V[i2] + z_V[i3])
                    x_V[i7] = 0.5 * (x_V[i3] + x_V[i0])
                    z_V[i7] = 0.5 * (z_V[i3] + z_V[i0])
                    x_V[i8] = 0.25 * (x_V[i0] + x_V[i1] + x_V[i2] + x_V[i3])
                    z_V[i8] = 0.25 * (z_V[i0] + z_V[i1] + z_V[i2] + z_V[i3])
            case _:
                raise ValueError("straighten_edges_axisymmetric: geometry not allowed")

    return x_V, z_V


###################################################################################################
