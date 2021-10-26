import numpy as np

def normalized(p): return p / np.linalg.norm(p)

# For dumpPrecomputedValues
def array_to_C_decl(varname:str, a):
    s = "static float %s[] = {\n" % varname
    if a.ndim == 1:
        values = [f"{v:.5f}" for v in a]
        s += "    " + ", ".join(values)
    elif a.ndim == 2:
        rows = []
        for r in range(0, a.shape[0]):
            values = [f"{v:.5f}" for v in a[r,:]]
            rows.append("    " + ", ".join(values))
        s += ",\n".join(rows)
    s += "\n};"
    return s
