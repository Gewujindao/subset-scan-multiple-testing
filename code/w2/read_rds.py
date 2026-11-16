"""Minimal reader for R's RDS serialization (XDR, versions 2/3), written for this project: enough for a named
list of numeric matrices with dimnames.  Run with python -I on downloaded files.
  python -I -X utf8 read_rds.py <file.rds> <out.npz>
"""
import gzip, struct, sys
import numpy as np

NILVALUE, REFSXP, GLOBALENV, EMPTYENV, BASEENV, MISSINGARG, UNBOUND, BASENAMESPACE = 254, 255, 253, 242, 241, 251, 252, 247
ALTREP = 238


class Reader:
    def __init__(self, data):
        self.d, self.p, self.refs = data, 0, []

    def i(self):
        v = struct.unpack(">i", self.d[self.p:self.p + 4])[0]; self.p += 4; return v

    def bytes_(self, n):
        b = self.d[self.p:self.p + n]; self.p += n; return b

    def item(self):
        flags = self.i()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == NILVALUE:
            return None
        if t in (GLOBALENV, EMPTYENV, BASEENV, MISSINGARG, UNBOUND, BASENAMESPACE):
            return None
        if t == REFSXP:
            idx = flags >> 8
            if idx == 0:
                idx = self.i()
            return self.refs[idx - 1]
        if t == 1:   # SYMSXP
            name = self.item(); self.refs.append(("sym", name)); return ("sym", name)
        if t in (2, 6):   # LISTSXP / LANGSXP pairlist
            attr = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car = self.item(); cdr = self.item()
            out = [(tag[1] if isinstance(tag, tuple) else tag, car)]
            if isinstance(cdr, list):
                out += cdr
            return out
        if t == 9:   # CHARSXP
            n = self.i()
            return None if n == -1 else self.bytes_(n).decode("utf8", "replace")
        if t in (10, 13):   # LGLSXP, INTSXP
            n = self.i(); v = np.frombuffer(self.bytes_(4 * n), dtype=">i4").astype(np.int64)
            return self.attach(v, has_attr)
        if t == 14:   # REALSXP
            n = self.i(); v = np.frombuffer(self.bytes_(8 * n), dtype=">f8").astype(np.float64)
            return self.attach(v, has_attr)
        if t == 16:   # STRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
            return self.attach(v, has_attr)
        if t in (19, 20):   # VECSXP, EXPRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
            return self.attach(v, has_attr)
        if t == ALTREP:
            info = self.item(); state = self.item(); attr = self.item()
            cls = info[0][1][1] if isinstance(info, list) else None
            if cls == "compact_intseq":
                n, start, step = state; v = start + step * np.arange(int(n))
                return {"value": v.astype(np.int64), "attr": attr}
            if cls in ("wrap_real", "wrap_integer", "wrap_string", "deferred_string"):
                return state[0][1] if isinstance(state, list) else state
            raise ValueError(f"unsupported ALTREP {cls}")
        raise ValueError(f"unsupported SEXP type {t} at byte {self.p}")

    def attach(self, v, has_attr):
        if not has_attr:
            return v
        attr = self.item()
        return {"value": v, "attr": dict((k, a) for k, a in attr)}


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    r = Reader(raw)
    assert r.bytes_(2) == b"X\n", "only XDR format supported"
    ver = r.i(); r.i(); r.i()
    if ver == 3:
        n = r.i(); r.bytes_(n)
    return r.item()


if __name__ == "__main__":
    obj = read(sys.argv[1])
    names = obj["attr"]["names"]
    names = names["value"] if isinstance(names, dict) else names
    out = {}
    for nm, el in zip(names, obj["value"]):
        if isinstance(el, dict) and "dim" in el["attr"]:
            dim = el["attr"]["dim"]; dim = dim["value"] if isinstance(dim, dict) else dim
            M = np.asarray(el["value"], float).reshape(int(dim[1]), int(dim[0])).T   # R is column-major
            out[nm] = M
            dn = el["attr"].get("dimnames")
            if dn is not None:
                dnv = dn["value"] if isinstance(dn, dict) else dn
                if dnv[1] is not None:
                    cols = dnv[1]["value"] if isinstance(dnv[1], dict) else dnv[1]
                    out[nm + "__cols"] = np.array(cols, dtype=object)
            print(nm, M.shape, f"NaN {np.isnan(M).sum()}")
        else:
            print(nm, type(el))
    np.savez(sys.argv[2], **{k: (v.astype(str) if v.dtype == object else v) for k, v in out.items()})
