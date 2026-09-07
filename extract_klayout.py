# KLayout batch: L2N + pin-letter attach. Invoked by extract.py.

import json
import os
import re

import pya

PIN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

POWER = frozenset({"VGND", "VPWR", "VNB", "VPB"})
FILLER_PREFIX = ("decap", "tap", "fill", "diode")

GDS = os.environ["EXTRACT_GDS"]
OUT = os.environ["EXTRACT_OUT"]

ly = pya.Layout()
ly.read(GDS)
top = ly.top_cell()
l67_5 = ly.find_layer(pya.LayerInfo(67, 5))
l67_16 = ly.find_layer(pya.LayerInfo(67, 16))
l70_5 = ly.find_layer(pya.LayerInfo(70, 5))
if l67_5 < 0 or l70_5 < 0:
    raise RuntimeError("missing pin text layers 67/5 or 70/5")

l2n = pya.LayoutToNetlist(pya.RecursiveShapeIterator(ly, top, []))
l2n.threads = 4


def lyr(l, d, name):
    idx = ly.find_layer(pya.LayerInfo(l, d))
    if idx < 0:
        raise RuntimeError("missing layer %s/%s" % (l, d))
    return l2n.make_polygon_layer(idx, name)


li1 = lyr(67, 20, "li1")
mcon = lyr(67, 44, "mcon")
met1 = lyr(68, 20, "met1")
via = lyr(68, 44, "via")
met2 = lyr(69, 20, "met2")
via2 = lyr(69, 44, "via2")
met3 = lyr(70, 20, "met3")
via3 = lyr(70, 44, "via3")
met4 = lyr(71, 20, "met4")
via4 = lyr(71, 44, "via4")
met5 = lyr(72, 20, "met5")
stack = [li1, mcon, met1, via, met2, via2, met3, via3, met4, via4, met5]
for a, b in zip(stack, stack[1:]):
    l2n.connect(a, b)
for metal in (li1, met1, met2, met3, met4, met5):
    l2n.connect(metal)
l2n.extract_netlist()


def net_id(net):
    if net is None:
        return None
    if net.name:
        return str(net.name)
    cid = net.cluster_id
    return "n%s" % (cid() if callable(cid) else cid)


def probe_xy(x, y, layers):
    pt = pya.Point(int(x), int(y))
    for layer in layers:
        nid = net_id(l2n.probe_net(layer, pt))
        if nid is not None:
            return nid
    for dx, dy in ((50, 0), (-50, 0), (0, 50), (0, -50), (100, 0), (0, 100)):
        pt = pya.Point(int(x) + dx, int(y) + dy)
        for layer in layers:
            nid = net_id(l2n.probe_net(layer, pt))
            if nid is not None:
                return nid
    return None


def pin_names(cell):
    names = set()
    for li in (l67_5, l67_16):
        if li < 0:
            continue
        for sh in cell.each_shape(li):
            if sh.is_text() and sh.text.string not in POWER and PIN_RE.match(sh.text.string):
                names.add(sh.text.string)
    return sorted(names)


def pin_points(cell, pin):
    pts = []
    for li in (l67_5, l67_16):
        if li < 0:
            continue
        for sh in cell.each_shape(li):
            if sh.is_text() and sh.text.string == pin:
                d = sh.text.trans.disp
                pts.append((d.x, d.y))
                box = sh.bbox()
                pts.append((box.center().x, box.center().y))
    return pts


def is_filler(kind):
    stem = kind.rsplit("_", 1)[0] if kind.rsplit("_", 1)[-1].isdigit() else kind
    return stem.startswith(FILLER_PREFIX)


ports = {}
for sh in top.each_shape(l70_5):
    if not sh.is_text():
        continue
    name = sh.text.string
    d = sh.text.trans.disp
    nid = probe_xy(d.x, d.y, (met3, met1, li1))
    if nid is None:
        raise RuntimeError("top port %s at %s,%s hit no net" % (name, d.x, d.y))
    ports[name] = nid
if not ports:
    raise RuntimeError("no top-level ports on 70/5")

instances = []
kind_i = {}
for inst in top.each_inst():
    cell_name = inst.cell.name
    if not cell_name.startswith("sky130_"):
        continue
    kind = cell_name.split("__", 1)[1]
    idx = kind_i.get(kind, 0)
    kind_i[kind] = idx + 1
    disp = inst.trans.disp
    rec = {
        "name": "%s_%d" % (kind, idx),
        "kind": kind,
        "x": int(disp.x),
        "y": int(disp.y),
        "pins": {},
    }
    need = [] if is_filler(kind) else pin_names(inst.cell)
    if not need:
        instances.append(rec)
        continue
    for pin in need:
        nid = None
        for lx, ly_ in pin_points(inst.cell, pin):
            w = inst.trans.trans(pya.Point(int(lx), int(ly_)))
            nid = probe_xy(w.x, w.y, (li1, met1, met3))
            if nid is not None:
                break
        if nid is None:
            raise RuntimeError(
                "pin %s.%s at inst %s,%s hit no net" % (kind, pin, disp.x, disp.y)
            )
        rec["pins"][pin] = nid
    instances.append(rec)

open(OUT, "w").write(
    json.dumps({"top": top.name, "ports": ports, "instances": instances}, indent=2)
)
