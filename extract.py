"""GDS extract (KLayout L2N + pin-attach), simplify, cluster, gate-level sim."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).parent
KLAYOUT = Path("/Applications/KLayout/klayout.app/Contents/MacOS/klayout")
KLAYOUT_SCRIPT = ROOT / "extract_klayout.py"
FILLER_PREFIX = ("decap", "tap", "fill", "diode")
FLOP_STEMS = frozenset({"dfrtp", "dfxtp", "dfrtn", "dfstp"})
BUF_STEMS = frozenset({"clkbuf", "buf", "clkdlybuf4s15", "clkdlybuf4s18", "clkdlybuf4s25", "clkdlybuf4s50"})
OUTPUT_GENERATOR = (133_000, 75_000, 198_000, 275_000)
COMBO_OUTS = frozenset({"X", "Y", "HI", "LO"})
DRIVE_PINS = COMBO_OUTS | {"Q"}
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Instance:
    name: str
    kind: str
    pins: dict[str, str]
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True)
class ParsedNetlist:
    ports: tuple[str, ...]
    instances: tuple[Instance, ...]


def _stem(kind: str) -> str:
    if "_" in kind:
        head, tail = kind.rsplit("_", 1)
        if tail.isdigit():
            return head
    return kind


def is_filler(kind: str) -> bool:
    return _stem(kind).startswith(FILLER_PREFIX)


def is_flop(kind: str) -> bool:
    return _stem(kind) in FLOP_STEMS


def _bit(x: int) -> int:
    if x not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got {x}")
    return x


def _run_klayout(script: Path, gds: Path, out: Path) -> None:
    """Run a pya script inside KLayout's own Python; it reads EXTRACT_GDS and writes EXTRACT_OUT."""
    if not KLAYOUT.is_file():
        raise FileNotFoundError(KLAYOUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EXTRACT_GDS": str(gds), "EXTRACT_OUT": str(out)}
    env.pop("PYTHONPATH", None)
    run = subprocess.run([str(KLAYOUT), "-b", "-r", str(script)], capture_output=True, text=True, env=env)
    if run.returncode != 0 or not out.is_file():
        raise RuntimeError(f"klayout failed: {run.stderr or run.stdout}")


def parsed_from_gds_json(data: dict) -> ParsedNetlist:
    ports = data.get("ports") or {}
    if not ports:
        raise ValueError("GDS extract has no top ports")
    rename = {nid: name for name, nid in ports.items()}
    insts = []
    for row in data["instances"]:
        kind = row["kind"]
        pins = {pin: rename.get(nid, nid) for pin, nid in row.get("pins", {}).items()}
        if not is_filler(kind) and not pins:
            raise ValueError(f"{row['name']} ({kind}) has no logic pins")
        insts.append(Instance(row["name"], kind, pins, row.get("x"), row.get("y")))
    if not insts:
        raise ValueError("GDS extract has no instances")
    return ParsedNetlist(tuple(ports), tuple(insts))


def extract_from_gds(gds: str | Path, cache: bool = True) -> ParsedNetlist:
    gds = Path(gds)
    if not gds.is_file():
        raise FileNotFoundError(gds)
    out = ROOT / "tmp" / f"{gds.stem}_extracted.json"
    if not (cache and out.is_file()):
        _run_klayout(KLAYOUT_SCRIPT, gds, out)
    return parsed_from_gds_json(json.loads(out.read_text()))


class _UF:
    def __init__(self, names: Sequence[str]) -> None:
        self.p = {n: n for n in names}

    def find(self, n: str) -> str:
        while self.p[n] != n:
            self.p[n] = self.p[self.p[n]]
            n = self.p[n]
        return n

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def in_bbox(inst: Instance, box: BBox) -> bool:
    x0, y0, x1, y1 = box
    if inst.x is None or inst.y is None:
        raise ValueError(f"{inst.name} has no XY")
    return x0 <= inst.x <= x1 and y0 <= inst.y <= y1


def spatial_groups(
    instances: Sequence[Instance],
    cut: float = 10_000,
    seeds: dict[str, BBox] | None = None,
) -> dict[str, list[Instance]]:
    logic = [i for i in instances if not is_filler(i.kind)]
    locked: dict[str, str] = {}
    for name, box in (seeds or {}).items():
        for inst in logic:
            if in_bbox(inst, box):
                locked[inst.name] = name
    groups: dict[str, list[Instance]] = defaultdict(list)
    for inst in logic:
        if inst.name in locked:
            groups[locked[inst.name]].append(inst)
    rest = [i for i in logic if i.name not in locked]
    if rest:
        uf = _UF([i.name for i in rest])
        edges = [
            (math.hypot(a.x - b.x, a.y - b.y), a.name, b.name)
            for i, a in enumerate(rest)
            for b in rest[i + 1 :]
        ]
        edges.sort()
        for dist, a, b in edges:
            if dist > cut:
                break
            uf.union(a, b)
        comps: dict[str, list[Instance]] = defaultdict(list)
        for inst in rest:
            comps[uf.find(inst.name)].append(inst)
        ordered = sorted(comps.values(), key=lambda xs: (min(i.x for i in xs), min(i.y for i in xs)))
        for n, insts in enumerate(ordered):
            groups[f"g{n}"] = insts
    return dict(groups)


def _buffer_aliases(instances: Sequence[Instance]) -> tuple[list[Instance], dict[str, str]]:
    keep: list[Instance] = []
    alias: dict[str, str] = {}
    for inst in instances:
        if is_filler(inst.kind):
            continue
        if _stem(inst.kind) in BUF_STEMS:
            alias[inst.pins["X"]] = inst.pins["A"]
            continue
        keep.append(inst)
    return keep, alias


def apply_aliases(instances: Sequence[Instance], alias: dict[str, str]) -> tuple[Instance, ...]:
    def resolve(net: str) -> str:
        seen: set[str] = set()
        while net in alias and net not in seen:
            seen.add(net)
            net = alias[net]
        return net

    return tuple(Instance(i.name, i.kind, {p: resolve(n) for p, n in i.pins.items()}, i.x, i.y) for i in instances)


def group_cut(
    insts: Sequence[Instance],
    others: Sequence[Instance],
    ports: Sequence[str],
) -> tuple[list[str], list[str]]:
    driven, used = set(), set()
    for inst in insts:
        for pin, net in inst.pins.items():
            (driven if pin in DRIVE_PINS else used).add(net)
    used_out = {n for inst in others for p, n in inst.pins.items() if p not in DRIVE_PINS}
    port_set = set(ports)
    incoming = sorted(n for n in used if n not in driven)
    outgoing = sorted(n for n in driven if n in used_out or n in port_set)
    return incoming, outgoing


class IntOps:
    """Boolean operators on 0/1 ints. solve.py passes a z3 twin to build SAT formulas."""

    T, F = 1, 0
    AND = staticmethod(lambda *a: int(all(a)))
    OR = staticmethod(lambda *a: int(any(a)))
    NOT = staticmethod(lambda a: 1 - a)
    XOR = staticmethod(lambda a, b: a ^ b)
    IF = staticmethod(lambda s, a, b: a if s else b)


def gate(kind: str, p: dict, ops=IntOps) -> dict:
    """Evaluate one sky130 combinational cell: {output pin: value}.

    The cell name spells the function, so no per-cell table is needed:
      [n]and/or N [b..]        N inputs A..D; a pin named X_N is inverted; leading n inverts the output
      aXY[b]o[i] / oXY[b]a[i]  input groups A,B,C.. of size X,Y.. AND-then-OR (or OR-then-AND); i inverts
      inv, xor2, xnor2, mux2[i], conb, buffers
    Inverted outputs are on pin Y, plain outputs on pin X.
    """
    return _gate_plan(_stem(kind))(p, ops)


def _pin(p: dict, name: str, ops):
    return p[name] if name in p else ops.NOT(p[name + "_N"])


@lru_cache(maxsize=None)
def _gate_plan(s: str):
    """Decode a cell name once into a function (pins, ops) -> {out pin: value}."""
    if s in BUF_STEMS:
        return lambda p, ops: {"X": p["A"]}
    if s in ("inv", "clkinv"):
        return lambda p, ops: {"Y": ops.NOT(p["A"])}
    if s == "conb":
        return lambda p, ops: {"HI": ops.T, "LO": ops.F}
    if s == "xor2":
        return lambda p, ops: {"X": ops.XOR(p["A"], p["B"])}
    if s == "xnor2":
        return lambda p, ops: {"Y": ops.NOT(ops.XOR(p["A"], p["B"]))}
    if s == "mux2":
        return lambda p, ops: {"X": ops.IF(p["S"], p["A1"], p["A0"])}
    if s == "mux2i":
        return lambda p, ops: {"Y": ops.NOT(ops.IF(p["S"], p["A1"], p["A0"]))}
    if m := re.fullmatch(r"(n?)(and|or)(\d)b*", s):
        neg, is_and, letters = bool(m.group(1)), m.group(2) == "and", "ABCD"[: int(m.group(3))]

        def flat(p, ops):
            v = (ops.AND if is_and else ops.OR)(*[_pin(p, c, ops) for c in letters])
            return {"Y": ops.NOT(v)} if neg else {"X": v}

        return flat
    if m := re.fullmatch(r"([ao])((?:\db*)+)([ao])(i?)", s):
        and_first, neg = m.group(1) == "a", bool(m.group(4))
        groups = [[f"{g}{i + 1}" for i in range(int(n))] for g, n in zip("ABCD", re.findall(r"\d", m.group(2)))]

        def nested(p, ops):
            inner, outer = (ops.AND, ops.OR) if and_first else (ops.OR, ops.AND)
            v = outer(*[inner(*[_pin(p, name, ops) for name in grp]) for grp in groups])
            return {"Y": ops.NOT(v)} if neg else {"X": v}

        return nested
    raise ValueError(f"no combinational model for {s}")


class PuzzleChip:
    def __init__(self, parsed: ParsedNetlist):
        need = ("clk", "rst_n", "enable", "I", "success")
        missing = [p for p in need if p not in parsed.ports]
        if missing:
            raise ValueError(f"missing ports {missing}")
        self.parsed = parsed
        self.logic = [i for i in parsed.instances if not is_filler(i.kind)]
        self.flops = [i for i in self.logic if is_flop(i.kind)]
        self.combo = [i for i in self.logic if not is_flop(i.kind)]
        self.o_ports = tuple(
            sorted(
                (p for p in parsed.ports if p.startswith("O[")),
                key=lambda n: int(n.split("[", 1)[1].rstrip("]")),
            )
        )
        self.nets: dict[str, int] = {}
        self.reset()

    def _drive(self, name: str, val: int) -> None:
        self.nets[name] = _bit(val)

    def _get(self, name: str) -> int:
        return self.nets[name]

    def _apply_async(self) -> bool:
        changed = False
        for inst in self.flops:
            stem = _stem(inst.kind)
            q = inst.pins["Q"]
            if stem == "dfrtp" and self._get(inst.pins["RESET_B"]) == 0:
                if self.nets.get(q) != 0:
                    self.nets[q] = 0
                    changed = True
            elif stem == "dfstp" and self._get(inst.pins["SET_B"]) == 0:
                if self.nets.get(q) != 1:
                    self.nets[q] = 1
                    changed = True
        return changed

    def _settle(self) -> None:
        for _ in range(64):
            changed = False
            for inst in self.combo:
                args = {pin: self._get(net) for pin, net in inst.pins.items() if pin not in COMBO_OUTS}
                for out_pin, val in gate(inst.kind, args).items():
                    dest = inst.pins[out_pin]
                    if self.nets.get(dest) != val:
                        self.nets[dest] = val
                        changed = True
            if self._apply_async():
                changed = True
            if not changed:
                return
        raise RuntimeError("combinational loop did not settle")

    def _posedge(self) -> None:
        nxt = {inst.pins["Q"]: self._get(inst.pins["D"]) for inst in self.flops}
        self.nets.update(nxt)
        self._apply_async()
        self._settle()

    def reset(self) -> None:
        self.nets = {}
        for inst in self.logic:
            for net in inst.pins.values():
                self.nets.setdefault(net, 0)
        for port in self.parsed.ports:
            self.nets.setdefault(port, 0)
        self._drive("rst_n", 0)
        self._settle()
        self._posedge()
        self._drive("rst_n", 1)
        self._settle()

    def _o_byte(self) -> int:
        n = 0
        for i, name in enumerate(self.o_ports):
            n |= self._get(name) << i
        return n

    def run(
        self,
        bits: Sequence[int],
        hold_clocks: int = 16,
        force: dict[str, int] | None = None,
    ) -> tuple[int, str]:
        force = force or {}
        self.reset()

        def apply_force() -> None:
            for name, val in force.items():
                self._drive(name, _bit(val))

        for b in bits:
            self._drive("I", _bit(b))
            self._drive("enable", 1)
            apply_force()
            self._settle()
            self._posedge()
        self._drive("enable", 0)
        apply_force()
        self._settle()
        hist = [self._o_byte()]
        for _ in range(hold_clocks):
            apply_force()
            self._posedge()
            hist.append(self._o_byte())
        return self.nets.get("success", 0), phrase(hist)


def phrase(values: Sequence[int]) -> str:
    out: list[int] = []
    prev = None
    for v in values:
        if prev is None or v != prev:
            out.append(int(v))
            prev = v
    data = bytes(b for b in out if b)
    if data and all(32 <= b < 127 or b in (9, 10, 13) for b in data):
        return data.decode("ascii")
    return ""


