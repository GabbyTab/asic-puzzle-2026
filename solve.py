"""Jane Street 2026 ASIC puzzle in one command:  python solve.py

  1. extract  puzzle.gds -> gate netlist (KLayout L2N + pin attach; see extract.py)
  2. carve    cluster the die by XY, fold buffers inside each blob, stitch back together
  3. cuts     print each blob's boundary; exhaust the tiny blob that drives `success`
  4. SAT      unroll the whole netlist for 121 load + 8 hold clocks in z3, ask for success=1
  5. ROM      isolate the printer, replay the bus the chip really fed it, read its rows,
              and show the flag row is XORed with a CRC-8 of the board (why no sweep finds it)
  6. eggs     the other messages hidden in the VCD, the GDS layers, and the warmup

The chip is an 11x11 Star Battle checker: I is the board, row by row, 1 = star.
"""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path

import z3

from extract import (
    COMBO_OUTS,
    OUTPUT_GENERATOR,
    Instance,
    ParsedNetlist,
    PuzzleChip,
    ROOT,
    _buffer_aliases,
    _run_klayout,
    _stem,
    apply_aliases,
    extract_from_gds,
    gate,
    group_cut,
    is_flop,
    phrase,
    spatial_groups,
)

WIDTH = 121  # 11 x 11 cells
HOLD = 8  # clocks after enable drops before success must be up
GDS = ROOT / "puzzle.gds"
VCD = ROOT / "example_inputs.vcd"
CTRL = ("clk", "rst_n", "enable", "I")  # not "data" when we look at a blob's boundary
MAX_CUT = 10  # exhaust a cut only if it has at most this many data wires
TOUCH = (  # two stars per row and column, but two of them touch diagonally
    ".......**.."
    "*....*....."
    ".......*.*."
    "*.*........"
    "....*.*...."
    "..*......*."
    "....*.....*"
    ".*....*...."
    "...*......*"
    ".....*..*.."
    ".*.*......."
)
STREAMS = {
    "zeros": [0] * WIDTH,
    "ones": [1] * WIDTH,
    "1010": ([1, 0] * 61)[:WIDTH],
    "touch": [1 if c == "*" else 0 for c in TOUCH],
}
Bus = list[dict[str, int]]  # one dict per clock: value of every wire we drive


# ---------------------------------------------------------------- 2. carve the die


def carve(raw: ParsedNetlist) -> tuple[ParsedNetlist, dict[str, list[Instance]]]:
    """Cluster by XY, drop fillers and fold buffers inside each blob, stitch back.

    Returns the stitched netlist (what SAT runs on) and the per-blob cell lists.
    """
    groups = spatial_groups(raw.instances, seeds={"output_generator": OUTPUT_GENERATOR})
    aliases: dict[str, str] = {}
    kept: dict[str, list[Instance]] = {}
    print("clusters (raw -> slim)")
    for name, insts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        kept[name], alias = _buffer_aliases(insts)
        aliases.update(alias)
        print(f"  {name:20} raw={len(insts):4} slim={len(kept[name]):4} dropped={len(insts) - len(kept[name]):3}")
    slim = {name: list(apply_aliases(insts, aliases)) for name, insts in kept.items()}
    stitched = ParsedNetlist(raw.ports, tuple(i for insts in slim.values() for i in insts))
    print(f"stitched {len(stitched.instances)} cells")
    return stitched, slim


def cut_of(slim: dict[str, list[Instance]], name: str, ports: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Wires entering and leaving one blob."""
    others = [i for n, insts in slim.items() if n != name for i in insts]
    return group_cut(slim[name], others, ports)


# ---------------------------------------------------------------- 3. cuts


def dump_cuts(slim: dict[str, list[Instance]], ports: tuple[str, ...]) -> None:
    print("cuts after per-blob simplify")
    for name, insts in sorted(slim.items(), key=lambda kv: -len(kv[1])):
        if not insts:
            continue
        incoming, outgoing = cut_of(slim, name, ports)
        data_in = [n for n in incoming if n not in CTRL]
        flops = sum(is_flop(i.kind) for i in insts)
        touched = sorted(set(incoming + outgoing) & set(ports))
        print(f"  {name:20} cells={len(insts):4} flops={flops:3} data_in={len(data_in):2} ports={touched}")


def dump_success_cut(slim: dict[str, list[Instance]], ports: tuple[str, ...]) -> None:
    """Find the blob that drives `success` and try every value on its data wires."""
    for name in slim:
        if name == "output_generator" or not slim[name]:
            continue
        incoming, outgoing = cut_of(slim, name, ports)
        if "success" in outgoing:
            break
    else:
        print("success cut: no cluster drives success")
        return
    data_in = [n for n in incoming if n not in CTRL]
    print(f"success cluster {name} data_in={len(data_in)} out={[n for n in outgoing if n in ports]}")
    if len(data_in) > MAX_CUT:
        return
    chip = PuzzleChip(ParsedNetlist(tuple(set(ports) | set(incoming) | set(outgoing)), tuple(slim[name])))
    hits = 0
    for bits in product((0, 1), repeat=len(data_in)):
        succ, _ = chip.run([0] * WIDTH, force=dict(zip(data_in, bits)))
        if succ:
            hits += 1
            print(f"    success=1 {dict(zip(data_in, bits))}")
    print(f"  {hits} / {1 << len(data_in)} assignments fire success")


# ---------------------------------------------------------------- 4. SAT for the board


class Z3Ops:
    """z3 twin of extract.IntOps, so the same gate() builds the SAT formula."""

    T, F = z3.BoolVal(True), z3.BoolVal(False)
    AND, OR, NOT, XOR, IF = z3.And, z3.Or, z3.Not, z3.Xor, z3.If


def zb(x) -> z3.BoolRef:
    return x if isinstance(x, z3.BoolRef) else z3.BoolVal(bool(x))


def combo_order(chip: PuzzleChip) -> list[Instance]:
    """Combinational cells in an order where every input is already known."""
    ready = {i.pins["Q"] for i in chip.flops} | set(chip.parsed.ports)
    produced = {n for i in chip.combo for p, n in i.pins.items() if p in COMBO_OUTS}
    order, remaining = [], list(chip.combo)
    while remaining:
        still = []
        for inst in remaining:
            ins = [n for p, n in inst.pins.items() if p not in COMBO_OUTS]
            if all(n in ready or n not in produced for n in ins):
                order.append(inst)
                ready.update(n for p, n in inst.pins.items() if p in COMBO_OUTS)
            else:
                still.append(inst)
        if len(still) == len(remaining):
            raise ValueError(f"combinational cycle, leftover {len(remaining)}")
        remaining = still
    return order


def settle_z3(nets: dict, order: list[Instance], flops: list[Instance]) -> None:
    """One combinational pass plus the async reset/set of the flops, symbolically."""
    for inst in order:
        args = {p: nets.get(n, Z3Ops.F) for p, n in inst.pins.items() if p not in COMBO_OUTS}
        for pin, val in gate(inst.kind, args, Z3Ops).items():
            nets[inst.pins[pin]] = val
    for inst in flops:
        q = inst.pins["Q"]
        if _stem(inst.kind) == "dfrtp" and inst.pins["RESET_B"] in nets:
            nets[q] = z3.If(zb(nets[inst.pins["RESET_B"]]), zb(nets.get(q, False)), Z3Ops.F)
        if _stem(inst.kind) == "dfstp" and inst.pins["SET_B"] in nets:
            nets[q] = z3.If(zb(nets[inst.pins["SET_B"]]), zb(nets.get(q, False)), Z3Ops.T)


def flop_next(inst: Instance, nets: dict) -> z3.BoolRef:
    d = zb(nets.get(inst.pins["D"], False))
    if _stem(inst.kind) == "dfrtp":
        return z3.If(zb(nets.get(inst.pins["RESET_B"], True)), d, Z3Ops.F)
    if _stem(inst.kind) == "dfstp":
        return z3.If(zb(nets.get(inst.pins["SET_B"], True)), d, Z3Ops.T)
    return d


def sat_success(chip: PuzzleChip) -> list[int]:
    """Unroll the netlist WIDTH load clocks + HOLD clocks; ask z3 for I with success=1."""
    order = combo_order(chip)
    chip.reset()
    q = {i.pins["Q"]: z3.BoolVal(bool(chip._get(i.pins["Q"]))) for i in chip.flops}
    i_vars = [z3.Bool(f"I_{t}") for t in range(WIDTH)]
    nets: dict = {}
    for i_bit, en in zip(i_vars + [Z3Ops.F] * HOLD, [True] * WIDTH + [False] * HOLD):
        nets = {**q, "I": i_bit, "enable": z3.BoolVal(en), "rst_n": Z3Ops.T, "clk": Z3Ops.T}
        settle_z3(nets, order, chip.flops)
        q = {inst.pins["Q"]: flop_next(inst, nets) for inst in chip.flops}
    settle_z3(nets, order, chip.flops)
    s = z3.Solver()
    s.set("timeout", 180_000)
    s.add(zb(nets["success"]))
    if s.check() != z3.sat:
        raise RuntimeError(f"z3 {s.check()}")
    m = s.model()
    return [1 if m.evaluate(v) else 0 for v in i_vars]


# ---------------------------------------------------------------- 5. the printer (ROM)
#
# The hatched block on layout.png ("output_generator") drives O[7:0]. We cut it out,
# record what the rest of the chip feeds into it during a real run, and replay that
# bus on the block alone. Then we can poke its wires and its flops directly.


def record_bus(whole: ParsedNetlist, wires: list[str], stream: list[int]) -> tuple[Bus, str, int]:
    """Run the whole chip on `stream`; log the value of `wires` every clock, plus O and success."""
    full = PuzzleChip(whole)
    full.reset()
    frames: Bus = []

    def snap(i_bit: int, en: int) -> None:
        frames.append({"I": i_bit, "enable": en, **{n: full._get(n) for n in wires}})

    for b in stream:
        full._drive("I", b)
        full._drive("enable", 1)
        full._settle()
        snap(b, 1)
        full._posedge()
    full._drive("enable", 0)
    full._settle()
    hist = [full._o_byte()]
    snap(0, 0)
    for _ in range(16):
        full._posedge()
        hist.append(full._o_byte())
        snap(0, 0)
    return frames, phrase(hist), full._get("success")


def replay(
    hatch: list[Instance],
    ports: tuple[str, ...],
    frames: Bus,
    board: list[int] | None = None,
    zero_flops: tuple[str, ...] = (),
) -> tuple[list[int], list[dict[str, int]]]:
    """Drive the isolated printer with a recorded bus. Returns O per hold clock and all flop Qs.

    `board` swaps in a different I stream under the same bus. `zero_flops` forces those
    flops to 0 just while O is read, which shows the stored byte behind an XOR mixer.
    """
    iso = PuzzleChip(ParsedNetlist(ports, tuple(hatch)))
    iso.reset()
    hist: list[int] = []
    qs: list[dict[str, int]] = []
    for k, vec in enumerate(frames):
        for n, v in vec.items():
            iso._drive(n, v)
        if board is not None and vec["enable"]:
            iso._drive("I", board[k])
        iso._settle()
        if not vec["enable"]:
            qs.append({f.name: iso._get(f.pins["Q"]) for f in iso.flops})
            saved = {f.pins["Q"]: iso._get(f.pins["Q"]) for f in iso.flops if f.name in zero_flops}
            for q in saved:
                iso._drive(q, 0)
            iso._settle()
            hist.append(iso._o_byte())
            for q, v in saved.items():
                iso._drive(q, v)
            iso._settle()
        if k + 1 < len(frames):
            iso._posedge()
    return hist, qs


def hexdump(bs: list[int]) -> str:
    return bytes(bs).hex() + "  " + "".join(chr(b) if 32 <= b < 127 else "." for b in bs)


def bits(vals: list[int]) -> str:
    return "".join(map(str, vals))


def dump_rom(whole: ParsedNetlist, slim: dict[str, list[Instance]], ports: tuple[str, ...], win: list[int]) -> None:
    hatch = slim["output_generator"]
    incoming, _ = cut_of(slim, "output_generator", ports)
    wires = [n for n in incoming if n not in CTRL]
    print(f"ROM / output_generator cells={len(hatch)} incoming={wires}")

    streams = {**STREAMS, "win": win}
    rec: dict[str, Bus] = {}
    said: dict[str, str] = {}
    print("  1. the cut is real: isolated hatch on the recorded bus says what the whole chip says")
    for name, stream in streams.items():
        rec[name], whole_text, succ = record_bus(whole, wires, stream)
        said[name] = phrase(replay(hatch, ports, rec[name])[0])
        ok = "ok" if said[name] == whole_text else "MISMATCH"
        print(f"     {name:6} success={succ} whole={whole_text!r:18} isolated={said[name]!r:18} {ok}")

    print("  2. address wires (level a few clocks into the hold phase, per stream)")
    level = {name: [f for f in frames if not f["enable"]][3] for name, frames in rec.items()}
    address = [w for w in wires if len({lv[w] for lv in level.values()}) > 1]
    print(f"     {'':6} {' '.join(f'{w:>7}' for w in address)}  row")
    for name, lv in level.items():
        print(f"     {name:6} {' '.join(f'{lv[w]:>7}' for w in address)}  {said[name]!r}")

    print("  3. raise one address wire alone on the TRY AGAIN bus (hold phase only)")
    for w in address:
        frames = [{**f, w: (1 if not f["enable"] else f[w])} for f in rec["1010"]]
        out = replay(hatch, ports, frames)[0]
        print(f"     {w:8}=1: {phrase(out)!r:18} {hexdump(out)}")

    print("  4. the success row is not stored in clear")
    o_win, q_win = replay(hatch, ports, rec["win"])
    o_wrong, q_wrong = replay(hatch, ports, rec["win"], board=[0] * WIDTH)
    key = sorted(n for n in q_win[0] if any(a[n] != b[n] for a, b in zip(q_win, q_wrong)))
    rest = sorted(n for n in q_win[0] if n not in key)
    o_rom, _ = replay(hatch, ports, rec["win"], zero_flops=tuple(key))
    print(f"     win bus, win board   O = {hexdump(o_win)}")
    print(f"     win bus, zeros board O = {hexdump(o_wrong)}")
    print(f"     {len(key)} hatch flops depend on the board (key register): {key}")
    print(f"     {len(rest)} do not (character counter etc.):              {rest}")
    print(f"     win bus, key forced 0 (stored bytes) = {hexdump(o_rom)}")
    print(f"     keystream = O_win XOR stored        = {bytes(a ^ b for a, b in zip(o_win, o_rom))[1:16].hex()}")
    print(f"     key register after load, win board  = {bits([q_win[0][n] for n in key])}")
    print(f"     key register after load, zeros board= {bits([q_wrong[0][n] for n in key])}")
    print("     the flag row is XORed with an 8-bit LFSR seeded by all 121 I bits;")
    print("     only the real board decrypts it, so no address sweep can read it out.")

    print(f"  5. every one of the {1 << len(key)} key states on the success row")
    hits = all_keys(hatch, ports, rec["win"], key)
    for k, text in hits:
        print(f"     key={k} -> {text!r}")
    print(f"     {len(hits)} / {1 << len(key)} keys decrypt to text: no second message behind the hash")

    print("  6. what the key register is (probe unit vectors through one clock)")
    identify_lfsr(hatch, ports, rec["win"], key)


def all_keys(hatch: list[Instance], ports: tuple[str, ...], frames: Bus, key: list[str]) -> list[tuple[str, str]]:
    """Load the board once, then force every key state and print the hold phase from there."""
    iso = PuzzleChip(ParsedNetlist(ports, tuple(hatch)))
    iso.reset()
    load = [f for f in frames if f["enable"]]
    hold = [f for f in frames if not f["enable"]]
    for vec in load:
        for n, v in vec.items():
            iso._drive(n, v)
        iso._settle()
        iso._posedge()
    loaded = dict(iso.nets)
    qpin = {f.name: f.pins["Q"] for f in iso.flops}
    hits = []
    for k in range(1 << len(key)):
        iso.nets = dict(loaded)
        for i, name in enumerate(key):
            iso._drive(qpin[name], (k >> i) & 1)
        out = []
        for vec in hold:
            for n, v in vec.items():
                iso._drive(n, v)
            iso._settle()
            out.append(iso._o_byte())
            iso._posedge()
        if phrase(out):
            hits.append((bits([(k >> i) & 1 for i in range(len(key))]), phrase(out)))
    return hits


def identify_lfsr(hatch: list[Instance], ports: tuple[str, ...], frames: Bus, key: list[str]) -> None:
    """Read the key register's update matrix by clocking unit vectors through it."""
    iso = PuzzleChip(ParsedNetlist(ports, tuple(hatch)))
    qpin = {f.name: f.pins["Q"] for f in iso.flops}
    n = len(key)

    def step(frame: dict[str, int], state: list[int], i_bit: int) -> list[int]:
        iso.reset()
        for name, v in frame.items():
            iso._drive(name, v)
        for name, b in zip(key, state):
            iso._drive(qpin[name], b)
        iso._drive("I", i_bit)
        iso._settle()
        iso._posedge()
        return [iso._get(qpin[name]) for name in key]

    unit = [[int(i == j) for i in range(n)] for j in range(n)]
    m_load = [step(frames[0], e, 0) for e in unit]  # column j = M e_j, while loading
    m_hold = [step(frames[-1], e, 0) for e in unit]  # same, while printing
    inject = step(frames[0], [0] * n, 1)
    linear = step(frames[0], [0] * n, 0) == [0] * n
    rows = [bits([m_load[j][i] for j in range(n)]) for i in range(n)]
    for name, row, inj in zip(key, rows, inject):
        print(f"     {name:11} next = {row} . key{'  <- I' if inj else ''}")
    print(f"     linear (zero stays zero): {linear}; {sum(r.count('1') == 1 for r in rows)} of {n} rows are pure shifts")

    poly = charpoly_gf2(m_load)
    val = sum(c << k for k, c in enumerate(poly[:-1]))
    names = {0x07: "CRC-8 (CCITT/ATM)", 0x1D: "CRC-8 SAE-J1850", 0x31: "CRC-8 Maxim", 0x2F: "CRC-8 AUTOSAR", 0x9B: "CRC-8 WCDMA"}
    terms = " + ".join("1" if k == 0 else "x" if k == 1 else f"x^{k}" for k in range(n, -1, -1) if poly[k])
    print(f"     characteristic polynomial {terms} = 0x{val:02X}: {names.get(val, 'not a common CRC')}")
    ticks = next((k for k in range(1, 256) if matpow_gf2(m_load, k) == m_hold), None)
    print(f"     hold-phase step = load step ^ {ticks}: the LFSR runs {ticks} ticks per printed character")
    iso.reset()
    print(f"     seed at reset (dfstp=1, dfrtp=0) = {bits([iso._get(qpin[x]) for x in key])}; "
          "key = CRC-8 of the 121 I bits from that seed")


# GF(2) linear algebra, just enough to name the LFSR polynomial.


def matmul_gf2(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n = len(a)
    return [[sum(a[i][k] & b[k][j] for k in range(n)) & 1 for j in range(n)] for i in range(n)]


def matpow_gf2(m: list[list[int]], k: int) -> list[list[int]]:
    out = [[int(i == j) for j in range(len(m))] for i in range(len(m))]
    for _ in range(k):
        out = matmul_gf2(out, m)
    return out


def solve_gf2(eqs: list[tuple[list[int], int]], n: int) -> list[int] | None:
    """Gaussian elimination over GF(2); returns one solution or None."""
    rows = [list(v) + [r] for v, r in eqs]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                rows[i] = [x ^ y for x, y in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    if any(rows[i][-1] for i in range(r, len(rows))):
        return None
    sol = [0] * n
    for i, c in enumerate(piv):
        sol[c] = rows[i][-1]
    return sol


def charpoly_gf2(m: list[list[int]]) -> list[int]:
    """Coefficients c[0..n] (c[n] = 1) of the minimal polynomial: M^n = sum c[k] M^k."""
    n = len(m)
    pows = [matpow_gf2(m, k) for k in range(n + 1)]
    flat = [[x for row in p for x in row] for p in pows]
    sol = solve_gf2([([flat[k][b] for k in range(n)], flat[n][b]) for b in range(n * n)], n)
    return (sol or [0] * n) + [1]


# ---------------------------------------------------------------- 6. eggs


def vcd_headers(path: Path) -> tuple[str, str]:
    text = path.read_text()
    date = re.search(r"\$date\s+(.*?)\s+\$end", text, re.S)
    ver = re.search(r"\$version\s+(.*?)\s+\$end", text, re.S)
    return (date.group(1).strip() if date else ""), (ver.group(1).strip() if ver else "")


def vcd_attempts(path: Path) -> list[list[int]]:
    """The I bit sampled on every clock edge with enable high, split at each reset."""
    ids: dict[str, str] = {}
    clk = rst = en = i_bit = prev_clk = prev_rst = 0
    started = False
    attempts: list[list[int]] = []
    cur: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("$var"):
            m = re.match(r"\$var \S+ \d+ (\S+) (\S+)", line)
            if m:
                ids[m.group(1)] = m.group(2)
        elif line and line[0] in "01" and len(line) >= 2:
            name, val = ids.get(line[1:]), int(line[0])
            if name == "clk":
                if val == 1 and prev_clk == 0 and started and rst == 1 and en == 1:
                    cur.append(i_bit)
                prev_clk = clk = val
            elif name == "rst_n":
                if val == 1 and prev_rst == 0:
                    if cur:
                        attempts.append(cur)
                    cur, started = [], True
                prev_rst = rst = val
            elif name == "enable":
                en = val
            elif name == "I":
                i_bit = val
    if cur:
        attempts.append(cur)
    return attempts


def vcd_caption(attempts: list[list[int]]) -> str:
    """Each 11-bit row of each attempt, read as a 7-bit ASCII code (LSB first)."""
    chars = []
    for board in attempts:
        for r in range(11):
            row = board[r * 11 : (r + 1) * 11]
            n = int("".join(map(str, row[:7][::-1])), 2)
            if 32 <= n < 127:
                chars.append(chr(n))
    return "".join(chars).rstrip()


def layer200_morse() -> str:
    """Boxes on GDS layer 200 are dots and dashes; gaps separate letters and words."""
    script = ROOT / "tmp" / "_morse_klayout.py"
    out = ROOT / "tmp" / "_morse.json"
    script.write_text(
        "import json, pya, os\n"
        "ly=pya.Layout(); ly.read(os.environ['EXTRACT_GDS'])\n"
        "top=ly.top_cell(); idx=ly.find_layer(pya.LayerInfo(200,0))\n"
        "boxes=[]\n"
        "it=pya.RecursiveShapeIterator(ly, top, idx)\n"
        "while not it.at_end():\n"
        "    sh=it.shape(); b=it.trans()* (sh.box if sh.is_box() else sh.bbox())\n"
        "    boxes.append((b.left,b.width()))\n"
        "    it.next()\n"
        "boxes.sort()\n"
        "open(os.environ['EXTRACT_OUT'],'w').write(json.dumps(boxes))\n"
    )
    _run_klayout(script, GDS, out)
    boxes = json.loads(out.read_text())
    short = min(w for _, w in boxes)
    morse = {".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
             "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
             "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
             "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
             "-.--": "Y", "--..": "Z"}
    gaps = [boxes[i + 1][0] - (boxes[i][0] + boxes[i][1]) for i in range(len(boxes) - 1)]
    ordered = sorted(set(gaps))
    # the two biggest jumps in the sorted gap sizes split symbol / letter / word gaps
    jumps = sorted(((ordered[i + 1] - ordered[i], (ordered[i] + ordered[i + 1]) / 2) for i in range(len(ordered) - 1)), reverse=True)
    t_letter, t_word = sorted(c for _, c in jumps[:2])
    words, word, letter = [], [], ["." if boxes[0][1] == short else "-"]
    for g, (_, w) in zip(gaps, boxes[1:]):
        sym = "." if w == short else "-"
        if g >= t_word:
            word.append("".join(letter))
            words.append(word)
            word, letter = [], [sym]
        elif g >= t_letter:
            word.append("".join(letter))
            letter = [sym]
        else:
            letter.append(sym)
    word.append("".join(letter))
    words.append(word)
    return " ".join("".join(morse.get(ch, ch) for ch in w) for w in words)


# ---------------------------------------------------------------- main


def main() -> None:
    print("extract", GDS.name)
    raw = extract_from_gds(GDS)
    whole, slim = carve(raw)
    dump_cuts(slim, raw.ports)
    dump_success_cut(slim, raw.ports)

    chip = PuzzleChip(whole)
    print(f"SAT success=1 after {WIDTH} enable + {HOLD} hold")
    board = sat_success(chip)
    succ, text = chip.run(board)
    print(f"I {bits(board)}")
    print(f"success={succ} O={text!r}")

    print()
    dump_rom(whole, slim, raw.ports, board)

    print("\neggs")
    for name, stream in STREAMS.items():
        succ, text = chip.run(stream)
        print(f"  {name:8} success={succ} O={text!r}")
    date, ver = vcd_headers(VCD)
    print(f"  VCD $date    {date}")
    print(f"  VCD $version {ver}")
    print(f"  VCD I as 7-bit {vcd_caption(vcd_attempts(VCD))!r}")
    print(f"  layer 200 Morse {layer200_morse()!r}")
    print("  warmup A+B==496 (perfect number)")


if __name__ == "__main__":
    main()
