"""Exact isfinite source comparison; no shared audit rules are changed.

This adapter compares the complete, unchanged writer and checked helper
metadata. Native evidence is an explicitly pinned *rejection* receipt.
Nothing in this module imports Triton or executes the pinned Python source.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

SOURCE_SHA256 = "9661c6a64cb20de85a75775c6daec1dc7e16397d1aa48b83296e9f37039a0ba6"
NATIVE_RECEIPT_SHA256 = "7a5927b11e17b1c4f4cfb716cb12d290f0ca49baebd81a97deae09e2da8b28aa"
NATIVE_CHECKER_SHA256 = "0f815dcc348b4ea85d0aaf0937d8e630c30100707a5c419fed22ea020c6e33a5"
CUDA_LIBDEVICE_SHA256 = "27b2a5d1e8db008bacefe6019f63922bbd65926de90bb1b527ee597477d2f365"
PRIMARY = "triton.language.extra.cuda.libdevice"
HELPER = "isfinite_func"
KERNEL = "isfinite_func_kernel_rank_1"
WRAPPER = "isfinite_func_wrapper_rank_1"
ALIASES = {"_isfinited": "isfinited", "_finitef": "finitef"}
PARAMETERS = ("in0_ptr", "out0_ptr", "in0_stride0", "in0_stride_order0",
              "out0_stride0", "out0_stride_order0", "s0", "num_tasks",
              "tiles_per_cta", "tile_size0", "one_tile_per_cta")
STORAGES = ("tl.float32", "tl.float64", "tl.bfloat16")

EXPECTED_HELPER = '''@triton.jit
def isfinite_func(x):
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
'''

# Independently recorded statement tree: no source text is used to construct
# the expected writer during a check or a negative control.
EXPECTED_WRITER = '''@triton.jit
def isfinite_func_kernel_rank_1(
    in0_ptr: tl.tensor, out0_ptr: tl.tensor,
    in0_stride0: int, in0_stride_order0: tl.constexpr,
    out0_stride0: int, out0_stride_order0: tl.constexpr,
    s0: int, num_tasks: int, tiles_per_cta: int,
    tile_size0: tl.constexpr, one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta:
        tile_id = pid
        tile_id0 = tile_id
        offset0 = tile_id0 * tile_size0
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty)
        out0 = isfinite_func(in0)
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            tile_id0 = tile_id
            offset0 = tile_id0 * tile_size0
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty)
            out0 = isfinite_func(in0)
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
'''

LEAN_PREAMBLE = '''macro_rules
  | `(isfiniteSource%[$storage:tritonDType]) =>
    `(fun (in0_ptr out0_ptr : RegionName)
      (in0_stride0 in0_stride_order0 out0_stride0 out0_stride_order0 : Nat)
      (s0 num_tasks tiles_per_cta tile_size0 : Nat) (one_tile_per_cta : Bool)
      (in_order_zero : in0_stride_order0 = 0)
      (out_order_zero : out0_stride_order0 = 0) =>
      ComputeKernel.mk [in0_ptr] [out0_ptr] ('''


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def ast_sha(node: ast.AST) -> str:
    return sha(dump(node))


def fail(message: str):
    raise ValueError("isfinite source audit: " + message)


def require(condition, message: str):
    if not condition:
        fail(message)


def unique_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def json_equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def chain_text(alias: str, symbol: str) -> str:
    return f'''try:
    from triton.language.extra.cuda.libdevice import {symbol} as {alias}
except ImportError:
    try:
        from triton.language.math import {symbol} as {alias}
    except ImportError:
        from triton.language.libdevice import {symbol} as {alias}
'''


def path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = path(node.value)
        return None if base is None else base + "." + node.attr
    return None


def binding_audit(module, helper, writer, wrapper, chains):
    """Whole-module scan, including function parameters and nested binders.

    The pinned module is ordinary source, not a hostile Python sandbox.
    This check rejects explicit shadowing, aliases and dynamic namespace paths
    rather than reasoning that their containing code happens not to execute.
    """
    protected = {"triton", "tl", "ImportError", HELPER, KERNEL, WRAPPER, *ALIASES}
    definitions = {HELPER: helper, KERNEL: writer, WRAPPER: wrapper}
    dynamic = {"exec", "eval", "globals", "locals", "vars", "getattr", "setattr", "delattr",
               "__dict__", "__getattribute__", "__setattr__", "__delattr__", "__import__",
               "__builtins__", "__globals__", "__closure__", "modules", "import_module"}
    parents = {c: n for n in ast.walk(module) for c in ast.iter_child_nodes(n)}
    chain_nodes = {n for chain in chains for n in ast.walk(chain)}
    helper_nodes = set(ast.walk(helper))
    writer_nodes = set(ast.walk(writer))
    canonical_imports = {"triton": [], "tl": []}
    extern_imports = {name: [] for name in ALIASES}
    for node in ast.walk(module):
        if ((isinstance(node, ast.Name) and node.id in dynamic)
                or (isinstance(node, ast.Attribute) and node.attr in dynamic)):
            fail("dynamic namespace access")
        if isinstance(node, (ast.Global, ast.Nonlocal)) and protected.intersection(node.names):
            fail("indirect protected binding")
        if isinstance(node, ast.Name) and node.id in protected:
            require(isinstance(node.ctx, ast.Load), "protected name rebound/deleted: " + node.id)
            parent = parents.get(node)
            if node.id in {"triton", "tl"}:
                allowed = isinstance(parent, ast.Attribute) and parent.value is node
            elif node.id == "ImportError":
                allowed = isinstance(parent, ast.ExceptHandler) and parent.type is node and parent in chain_nodes
            elif node.id == KERNEL:
                allowed = isinstance(parent, ast.Subscript) and parent.value is node
            else:
                allowed = isinstance(parent, ast.Call) and parent.func is node
                if node.id in ALIASES:
                    allowed = allowed and node in helper_nodes
                if node.id == HELPER:
                    allowed = allowed and node in writer_nodes
            require(allowed, "protected name aliased or called outside its recorded scope: " + node.id)
        if isinstance(node, ast.Attribute):
            name = path(node)
            root = (name or "").split(".")[0]
            if root in protected:
                require(isinstance(node.ctx, ast.Load), "protected attribute mutation")
            if root in {"tl", "triton"}:
                parent = parents.get(node)
                allowed = isinstance(parent, ast.Call) and parent.func is node
                allowed |= name in {"tl.tensor", "tl.constexpr", "tl.float32"}
                allowed |= (name == "triton.jit" and isinstance(parent, ast.FunctionDef)
                            and node in parent.decorator_list)
                require(allowed, "protected namespace attribute aliased")
        if isinstance(node, ast.arg):
            require(node.arg not in protected, "shadowed parameter: " + node.arg)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in protected:
            require(definitions.get(node.name) is node, "shadowed definition: " + node.name)
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            require(node.name not in protected, "exception/pattern shadowing")
        if isinstance(node, ast.MatchMapping):
            require(node.rest not in protected, "pattern-rest shadowing")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                origin = alias.name if isinstance(node, ast.Import) else node.module or ""
                require(alias.name != "*" and origin.split(".")[0] not in {"sys", "builtins", "importlib"},
                        "dynamic/wildcard namespace import")
                if node in chain_nodes:
                    require(name in ALIASES, "unrecorded extern binding")
                    extern_imports[name].append(node)
                    continue
                canonical = ((isinstance(node, ast.Import) and alias.name == "triton" and alias.asname is None)
                             or (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "triton"
                                 and alias.name == "language" and alias.asname == "tl"))
                if origin == "triton" or origin.startswith("triton."):
                    require(canonical, "additional Triton import path")
                if name in protected:
                    require(canonical and node in module.body, "additional protected import")
                    canonical_imports[name].append(node)
    require(all(len(v) == 1 for v in canonical_imports.values()), "canonical namespace imports are not unique")
    require(all(len(v) == 3 for v in extern_imports.values()), "each complete import chain must bind its alias three times")


def python_structure(text: str):
    """Structural check independent of the byte pin, usable by negative controls."""
    module = ast.parse(text, type_comments=True)
    def unique_function(name):
        matches = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(matches) == 1, "expected one top-level " + name)
        return matches[0]
    helper, writer, wrapper = map(unique_function, (HELPER, KERNEL, WRAPPER))
    require(dump(helper) == dump(ast.parse(EXPECTED_HELPER, type_comments=True).body[0]),
            "helper signature, decorator, return, dtype predicate or explicit cast changed")
    require(dump(writer) == dump(ast.parse(EXPECTED_WRITER, type_comments=True).body[0]),
            "complete writer signature/decorator/statement/control AST changed")
    chains = []
    for alias, symbol in ALIASES.items():
        expected = ast.parse(chain_text(alias, symbol)).body[0]
        matches = [n for n in module.body if isinstance(n, ast.Try) and dump(n) == dump(expected)]
        require(len(matches) == 1, "complete three-branch import chain changed: " + alias)
        chains.append(matches[0])
    require(sum(isinstance(n, ast.Try) for n in module.body) == 2, "unexpected top-level import/control chain")
    require(module.body.index(chains[0]) < module.body.index(chains[1]) < module.body.index(helper) < module.body.index(writer),
            "import/helper/writer declaration order changed")
    binding_audit(module, helper, writer, wrapper, chains)
    return module, helper, writer, chains


def python_pinned(raw: bytes):
    require(sha(raw) == SOURCE_SHA256, "pinned Python bytes changed")
    return python_structure(raw.decode())


def tokens(text: str):
    return re.findall(r"[A-Za-z_][\w']*|[0-9]+|[^\s]", text)


def braces(text: str):
    """Tiny notation conversion only: braces to Python indentation, with else."""
    lines, current, parens = [], [], []
    depth, quote, escaped = 0, None, False
    def flush(block=False):
        line = "".join(current).strip()
        current.clear()
        if not line:
            return
        if block:
            require(bool(re.fullmatch(r"for\s+\w+\s+in\s+range\(.+\)|if\s+.+|else", line, re.S)),
                    "unrecognized writer control header")
            line += ":"
        lines.append("    " * depth + line)
    for c in text:
        if quote is not None:
            current.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            current.append(c)
        elif c in "([":
            parens.append(c)
            current.append(c)
        elif c in ")]":
            require(bool(parens) and parens.pop() == ("(" if c == ")" else "["), "unbalanced writer delimiter")
            current.append(c)
        elif c == "{":
            require(not parens, "expression braces are unsupported")
            flush(True)
            depth += 1
        elif c == "}":
            require(not parens and depth > 0, "unbalanced closing writer brace")
            flush()
            depth -= 1
        elif c == "\n" and not parens:
            flush()
        else:
            current.append(c)
    require(not parens and depth == 0 and quote is None, "unclosed writer syntax")
    flush()
    return ast.parse("\n".join(lines))


class Notation(ast.NodeTransformer):
    """Normalize only make_block_ptr keyword notation and literal sequences."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        keywords = [kw.arg for kw in node.keywords]
        require(None not in keywords and len(keywords) == len(set(keywords)), "duplicate or indirect keyword")
        if path(node.func) == "tl.make_block_ptr":
            fields = ("base", "shape", "strides", "offsets", "block_shape", "order")
            require(len(node.args) <= len(fields), "extra block-pointer argument")
            arguments = dict(zip(fields, node.args))
            for kw in node.keywords:
                require(kw.arg in fields and kw.arg not in arguments, "invalid block-pointer keyword")
                arguments[kw.arg] = kw.value
            require(set(arguments) == set(fields), "incomplete block pointer")
            for key in fields[1:]:
                value = arguments[key]
                arguments[key] = ast.Tuple(elts=value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value], ctx=ast.Load())
            node.args = []
            node.keywords = [ast.keyword(arg=k, value=arguments[k]) for k in fields]
        for kw in node.keywords:
            if kw.arg == "boundary_check":
                require(isinstance(kw.value, (ast.List, ast.Tuple)), "nonliteral boundary sequence")
                kw.value = ast.Tuple(elts=kw.value.elts, ctx=ast.Load())
        return node


class RankOneOrderNotation(ast.NodeTransformer):
    """Four constructor orders and four boundary checks under zero proofs.

    Applies only to the expected original comparison tree. In particular it
    replaces only the precisely matched input/output constructor and memory
    keyword operands, never arbitrary uses. Lean retains both order binders and
    supplies explicit equality proof binders in its checked macro preamble.
    """
    def __init__(self):
        self.changed = 0
        self.boundaries = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if path(node.func) in {"tl.load", "tl.store"}:
            incoming = path(node.func) == "tl.load"
            require(len(node.args) == (1 if incoming else 2)
                    and path(node.args[0]) == ("in0_bptr" if incoming else "out0_bptr"),
                    "rank-one boundary comparison encountered a changed memory operand")
            require(len(node.keywords) == 1 and node.keywords[0].arg == "boundary_check",
                    "rank-one boundary comparison requires the original explicit boundary check")
            parameter = "in0_stride_order0" if incoming else "out0_stride_order0"
            expected = ast.Tuple(elts=[ast.Name(id=parameter, ctx=ast.Load())], ctx=ast.Load())
            require(dump(node.keywords[0].value) == dump(expected),
                    "original boundary check is not its matching source order parameter")
            node.keywords[0].value = ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load())
            self.boundaries += 1
            return node
        if path(node.func) != "tl.make_block_ptr":
            return node
        fields = {kw.arg: kw for kw in node.keywords}
        require(not node.args and set(fields) == {"base", "shape", "strides", "offsets", "block_shape", "order"},
                "rank-one order comparison needs canonical complete block-pointer fields")
        expected_names = {"in0_ptr": "in0_stride_order0", "out0_ptr": "out0_stride_order0"}
        base = path(fields["base"].value)
        require(base in expected_names, "rank-one order comparison encountered an unrecorded pointer")
        expected = ast.Tuple(elts=[ast.Name(id=expected_names[base], ctx=ast.Load())], ctx=ast.Load())
        require(dump(fields["order"].value) == dump(expected), "original order is not its matching source parameter")
        fields["order"].value = ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load())
        self.changed += 1
        return node


def lean_writer(lean_text: str, storage: str, source):
    require(storage in STORAGES, "unreviewed Lean storage specialization")
    metadata = "/-\nSource-audit Lean: isfiniteSource%\nSource-audit args: storage=" + storage + "\n-/\n"
    selected = source.select_lean_audit_source(lean_text, metadata)
    code = source.strip_lean_comments(selected)
    require(code.count("triton {") == 1, "expected exactly one selected literal Triton body")
    preamble = source.lean_first_preamble(code)
    require(tokens(preamble) == tokens(LEAN_PREAMBLE), "Lean source macro or complete external binder signature changed")
    # Check the enclosing constructor, too: auditing a discarded quotation is
    # not evidence about the resulting ComputeKernel.
    mask = source.lean_code_mask(code)
    begin = mask.index("triton {") + len("triton ")
    level, end = 0, None
    for index in range(begin, len(mask)):
        level += (mask[index] == "{") - (mask[index] == "}")
        if level == 0:
            end = index + 1
            break
    require(end is not None and tokens(code[end:]) == tokens(").surfaceBody)"), "source constructor tail changed")
    body = source.lean_first_triton_body(selected)
    for alias, symbol in ALIASES.items():
        match = re.match(r"\s*tl\.source_alias\s+" + alias + r"\s*=\s*" + re.escape(PRIMARY + "." + symbol) + r"\s*(?:\n|$)", body)
        require(match is not None, "Lean alias prefix must retain both primary CUDA bindings in source order")
        body = body[match.end():]
    helper = re.match(r"\s*tl\.source_helper\s+isfinite_func\s*\(\s*x\s*\)\s*\{\s*return\s+([^{}]+)\}\s*", body)
    require(helper is not None, "Lean helper declaration/name/parameter changed")
    expression = ast.parse(helper[1].strip(), mode="eval").body
    expected = ast.parse(EXPECTED_HELPER).body[0].body[0].value
    require(dump(expression) == dump(expected), "Lean helper return AST changed")
    body = body[helper.end():]
    region = re.match(r"tl\.region\s+in0_ptr\s*=\s*" + re.escape(storage) + r"\s*,\s*out0_ptr\s*=\s*tl\.int1\s*(?:\n|$)", body)
    require(region is not None, "Lean input/output storage declaration changed")
    body = body[region.end():]
    require("tl.source_" not in body and "__finite_helper_resolved" not in body and "tl.region" not in body,
            "writer contains extra metadata or an internal helper call")
    def anti(match):
        value = match[1]
        require(value in PARAMETERS or value == "0", "non-direct or undeclared source antiquotation")
        return value
    body = re.sub(r"\$\((\w+)\)", anti, body)
    body = re.sub(r"\((\[\s*(?:[0-9]+\s*(?:,\s*[0-9]+\s*)*)?\])\s*:\s*List Nat\)", r"\1", body)
    require("$" not in body and ":=" not in body, "unresolved source splice or alternate assignment")
    actual = Notation().visit(braces(body))
    expected_writer = ast.parse(EXPECTED_WRITER).body[0]
    expected_body = Notation().visit(ast.Module(body=expected_writer.body, type_ignores=[]))
    orders = RankOneOrderNotation()
    expected_body = orders.visit(expected_body)
    require(orders.changed == 4, "rank-one order notation must cover exactly four original constructors")
    require(orders.boundaries == 4, "rank-one boundary notation must retain exactly four original memory checks")
    require(dump(actual) == dump(expected_body), "complete Lean writer expression/control AST differs")
    return {"storage": storage, "body_sha256": sha(body), "writer_ast_sha256": ast_sha(actual),
            "helper_ast_sha256": ast_sha(expression), "both_writer_arms_checked": True,
            "rank_one_order_proofs": {"in0_stride_order0": 0, "out0_stride_order0": 0},
            "order_constructor_notation_sites": orders.changed,
            "boundary_check_notation_sites": orders.boundaries,
            "source_order_parameters_retained": True, "boundary_checks_retained": True}


def native_evidence(receipt_raw: bytes, checker_raw: bytes, extracted_raw: bytes,
                    cuda_raw: bytes, original_raw: bytes, *, check_pins=True):
    """Consume root's real rejection receipt, never promote it to admission."""
    if check_pins:
        require(sha(receipt_raw) == NATIVE_RECEIPT_SHA256, "unreviewed native receipt bytes")
        require(sha(checker_raw) == NATIVE_CHECKER_SHA256, "unreviewed native checker bytes")
    require(sha(original_raw) == SOURCE_SHA256, "native source pin mismatch")
    require(sha(cuda_raw) == CUDA_LIBDEVICE_SHA256, "unreviewed CUDA libdevice snapshot")
    record = unique_json(receipt_raw.decode())
    require(record["source_sha256"] == SOURCE_SHA256 and record["checker_sha256"] == sha(checker_raw),
            "native source/checker linkage mismatch")
    require(record["triton_version"] == "3.8.0" and record["target"] == {"backend": "cuda", "arch": 90, "warp_size": 32},
            "native environment changed")
    require(record["primary_cuda_identity_checked"] is True, "primary CUDA alias identity was not checked")
    require(record["kernel_launched"] is False and record["independent_replay"] is False and record["complete_task"] is False,
            "native receipt scope changed")
    expected_aliases = {alias: {"module": PRIMARY, "source_sha256": CUDA_LIBDEVICE_SHA256} for alias in ALIASES}
    require(json_equal(record["aliases"], expected_aliases), "fallback or unrecorded libdevice identity")
    original = ast.parse(original_raw)
    nodes = [n for n in original.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))
             or isinstance(n, ast.FunctionDef) and n.name in {HELPER, KERNEL}]
    require([ast_sha(n) for n in nodes] == record["extracted_ast_sha256"], "native extraction does not retain original import/helper/writer ASTs")
    require([dump(n) for n in nodes] == [dump(n) for n in ast.parse(extracted_raw).body], "native extracted module differs")
    cuda = ast.parse(cuda_raw)
    for symbol, precision in (("finitef", "fp32"), ("isfinited", "fp64")):
        declarations = [n for n in cuda.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
        expected = ast.parse(f'''@core.extern
def {symbol}(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {{
        (core.dtype("{precision}"),): ("__nv_{symbol}", core.dtype("int32")),
    }}, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)
''').body[0]
        require(len(declarations) == 1 and dump(declarations[0]) == dump(expected), "native classifier declaration/conversion changed")
    expected_profiles = {}
    for dt, one, tile, abi in itertools.product(("fp16", "bf16", "fp32", "fp64", "fp8e4nv", "fp8e5"), (False, True), (16, 512), ("i32", "i64")):
        name = f"{dt}-one{int(one)}-tile{tile}-{abi}"
        expected_profiles[name] = {
            "signature": {"in0_ptr": "*" + dt, "out0_ptr": "*i1",
                          **{k: abi for k in ("in0_stride0", "out0_stride0", "s0", "num_tasks", "tiles_per_cta")}},
            "constexprs": {"in0_stride_order0": 0, "out0_stride_order0": 0,
                           "tile_size0": tile, "one_tile_per_cta": one},
            "options": {"num_warps": 4},
        }
    require(len(record["results"]) == 48 and record["counts"] == {"compiled": 0, "rejected": 48}, "native rejection matrix changed")
    seen = set()
    for row in record["results"]:
        name = row["name"]
        require(name in expected_profiles and name not in seen, "unknown/duplicate native profile")
        seen.add(name)
        expected = expected_profiles[name]
        require(json_equal({k: row[k] for k in expected}, expected), "native ABI/constexpr profile changed")
        require(row["status"] == "rejected" and row["error_type"] == "CompilationError", "native rejection was changed to acceptance")
        require("'_aggregate_type' object has no attribute 'element_ty'" in row["error"]
                and "out0_bptr.type.element_ty" in row["error"], "native rejection cause changed")
        require("artifacts" not in row, "rejected profile unexpectedly claims emitted artifacts")
    return {"primary_cuda_alias_identity": True, "fallback_native_acceptance": False,
            "native_compile_acceptance": False, "compiled": 0, "rejected": 48,
            "blocker": "original out0_bptr.type.element_ty is rejected by Triton 3.8.0",
            "native_receipt_sha256": sha(receipt_raw), "kernel_launched": False,
            "lean_native_refinement": False, "independent_replay": False}
