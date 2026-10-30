#!/usr/bin/env python3
"""Read-only inventory of frozen FP admissions and candidate classifications.

Historical rows are immutable evidence, not the current empirical rule set.
By default no strict exclusions are applied: a source name or classification
label is not a checked, domain-matched Lean certificate. --current verifies
the compiled typed registry and evaluates its metadata with Lean first.
It imports no experiment runner, runs no GPU work, and writes no reports.
"""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


TABLES = {
    "ReportedAdmission": 34,
    "SupplementalAdmission": 36,
    "CountAdmission": 2,
    "ExpAdmission": 5,
    "LogAdmission": 4,
}
PRECISIONS = ("input", "compute", "accumulator", "output")
STRING = r'"(?:\\.|[^"\\])*"'
PARSED_JSON = re.compile(r"Lean\.Json\.parse\s+(" + STRING + r")")
ROW_START = re.compile(r"^def (\w+) : (ReportedRule|ReportedScalarRule) where", re.M)
NEXT_DECL = re.compile(r"^(?:def|theorem|end)\b", re.M)
GUARD = re.compile(r'⟨\s*(' + STRING + r')\s*,\s*\.(finite|nonzero|positive)\s*⟩')
CLASSIFICATIONS = {"exact-existing", "exact-provable", "conditional-boundary", "genuinely-relaxed"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field(body, name):
    match = re.search(r"^\s*" + re.escape(name) + r"\s*:=\s*(" + STRING + r")\s*$", body, re.M)
    if not match:
        raise ValueError(f"missing/unrecognized generated field: {name}")
    return json.loads(match[1])


def parse_json_literals(text):
    return [json.loads(json.loads(m[1])) for m in PARSED_JSON.finditer(text)]


def parse_guards(text):
    """Parse the complete generated list contents, rejecting partial matches."""
    remaining = text.strip()
    guards = []
    while remaining:
        match = GUARD.match(remaining)
        if not match:
            raise ValueError("unrecognized generated guard syntax")
        guards.append({"operand": json.loads(match[1]), "predicate": match[2]})
        remaining = remaining[match.end():].strip()
        if remaining:
            if not remaining.startswith(",") or not remaining[1:].strip():
                raise ValueError("unrecognized generated guard separator")
            remaining = remaining[1:].strip()
    if len({(g["operand"], g["predicate"]) for g in guards}) != len(guards):
        raise ValueError("duplicate generated guard")
    return sorted(guards, key=lambda g: (g["operand"], g["predicate"]))


def policy_guards(root):
    """Read current policy data without executing/importing the experiment code.

    These are catalogue domains, not a claim about the old sampler or a typed
    adapter. In particular, the legacy raw ReportedRule has no guard field.
    """
    path = root / "scripts/numerical_domains.py"

    def literal_mapping(node):
        if not isinstance(node, ast.Dict):
            raise ValueError("unrecognized policy mapping")
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is not None:
                result[ast.literal_eval(key)] = ast.literal_eval(value)
            elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                  and isinstance(value.func.value, ast.Name)
                  and value.func.value.id == "dict" and value.func.attr == "fromkeys"
                  and len(value.args) == 2 and not value.keywords):
                result.update(dict.fromkeys(ast.literal_eval(value.args[0]),
                                            ast.literal_eval(value.args[1])))
            else:
                raise ValueError("unrecognized policy mapping expansion")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in result.items()):
            raise ValueError("non-string policy mapping")
        return result

    mappings = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"OPERANDS", "POSITIVE", "NONZERO"}:
                if target.id in mappings:
                    raise ValueError("duplicate policy mapping")
                mappings[target.id] = literal_mapping(node.value)
    if set(mappings) != {"OPERANDS", "POSITIVE", "NONZERO"}:
        raise ValueError("missing policy mappings")
    result = {}
    for rule in mappings["OPERANDS"]:
        guards = [{"operand": operand, "predicate": predicate}
                  for name, predicate in (("OPERANDS", "finite"), ("POSITIVE", "positive"),
                                          ("NONZERO", "nonzero"))
                  for operand in mappings[name].get(rule, "")]
        result[rule] = sorted(guards, key=lambda g: (g["operand"], g["predicate"]))
    return result, sha(path.read_bytes())


def semantic_identity(row):
    """A grouping key, not experimental evidence or a raw instance digest."""
    return {
        "rule_id": row["rule_id"],
        "precision": row["precision"],
        "guards": row["guards"],
        "syntax_domain": row["syntax_domain"],
    }


def read_table(root, table):
    path = root / "VeriTile/Triton/Float" / (table + ".lean")
    source = path.read_text()
    metadata_line = next(line for line in source.splitlines()
                         if line.startswith("def reportMetadata :"))
    metadata, = parse_json_literals(metadata_line)
    expected_hashes = metadata["report_sha256"]
    snapshot_match = re.search(r"^def snapshot : String := (" + STRING + r")$", source, re.M)
    if not snapshot_match:
        raise ValueError(f"{table}: missing snapshot")
    snapshot = json.loads(snapshot_match[1])
    if snapshot != sha(canonical(expected_hashes)):
        raise ValueError(f"{table}: snapshot disagrees with report hashes")
    count_match = re.search(r"theorem accepted_count : all.length = (\d+)", source)
    if not count_match or int(count_match[1]) != TABLES[table]:
        raise ValueError(f"{table}: historical accepted_count changed")
    checked_reports = {}
    rows = []
    for match in ROW_START.finditer(source):
        following = NEXT_DECL.search(source, match.end())
        body = source[match.start():following.start() if following else len(source)]
        name, kind = match.groups()
        artifact = field(body, "artifact")
        report_file, fragment = artifact.split("#", 1)
        report_path = (root / report_file).resolve()
        if not report_path.is_relative_to(root.resolve()):
            raise ValueError("report artifact escapes repository")
        if report_file not in checked_reports:
            for filename, expected in expected_hashes.items():
                if sha((report_path.parent / filename).read_bytes()) != expected:
                    raise ValueError(f"{table}: frozen evidence hash mismatch: {filename}")
            checked_reports[report_file] = json.loads(report_path.read_text())
        summary = checked_reports[report_file]
        rule, fmt = field(body, "ruleID"), field(body, "format")
        if fragment != f"{fmt}/{rule}":
            raise ValueError(f"{table}.{name}: artifact selector mismatch")
        selected = [r for r in summary["rows"] if (r["rule"], r["format"]) == (rule, fmt)]
        if len(selected) != 1:
            raise ValueError(f"{table}.{name}: missing/duplicate historical report row")
        original = selected[0]
        if not (original["accept"] is True and original["replayed"] is True
                and original["state"] == "COMPLETE" and original["decision"] == "ACCEPT"
                and original["bias"] == original["vars"] == "PASS"):
            raise ValueError(f"{table}.{name}: historical row is not accepted/replayed PASS")
        payloads = parse_json_literals(body)
        if len(payloads) != 1:
            raise ValueError(f"{table}.{name}: ambiguous frozen row payload")
        payload = payloads[0]
        recorded = payload.get("row", payload) if kind == "ReportedScalarRule" else payload
        if "rule" in recorded and canonical(recorded) != canonical(original):
            raise ValueError(f"{table}.{name}: Lean row differs from original evidence")
        if table == "CountAdmission":
            if payload["rule_id"] != rule or payload["format"] != fmt:
                raise ValueError("count entry identity mismatch")
            if (payload["replicates"] != original["replicates"]
                    or payload["statistics"]["bias"]["upper"] != original["B"]
                    or payload["statistics"]["vars"]["upper"] != original["U"]):
                raise ValueError("count entry statistics disagree with frozen row")
        key = field(body, "key")
        if key != f"report:{snapshot}:{fmt}:{rule}":
            raise ValueError(f"{table}.{name}: report key mismatch")
        guards = []
        if kind == "ReportedScalarRule":
            guard_line = re.search(r"^\s*guards := \[(.*)\]\s*$", body, re.M)
            if not guard_line:
                raise ValueError(f"{table}.{name}: missing guards")
            guards = parse_guards(guard_line[1])
        syntax_domain = None
        if rule == "COUNT-ZERO":
            syntax_domain = {"integer": True, "source_type": "int32", "constant": 0}
        if rule == "COUNT-SUCCESSOR":
            upper = re.search(r"def upperExclusive : Nat := (\d+)", source)
            if not upper:
                raise ValueError("missing bounded count syntax domain")
            syntax_domain = {"operand": "i", "integer": True,
                             "source_type": "int32",
                             "lower_inclusive": 0, "upper_exclusive": int(upper[1])}
        rows.append({
            "declaration": f"VeriTile.Triton.FP.{table}.{name}",
            "file": path.relative_to(root).as_posix(),
            "line": source.count("\n", 0, match.start()) + 1,
            "table": table, "rule_id": rule, "format": fmt,
            "precision": {p: field(body, p) for p in PRECISIONS},
            "guards": guards, "syntax_domain": syntax_domain,
            "guards_origin": ("explicit Admission row" if kind == "ReportedScalarRule"
                              else "raw ReportedRule has no guards field"),
            "report_key": key, "artifact": artifact,
            "report_sha256": expected_hashes,
            "report_row_sha256": sha(canonical(original)),
            "lean_payload_sha256": sha(canonical(payload)),
            "description": field(body, "description"),
        })
    if len(rows) != TABLES[table]:
        raise ValueError(f"{table}: historical generated rows changed")
    all_list = re.search(r"def all : List \w+ := \[(.*?)\]", source, re.S)
    listed = re.findall(r"\b\w+\b", all_list[1]) if all_list else []
    if listed != [r["declaration"].rsplit(".", 1)[1] for r in rows]:
        raise ValueError(f"{table}: all does not preserve every generated historical row")
    return rows


def join_catalogue(root, catalogue, rows, groups):
    """Check exact historical coverage, not Lean proofs or probe semantics."""
    classification = json.loads(catalogue.read_text())
    instances = classification["instances"]
    identities = [r["instance_id"] for r in instances]
    if len(set(identities)) != len(identities) or len(instances) != len(groups):
        raise ValueError("classification must contain exactly 78 distinct semantic instances")
    tables = classification["source_tables"]
    if len(tables) != len(TABLES) or {t["id"] for t in tables} != set(TABLES):
        raise ValueError("classification source tables do not match historical tables")
    for table in tables:
        path = f"VeriTile/Triton/Float/{table['id']}.lean"
        if (table["path"] != path or table["row_count"] != TABLES[table["id"]]
                or table["sha256_at_audit"] != sha((root / path).read_bytes())):
            raise ValueError(f"{table['id']}: classification refers to a different table snapshot")
        row = next(r for r in rows if r["table"] == table["id"])
        if row["report_key"].split(":")[1] != table["report_snapshot"]:
            raise ValueError(f"{table['id']}: classification report snapshot mismatch")
    current_policy, policy_hash = policy_guards(root)
    by_decl = {r["declaration"]: r for r in rows}
    seen = set()
    seen_groups = set()
    origins = Counter()
    for instance in instances:
        if instance["classification"] not in CLASSIFICATIONS:
            raise ValueError("unknown classification")
        if not instance["sources"]:
            raise ValueError("classification instance has no historical source")
        group_keys = set()
        for source in instance["sources"]:
            declaration = source["declaration"]
            if declaration in seen or declaration not in by_decl:
                raise ValueError(f"duplicate/unknown classification source: {declaration}")
            seen.add(declaration)
            row = by_decl[declaration]
            for old, new in (("table", "table"), ("path", "file"), ("line", "line"),
                             ("artifact", "artifact"), ("historical_key", "report_key")):
                if source[old] != row[new]:
                    raise ValueError(f"{declaration}: classification source {old} mismatch")
            for key in ("rule_id", "format", "precision"):
                if instance[key] != row[key]:
                    raise ValueError(f"{declaration}: classification {key} mismatch")
            expected_guards = row["guards"]
            if row["table"] == "ReportedAdmission":
                expected_guards = current_policy[row["rule_id"]]
                expected_origin = "scripts/numerical_domains.py policy; raw ReportedRule has no guards field"
            elif row["table"] == "CountAdmission":
                expected_origin = "empty GuardedFragment guards; count domain is encoded in syntax"
            else:
                expected_origin = "explicit Admission row"
            if instance["guards"] != expected_guards or source["guard_origin"] != expected_origin:
                raise ValueError(f"{declaration}: classification guard/domain origin mismatch")
            origins[expected_origin] += 1
            domain = instance["syntax_domain"]
            if row["rule_id"] == "COUNT-ZERO":
                if domain != "constant int32 zero":
                    raise ValueError("classification count-zero domain mismatch")
            elif row["rule_id"] == "COUNT-SUCCESSOR":
                bound = row["syntax_domain"]["upper_exclusive"]
                if domain != f"0 <= i < {bound}; int32 i+1 cannot overflow" or not 0 < bound <= 2**24:
                    raise ValueError("classification count-successor domain mismatch")
            elif domain is not None:
                raise ValueError(f"{declaration}: unexpected classification syntax domain")
            group_keys.add(canonical(semantic_identity(row)))
        if len(group_keys) != 1 or seen_groups.intersection(group_keys):
            raise ValueError("classification splits or merges distinct historical semantic groups")
        seen_groups.update(group_keys)
        guard_key = ",".join(f"{g['operand']}:{g['predicate']}" for g in instance["guards"]) or "none"
        expected_key = instance["rule_id"] + "::" + "/".join(instance["precision"][p] for p in PRECISIONS) + "::" + guard_key
        expected_id = expected_key + ("::" + instance["syntax_domain"] if instance["syntax_domain"] else "")
        if instance["catalogue_dedup_key"] != expected_key or instance["instance_id"] != expected_id:
            raise ValueError("classification instance identity does not match its fields")
    if seen != set(by_decl) or seen_groups != set(groups):
        raise ValueError("classification omits historical evidence")
    counts = dict(Counter(r["classification"] for r in instances))
    if (classification["counts"]["instances_by_classification"] != counts
            or classification["counts"]["historical_rows"] != len(rows)
            or classification["counts"]["semantic_instances"] != len(groups)):
        raise ValueError("classification summary counts disagree with joined instances")
    return {
        "classification_counts_source_joined": counts,
        "classification_source_rows_verified": len(seen),
        "classification_semantic_groups_verified": len(seen_groups),
        "classification_guard_origins": dict(origins),
        "classification_sha256": sha(catalogue.read_bytes()),
        "derived_guard_policy_sha256": policy_hash,
        "classification_warning": "Source coverage and domain provenance checked; labels, probe semantics and Lean certificates are not proof-verified here. No strict exclusions applied.",
        "historical_guard_warning": "Legacy raw ReportedAdmission has no guards field. Its catalogue guards come from the current policy, not proof that the frozen historical sampler applied that policy or a typed adapter exists for every row.",
    }


def inventory(root, catalogue=None, details=False):
    rows = [row for table in TABLES for row in read_table(root, table)]
    if len({r["report_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate historical report identity")
    groups = defaultdict(list)
    for row in rows:
        groups[canonical(semantic_identity(row))].append(row)
    if len(rows) != 81 or len(groups) != 78:
        raise ValueError(f"historical invariant changed: {len(rows)} rows / {len(groups)} instances")
    result = {
        "historical_report_rows": len(rows), "historical_semantic_instances": len(groups),
        "historical_rows_by_table": dict(Counter(r["table"] for r in rows)),
        "duplicate_semantic_groups": [
            {"identity": json.loads(key), "report_keys": [r["report_key"] for r in group]}
            for key, group in groups.items() if len(group) > 1],
        "evidence_unchanged": True,
        "verified_strict_exclusions": 0,
        "current_empirical_pool_instances_conservative": len(groups),
        "proof_status": "No typed Lean certificate exclusions applied; classification alone cannot remove evidence.",
        "actual_kernel_use_status": "Not counted here; requires the instantiated Lean proof dependency audit.",
    }
    if catalogue:
        result.update(join_catalogue(root, catalogue, rows, groups))
    if details:
        result["historical_rows"] = rows
        result["instances"] = [
            {**json.loads(key), "historical_declarations": [r["declaration"] for r in group]}
            for key, group in groups.items()]
    return result


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "VeriTile/Spec.lean").is_file())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--current", action="store_true",
                        help="verify/evaluate the typed current view with Lean before excluding strict instances")
    parser.add_argument("--receipt", type=Path, help="workspace build receipt for --current")
    parser.add_argument("--lean", type=Path, help="exact Lean executable for --current --receipt")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        result = inventory(root, args.catalogue, args.details)
        if args.current:
            import export_current_fp_rules as current
            view, build = current.lean_view(root, args.receipt, args.lean)
            view = current.checked_view(root, view)
            result.pop("current_empirical_pool_instances_conservative")
            result.update({
                "verified_strict_exclusions": view["counts"]["strict_instances"],
                "verified_current_empirical_pool_instances": view["counts"]["empirical_instances"],
                "current_counts": view["counts"],
                "current_verification": view["verification"],
                "checked_source_receipts": len(build.get("checked_source_receipts", {})),
                "proof_status": "Typed Lean source certificates evaluated after build validation; exact historical domain/precision mapping verified.",
                "build_validation_mode": ("recursive source/object/import receipts" if args.receipt else "fresh lake build"),
            })
        elif args.receipt or args.lean:
            raise ValueError("--receipt and --lean require --current")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
