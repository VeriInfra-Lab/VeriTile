#!/usr/bin/env python3
"""Render the single task ledger; acceptance is an explicit review decision.

Historical acceptance, prior contract review and the kernel-only revision are separate.
This utility neither infers correctness from files nor reruns corpus proofs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "bench/tritonbench_g/task_progress.json"
REPORT = LEDGER.with_suffix(".md")
STATUSES = (
    "verified",
    "verified_with_deferred_numerics",
    "source_defect_pending_repair",
    "pending",
)


def validate(data):
    rows = data["tasks"]
    names = [row["task"] for row in rows]
    assert len(names) == len(set(names)) == 183, "Expected 183 distinct tasks"
    assert "fp4_to_bf16_conversion" not in names
    for row in rows:
        assert row["retained_acceptance"]["status"] in STATUSES
        revision = row["contract_revision"]
        assert revision["status"] in STATUSES
        if revision["status"] in STATUSES[:2]:
            assert revision["stage"] == "accepted"
            assert revision.get("source"), row["task"]
            assert revision.get("public_entries"), row["task"]
            assert revision.get("validation"), row["task"]
            assert not revision["blockers"], row["task"]
            for path in [revision["source"], *revision["public_entries"]]:
                assert (ROOT / path).is_file(), path
        if revision["status"] == STATUSES[1]:
            assert revision.get("numerical_assumptions"), row["task"]
        kernel = row["kernel_revision"]
        assert kernel["status"] in STATUSES, row["task"]
        if kernel["status"] in STATUSES[:2]:
            assert kernel["stage"] == "accepted", row["task"]
            assert not kernel["blockers"], row["task"]
            assert kernel.get("proof_scope") and kernel.get("validation"), row["task"]
            assert kernel.get("public_entries"), row["task"]
            for path in [kernel["source"], kernel["kernel_source"], *kernel["public_entries"]]:
                assert (ROOT / path).is_file(), path
        if kernel["status"] == STATUSES[1]:
            assert kernel.get("deferred_experiments"), row["task"]
        if kernel["status"] == STATUSES[0]:
            assert not kernel.get("deferred_experiments"), row["task"]
        for experiment in kernel.get("auxiliary_numerical_experiments", []):
            assert experiment["scope"] == "auxiliary_native_result", row["task"]
            assert experiment["status"] == "deferred experiment", row["task"]
            assert experiment.get("formal") and experiment.get("required_evidence"), row["task"]
        for path in kernel.get("auxiliary_entries", []):
            assert (ROOT / path).is_file(), path
        for path in kernel.get("misc_entries", []):
            assert "/Misc/" in path and (ROOT / path).is_file(), path
        public_io = kernel.get("public_io_format")
        if public_io:
            assert public_io["status"] in ("complete", "in_progress"), row["task"]
            assert public_io.get("inspection"), row["task"]
            if public_io["status"] == "complete":
                assert public_io.get("report") and public_io.get("validation"), row["task"]
                assert not public_io.get("blockers"), row["task"]
    assert names == sorted(names), "Keep task order stable"
    public_io = data.get("public_io_format", {})
    if "semantically_reviewed" in public_io:
        reviewed = sum(row["kernel_revision"].get("public_io_format", {}).get("semantic_review")
                       == "current" for row in rows)
        assert public_io["semantically_reviewed"] == reviewed
        assert public_io["completed"] <= reviewed <= 183
    joint = public_io.get("latest_combined_entry_check")
    if joint:
        assert joint["passed"] + len(joint["failed_tasks"]) == joint["tasks"]
        assert 0 <= joint["tasks"] <= 183
        assert joint["reused"] == 0, "Final combined checks must run afresh"
        assert set(joint["failed_tasks"]) <= set(names)
    organization = public_io.get("entry_organization_review")
    if organization:
        assert organization["reviewed"] == 183
        assert set(organization["changed_tasks"]) <= set(names)
        assert (ROOT / organization["report"]).is_file()
    pairs = data.get("program_pair_review")
    if pairs:
        assert pairs["reviewed"] == len(rows)
        assert (ROOT / pairs["report"]).is_file()
        reviews = [row["kernel_revision"]["program_pair_review"] for row in rows]
        assert pairs["counts"] == dict(Counter(review["status"] for review in reviews))
        for row, review in zip(rows, reviews):
            complete = review["status"] in ("pair_verified", "fixed_pair_verified")
            assert complete == (row["kernel_revision"]["status"] in STATUSES[:2]), row["task"]
            assert review["entry_validation"]["exit_code"] == 0, row["task"]
            assert bool(review["actual_pair_declarations"]) or not complete, row["task"]
            assert not row["kernel_revision"]["blockers"] if complete else row["kernel_revision"]["blockers"], row["task"]
    misc = data.get("misc_layout")
    if misc:
        assert misc["tasks"] == len(rows)
        assert misc["primary_misc_dependencies"] == 0
        assert misc["entry_checks_passed"] == len(rows)
        assert (ROOT / misc["report"]).is_file()
        for row in rows:
            kernel = row["kernel_revision"]
            assert all("/Misc/" not in p for p in kernel["public_entries"])
            assert all("/Misc/" in p for p in kernel.get("auxiliary_entries", []))
            assert set(kernel.get("auxiliary_entries", [])) <= set(kernel["misc_entries"])
    cross = data.get("cross_review")
    if cross:
        assert cross["tasks"] == len(rows)
        assert cross["fresh_changed_task_entry_checks"] + cross["unchanged_task_entry_checks_reused"] == len(rows)
        assert cross["primary_misc_dependencies"] == 0
        assert (ROOT / cross["report"]).is_file()
        assert (ROOT / cross["audit"]).is_file()
        assert all(row["kernel_revision"].get("cross_review") for row in rows)
    latest = data.get("latest_completion")
    if latest:
        assert latest["completed_tasks"] == len(latest["tasks"])
        assert set(latest["tasks"]) <= set(names)
        assert latest["after"] == dict(Counter(row["kernel_revision"]["status"] for row in rows))
        assert latest["entry_checks"]["tasks"] == latest["entry_checks"]["passed"] == len(latest["tasks"])
        assert (ROOT / latest["report"]).is_file()
        assert (ROOT / latest["audit"]).is_file()
        assert all(row["kernel_revision"]["stage"] == "accepted"
                   for row in rows if row["task"] in latest["tasks"])
    current_rules = data.get("current_fp_rule_use")
    if current_rules:
        uses = current_rules["experimental_consumers"]
        admitted = set(current_rules["existing_admitted_rule_tasks"])
        deferred = set(current_rules["deferred_rule_tasks"])
        assert admitted.isdisjoint(deferred)
        assert admitted | deferred == set(uses) <= set(names)
        assert deferred == {row["task"] for row in rows
                            if row["kernel_revision"].get("deferred_experiments")}
        assert current_rules["counts"] == {
            "tasks": len(rows), "strict_or_structural": len(rows) - len(uses),
            "existing_admitted_rule_tasks": len(admitted),
            "deferred_rule_tasks": len(deferred),
        }
        assert all((ROOT / use["evidence"]).is_file() for use in uses.values())
    readability = data.get("source_readability_review")
    if readability:
        assert readability["tasks"] == len(rows)
        assert (ROOT / readability["report"]).is_file()
        assert (ROOT / readability["audit"]).is_file()
        reviews = [row["kernel_revision"]["source_readability_review"] for row in rows]
        assert readability["counts"] == dict(Counter(r["status"] for r in reviews))
        assert readability["primary_misc_dependencies"] == 0
        for review in reviews:
            assert review["status"] in ("no_issue_identified", "fixed", "cleanup_pending")
            assert bool(review["remaining_issues"]) == (review["status"] == "cleanup_pending")
            assert review["report"] == readability["report"]
            assert (ROOT / review["review_evidence"]).is_file()
            if review["status"] == "fixed":
                assert review.get("fix_validation"), "Source cleanup requires preservation evidence"
    entry_roles = data.get("main_entry_role_review")
    if entry_roles:
        assert entry_roles["tasks"] == len(rows)
        assert entry_roles["primary_misc_dependencies"] == 0
        assert entry_roles["remaining_reader_role_issues"] == 0
        assert (ROOT / entry_roles["report"]).is_file()
        assert (ROOT / entry_roles["audit"]).is_file()
        reviews = [row["kernel_revision"]["main_entry_role_review"] for row in rows]
        assert entry_roles["counts"] == dict(Counter(r["status"] for r in reviews))
        for review in reviews:
            assert review["status"] in ("fixed", "no_issue_identified")
            assert review["audit"] == entry_roles["audit"]
            assert (ROOT / review["review_evidence"]["path"]).is_file()
    io_organization = data.get("io_organization_review")
    if io_organization:
        assert io_organization["tasks"] == len(rows)
        assert io_organization["remaining_io_organization_issues"] == 0
        assert (ROOT / io_organization["report"]).is_file()
        assert (ROOT / io_organization["audit"]).is_file()
        reviews = [row["kernel_revision"]["io_organization_review"] for row in rows]
        assert io_organization["counts"] == dict(Counter(r["status"] for r in reviews))
        for review in reviews:
            assert review["status"] in ("fixed", "no_issue_identified")
            assert review["audit"] == io_organization["audit"]
            assert (ROOT / review["review_evidence"]["path"]).is_file()
            assert review["canonical_io_sources"]
            assert not review["remaining_issues"]
            if review["status"] == "fixed":
                assert review["preservation_and_build_passed"]
    formulas = data.get("main_formula_review")
    if formulas:
        assert formulas["tasks"] == len(rows)
        assert (ROOT / formulas["report"]).is_file()
        assert (ROOT / formulas["audit"]).is_file()
        reviews = [row["kernel_revision"]["main_formula_review"] for row in rows]
        assert formulas["counts"] == dict(Counter(r["status"] for r in reviews))
        assert formulas["remaining_substantive_gaps"] == sum(
            r["status"] == "substantive_gap" for r in reviews)
        for review in reviews:
            assert review["status"] in ("fixed", "unchanged", "substantive_gap")
            assert review["audit"] == formulas["audit"]
            assert (ROOT / review["review_evidence"]["path"]).is_file()
            assert bool(review["remaining_issues"]) == (review["status"] == "substantive_gap")
            assert review["verification_status"] in (
                "targeted_passed", "unchanged_reused", "passed_with_recorded_limit")
    fp_review = data.get("fp_rule_review")
    if fp_review:
        reviews = [row["kernel_revision"]["fp_rule_review"] for row in rows]
        for review in reviews:
            assert review["interface_action"] in (
                "none", "abstract_main_interface", "separate_host_abi")
            assert review["correct_action"] in ("none", "separate_real_specification")
            assert isinstance(review["new_experiment_required"], bool)
            assert review["report"] == fp_review["report"]
            assert review["implementation_status"] == (
                "pending" if review["interface_action"] != "none" or
                review["correct_action"] != "none" else "no_action_identified")
        assert (ROOT / fp_review["report"]).is_file()
        counts = fp_review["counts"]
        assert counts["tasks"] == len(reviews)
        assert counts["interface_pending"] == sum(
            review["interface_action"] != "none" for review in reviews)
        assert counts["real_specification_pending"] == sum(
            review["correct_action"] != "none" for review in reviews)
        assert counts["active_admitted_rule_tasks"] == sum(
            bool(review["active_admitted_rules"]) for review in reviews)
        assert counts["new_experiment_tasks"] == sum(
            review["new_experiment_required"] for review in reviews)
        if "rule_status" in counts:
            assert counts["rule_status"] == dict(Counter(
                review["rule_status"] for review in reviews))
            assert counts["rule_issue_tasks"] == sum(
                bool(review["rule_issues"]) for review in reviews)


def fp_review_summary(data):
    current = data.get("current_fp_rule_use")
    if current:
        counts = current["counts"]
        return [
            f"Current main-pair numerical use in {counts['tasks']} tasks: "
            f"{counts['strict_or_structural']} use strict-model or structural proofs; "
            f"{counts['existing_admitted_rule_tasks']} use existing admitted rules; "
            f"{counts['deferred_rule_tasks']} depend on deferred numerical evidence.",
            "Existing admitted-rule consumers: " + ", ".join(current["existing_admitted_rule_tasks"]) + ".",
            "Deferred numerical evidence: " + ", ".join(current["deferred_rule_tasks"]) + ". "
            "Strict software-model certificates do not claim unconditional hardware conformance.",
            "The earlier FP review remains a historical snapshot in the JSON. "
            "Imports and unused rule parameters do not count as actual rule use.",
            "See [current rule use and experiment queue](../../documents/FPRemainingAdmissionGaps.md).",
            "",
        ]
    review = data.get("fp_rule_review")
    if not review:
        return []
    counts = review["counts"]
    report = "../" + review["report"].removeprefix("bench/")
    if "rule_status" in counts:
        kinds = counts["rule_status"]
        return [
            f"FP numerical-rule review of existing results in {counts['tasks']} tasks; "
            f"{kinds.get('exact_or_structural', 0)} classified exact/model or structural, "
            f"{kinds.get('admitted_numerics', 0)} using existing admitted rules; "
            f"{counts['rule_issue_tasks']} rule-evidence issues identified, "
            f"{counts['new_experiment_tasks']} new experiments required.",
            "Concrete models and proved local laws are allowed in main results. "
            "Generic scalar facts are shared without requiring all kernel interfaces "
            "to be parameterized.",
            "This classification includes retained single-source results in pending tasks; "
            "actual two-program completion is counted separately below.",
            "Accepted numerical rules remain explicit proof premises. Pending experiments "
            "for auxiliary results are counted separately from primary completion; "
            "replay/GPU status is unchanged.",
            "See [current rule use and experiment queue](../../documents/FPRemainingAdmissionGaps.md) "
            f"and the [shared-law validation]({report}).",
            "",
        ]
    return [
        f"FP rule layering review: {counts['tasks']} tasks recorded; "
        f"{counts['interface_pending']} interface actions pending, "
        f"{counts['real_specification_pending']} independent Real specifications need separation; "
        f"{counts['active_admitted_rule_tasks']} tasks use existing admitted rules; "
        f"{counts['new_experiment_tasks']} new numerical experiments required by this review.",
        "This classifies proof evidence and remaining interface work; it does not implement "
        "those changes, retract accepted model theorems, clear existing deferrals, "
        "or claim new proof replay/GPU validation.",
        f"See the [refined FP rule review]({report}); per-task evidence is in this ledger's JSON.",
        "",
    ]


def render(data):
    rows = data["tasks"]
    retained = Counter(row["retained_acceptance"]["status"] for row in rows)
    current = Counter(row["contract_revision"]["status"] for row in rows)
    stages = Counter(row["contract_revision"]["stage"] for row in rows)
    kernel_counts = Counter(row["kernel_revision"]["status"] for row in rows)
    kernel_stages = Counter(row["kernel_revision"]["stage"] for row in rows)
    auxiliary_experiments = [
        (row["task"], experiment)
        for row in rows
        for experiment in row["kernel_revision"].get("auxiliary_numerical_experiments", [])
    ]
    lines = [
        "# TritonBench-G task progress",
        "",
        "Generated from `task_progress.json` by `python bench/task_progress.py`.",
        "This is one checklist with separately recorded review scopes. Historical accepted results",
        "retain their original source, contract, numerical assumptions and review.",
        "They are not withdrawn when a new contract revision is pending, and do not",
        "automatically certify changed Python or Lean files in this workspace.",
        "Archived review decisions remain unchanged. A concrete counterexample may",
        "correct their retained applicability; each such correction records its",
        "evidence explicitly in the JSON and task documentation.",
        "",
        "The active kernel-only revision follows the approved add_value pattern: explicit",
        "justified input checks, matching public theorem premises, source precision,",
        "kernel results, and reuse of applicable proofs. The wrapper supplies input",
        "assertions; host execution and returned-Tensor proofs are optional. Original source",
        "defects remain separate from accepted repairs. Module/file counts are not",
        "task completions. GPU execution and independent replay are separate fields.",
        "The prior contract review is retained at its recorded source hashes. Its",
        "counts do not certify that those tasks satisfy the new kernel-only standard,",
        "or that edits in progress have been accepted. The kernel-only row records",
        "the accepted proof scope. Public IO format completion is counted separately below.",
        "",
        *([data["numerical_classification_note"], ""]
          if data.get("numerical_classification_note") else []),
        *fp_review_summary(data),
        *(["Actual two-program review: " + ", ".join(
            f"{name}: {count}" for name, count in sorted(data["program_pair_review"]["counts"].items())
        ) + ".",
            "Single-source native results remain valid auxiliary theorems; they do not complete a two-program pair. "
            "Missing targets and program-proof connections are not numerical deferrals.",
            "See [per-task program pairs and blockers](../" +
            data["program_pair_review"]["report"].removeprefix("bench/") + ").", ""]
          if data.get("program_pair_review") else []),
        *([f"Tracked optional-entry layout: {data['misc_layout']['tasks']}/183 tasks inspected; "
            "The tracked NativeCorrect, TensorBridge and other optional entry files are under Misc/. "
            "Correct contains independent Real correctness; FPEquiv contains actual program equivalence. "
            "Required helpers stay in Proofs/ and the primary import closure contains no Misc/ module.",
            "This move adds no FP32/native or host correctness requirement and does not change proof completion. "
            "Tracked auxiliary checks and other optional (including historical) files remain separate. This baseline does not certify every declaration inside required helpers; the source-readability review records any mixed helper modules.",
            "See [layout and validation](../" + data['misc_layout']['report'].removeprefix('bench/') + ").", ""]
          if data.get("misc_layout") else []),
        *([f"Standard-review baseline: {data['final_standard_review']['tasks']}/183 tasks reviewed; "
            f"{data['final_standard_review']['entry_checks_passed']} fresh per-task entry checks passed. "
            "Source/contract, independent mathematical targets, actual program pairs and rule evidence "
            "were reviewed; internal compiled proofs were reused. This is not independent full-source "
            "replay or GPU validation.",
            "See [findings and remaining proof gaps](../" +
            data['final_standard_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('final_standard_review') else []),
        *([f"Cross-review baseline: {data['cross_review']['tasks']}/183 tasks reread by rotated reviewers; "
            f"{data['cross_review']['fresh_changed_task_entry_checks']} changed tasks checked afresh, "
            f"{data['cross_review']['unchanged_task_entry_checks_reused']} unchanged task checks reused after hash verification. "
            "That review did not change completion counts or add numerical deferrals. No GPU run or full-source replay.",
            "See [cross-review baseline and scoped validation](../" +
            data['cross_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('cross_review') else []),
        *([f"Latest completion: {data['latest_completion']['completed_tasks']} whole tasks accepted "
            f"({', '.join(data['latest_completion']['tasks'])}); "
            f"{data['latest_completion']['entry_checks']['passed']} fresh main/auxiliary entry checks passed. "
            f"New numerical deferrals: {data['latest_completion'].get('new_numerical_deferrals', 0)}; "
            "no GPU run or independent full-source replay.",
            "See [completed tasks and source/proof entries](../" +
            data['latest_completion']['report'].removeprefix('bench/') + ").", ""]
          if data.get('latest_completion') else []),
        *(["Cross-task combined import: not passed. The current shared generation has a "
            "generated-declaration conflict; KV retains its separately checked reduceMin generation. "
            "These compatibility items are tracked separately from individual task acceptance "
            "and numerical evidence in [bench/TODO.md](../TODO.md).", ""]
          if data.get('current_cross_task_import_check', {}).get('status') == 'not_passed' else []),
        *([f"Source readability review: {data['source_readability_review']['tasks']}/183 tasks inspected; "
            + ", ".join(f"{key}: {value}" for key, value in
                        sorted(data['source_readability_review']['counts'].items())) + ".",
            "This checks readable authoritative original/candidate source and file roles. "
            "Confirmed source-organization work is tracked separately from proof completion; "
            "existing strict proofs are not withdrawn solely for readability. "
            "A passing entry check does not certify this separate source standard.",
            "See [source construction findings and validated cleanups](../" +
            data['source_readability_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('source_readability_review') else []),
        *([f"Main-entry role review: {data['main_entry_role_review']['tasks']}/183 tasks; "
            f"{data['main_entry_role_review']['counts']['fixed']} organized, "
            f"{data['main_entry_role_review']['counts']['no_issue_identified']} unchanged; "
            "no remaining auxiliary proof conclusions in selected readers or main-to-Misc imports. "
            "Genuine multi-kernel/stage headlines retained; proof completion and numerical scope unchanged.",
            "See [per-declaration roles and targeted validation](../" +
            data['main_entry_role_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('main_entry_role_review') else []),
        *([f"IO organization review: {data['io_organization_review']['tasks']}/183 tasks; "
            f"{data['io_organization_review']['counts']['fixed']} consolidated, "
            f"{data['io_organization_review']['counts']['no_issue_identified']} unchanged. "
            "Selected readers reuse canonical IO data; genuinely different encodings and schedules retained. "
            "Changed tasks have definition-preservation and scoped build evidence; this is not full-source replay.",
            "See [per-task IO sources, preservation and validation limits](../" +
            data['io_organization_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('io_organization_review') else []),
        *([f"Main-formula review: {data['main_formula_review']['tasks']}/183 tasks; "
            + ", ".join(f"{key}: {value}" for key, value in
                        sorted(data['main_formula_review']['counts'].items())) + ".",
            "Each review records the independent Real formula, required domain, actual program/grid/stage scope, "
            "proof connection and FP program pair. Targeted checks reuse applicable internal proofs; "
            "recorded replay limits remain separate from proof-completion and numerical-experiment counts.",
            "See [per-task formulas, conditions and verification limits](../" +
            data['main_formula_review']['report'].removeprefix('bench/') + ").", ""]
          if data.get('main_formula_review') else []),
        "| Scope | Completed | Numerical validation deferred | Source defect pending repair | Remaining |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Kernel-only standard (active) | " + " | ".join(str(kernel_counts[s]) for s in STATUSES) + " |",
        "| Retained historical results | " + " | ".join(str(retained[s]) for s in STATUSES) + " |",
        "| Prior contract review | " + " | ".join(str(current[s]) for s in STATUSES) + " |",
        "",
        "Completed primary results include proofs under already admitted numerical rules. "
        "This does not claim unconditional IEEE equality.",
        f"Auxiliary numerical experiments pending: {len(auxiliary_experiments)}" +
        (" (" + "; ".join(f"{task}: {e['id']}" for task, e in auxiliary_experiments) + ")."
         if auxiliary_experiments else "."),
        "",
        "Kernel-only stages: " + ", ".join(f"{k}: {v}" for k, v in sorted(kernel_stages.items())) + ".",
        "Prior contract stages: " + ", ".join(f"{k}: {v}" for k, v in sorted(stages.items())) + ".",
        "",
        "Historical provenance is pinned in the JSON. It refers to the earlier",
        "checkout; historical evidence trees have not been duplicated. See each",
        "current task README for its exact formula, input domain and validation.",
        "Shared launcher grid-product compatibility and historical cleanup remain",
        "deferred in [`bench/TODO.md`](../TODO.md).",
        "",
        "Tensor applications, check correspondence and retained auxiliary results are listed separately",
        "from the primary kernel results. Public entries state logical input/output contracts;",
        "operational State/run/memory plumbing belongs in supporting proofs, and Tensor metadata",
        "and admission belong in the bridge. Preserved proof acceptance does not imply IO-format completion.",
        *(["Strict kernel/Tensor interface split: " + ", ".join(
            f"{name}: {count}" for name, count in sorted(Counter(
                row["kernel_revision"]["file_organization"]["disposition"] for row in rows
            ).items())) + ". Each row's JSON records expanded interface checks and targeted builds."]
          if data.get("kernel_interface_split") else []),
        "",
        *(["Public IO format (per-task review and targeted builds): " + ", ".join(
            f"{name}: {count}" for name, count in sorted(Counter(
                row["kernel_revision"]["public_io_format"]["status"] for row in rows
            ).items())) + ". All 183 selected entry pairs are tracked; completion requires whole-task",
            "source/IO/independent-formula/safety/output/frame connections and targeted builds.",
            *([f"Current semantic interface reviews: {data['public_io_format']['semantically_reviewed']}/183; "
               f"awaiting dependency refresh: {data['public_io_format']['awaiting_dependency_refresh']}. "
               "Refreshing build dependencies does not withdraw unchanged interface reviews or proof acceptance."]
              if "semantically_reviewed" in data["public_io_format"] else []),
            *([f"Latest combined public-entry check: {data['public_io_format']['latest_combined_entry_check']['passed']}/"
               f"{data['public_io_format']['latest_combined_entry_check']['tasks']} passed; "
               f"{len(data['public_io_format']['latest_combined_entry_check']['failed_tasks'])} failed; "
               f"{data['public_io_format']['latest_combined_entry_check']['reused']} reused. "
               f"Checked at {data['public_io_format']['latest_combined_entry_check']['checked_at']}.",
               ("Final batch acceptance remains pending. Failed tasks: " + ", ".join(
                    data['public_io_format']['latest_combined_entry_check']['failed_tasks']) + "."
                if data['public_io_format']['latest_combined_entry_check']['failed_tasks'] else
                "The combined-entry gate passed for that object snapshot."),
               "Combined checks apply to the recorded source/object snapshot. Subsequent source/object changes "
               "require a fresh check; targeted build receipts alone do not establish this gate."]
              if data['public_io_format'].get('latest_combined_entry_check') else []),
            *([f"Public entry organization: {data['public_io_format']['entry_organization_review']['reviewed']}/183 reviewed; "
               f"{len(data['public_io_format']['entry_organization_review']['changed_tasks'])} tasks reorganized. "
               "This checks headline/helper roles and preserves prior semantic reviews; "
               "it is not a new whole-corpus semantic audit.",
               "See [entry organization and final combined check](../" +
               data['public_io_format']['entry_organization_review']['report'].removeprefix('bench/') + ")."]
              if data['public_io_format'].get('entry_organization_review') else []),
            "Shared regional IO makes no automatic flat-pointer claim. Independent proof replay",
            "and GPU execution remain at their separately recorded status.", ""]
          if data.get("public_io_format") else []),
        "| Task | Retained result | Original status | Prior contract review | Kernel-only revision | Public IO format | Source readability | Auxiliary entries | Outstanding work |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        revision = row["contract_revision"]
        kernel = row["kernel_revision"]
        name = row["task"]
        readme = kernel.get("readme")
        if not readme and kernel.get("kernel_source"):
            candidate_readme = Path(kernel["kernel_source"]).with_name("README.md")
            if (ROOT / candidate_readme).is_file():
                readme = str(candidate_readme)
        if not readme:
            readme = revision.get("readme")
        if readme and (ROOT / readme).is_file():
            name = f"[{name}](../{readme.removeprefix('bench/')})"
        retained_row = row["retained_acceptance"]
        retained_status = retained_row["status"]
        if retained_row.get("applicability_correction"):
            retained_status += "; applicability corrected with concrete evidence (see JSON)"
        current_scope = revision["status"] + " / " + revision["stage"]
        if revision.get("scope_note"):
            current_scope += "; " + revision["scope_note"]
        public_io = kernel.get("public_io_format", {})
        io_status = public_io.get("status", "not yet recorded")
        pair_review = kernel.get("program_pair_review")
        if io_status != "complete" and pair_review:
            io_status = "reviewed; " + pair_review["status"]
        elif io_status != "complete" and public_io.get("semantic_review") == "current":
            io_status = "reviewed; build validation pending"
        outstanding = list(dict.fromkeys([*kernel["blockers"], *public_io.get("blockers", [])]))
        outstanding.extend("Auxiliary numerical experiment pending: " + experiment["id"]
                           for experiment in kernel.get("auxiliary_numerical_experiments", []))
        fp_review = kernel.get("fp_rule_review", {})
        action = fp_review.get("interface_action", "none")
        if action == "abstract_main_interface":
            outstanding.append("FP main interface abstraction pending; retain proved model facts")
        elif action == "separate_host_abi":
            outstanding.append("Separate host ABI binding from main FP interface")
        if fp_review.get("correct_action", "none") == "separate_real_specification":
            outstanding.append("Separate independent Real specification from concrete FP profile")
        fields = [name, retained_status, row["original_status"],
                  current_scope,
                  kernel["status"] + " / " + kernel["stage"],
                  io_status,
                  kernel.get("source_readability_review", {}).get("status", "not reviewed"),
                  ", ".join(f"[{Path(p).stem}](../{p.removeprefix('bench/')})"
                            for p in kernel.get("auxiliary_entries", [])) or "—",
                  "; ".join(outstanding) or "None; see numerical/replay fields"]
        lines.append("| " + " | ".join(s.replace("|", "\\|").replace("\n", " ") for s in fields) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the ledger and rendered table")
    args = parser.parse_args()
    data = json.loads(LEDGER.read_text())
    validate(data)
    report = render(data)
    if args.check:
        assert REPORT.read_text() == report, "Run python bench/task_progress.py"
    else:
        REPORT.write_text(report)
    for scope in ("kernel_revision", "retained_acceptance", "contract_revision"):
        print(scope, dict(Counter(row[scope]["status"] for row in data["tasks"])))
    if data.get("public_io_format"):
        print("public_io_format", dict(Counter(
            row["kernel_revision"]["public_io_format"]["status"] for row in data["tasks"])))
        if "semantically_reviewed" in data["public_io_format"]:
            print("public_io_semantic_reviews", data["public_io_format"]["semantically_reviewed"])
        joint = data["public_io_format"].get("latest_combined_entry_check")
        if joint:
            print("latest_combined_entry_check", {
                "checked": joint["tasks"], "passed": joint["passed"],
                "failed": len(joint["failed_tasks"]), "reused": joint["reused"]})


if __name__ == "__main__":
    main()
