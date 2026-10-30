import VeriTile.Triton.Float.ScalarProbeExact
import VeriTile.Triton.Float.MaximumProbeExact
import VeriTile.Triton.Float.PrecisionProbeExact
import VeriTile.Triton.Float.ExponentialProbeExact
import VeriTile.Triton.Float.CountConversion
import VeriTile.Triton.Float.CountAdmission
import VeriTile.Triton.Float.LogAdmission

/-! Current evidence view, separate from the immutable historical admissions.

Each strict source selects a fixed, typed source equality; its proof is stored
in `StrictEntry`. Metadata strings describe that source and match its complete
precision/domain signature to historical rows. A theorem-name string cannot
construct an entry. The model identities below describe selected software
models, not GPU conformance or equality of arbitrary opaque primitives.

Legacy ReportedAdmission has no guard field. Its domain below is an explicit
current source-domain adapter, not a claim about the frozen report's sampler.
Historical report keys, payloads, tables and counts are preserved verbatim.
-/
namespace VeriTile.Triton.FP.CurrentAdmission
open ScalarProbeExact

structure Signature where
  ruleID : String
  input : String
  compute : String
  accumulator : String
  output : String
  guards : List OperandGuard
  syntaxDomain : String := ""
  deriving DecidableEq, Repr

instance : BEq Signature := ⟨fun a b => decide (a = b)⟩

private def formatName : Format → String
  | .fp32 => "fp32"
  | .bf16 => "bf16"

private def profileSignature (p : Profile) (ruleID : String)
    (guards : List OperandGuard) : Signature :=
  ⟨ruleID, formatName p.inputFormat, formatName p.computeFormat,
    formatName p.accumulatorFormat, formatName p.outputFormat, guards, ""⟩

private def finiteGuards (operands : List String) : List OperandGuard :=
  operands.map (fun name => ⟨name, .finite⟩)

inductive StrictSource where
  | scalar (profile : Profile) (rule : ScalarProbeExact.Rule)
  | maximum (profile : Profile) (rule : MaximumProbeExact.Rule)
  | precision (ruleInstance : PrecisionProbeExact.Instance)
  | count (atom : CountConversion.Atom)
  | exponential (profile : Profile) (backend : Exponential.Backend)
      (rule : ExponentialProbeExact.Rule)

def CountSourceSound (a : CountConversion.Atom) : Prop :=
  Guarded.StrictFragmentSound (CountConversion.lhs a) (CountConversion.rhs a) ∧
  match a with
  | .zero => CountConversion.Exact.fromInt32 0 = Value.zero .fp32
  | .successor => ∀ i : BitVec 32, 0 ≤ i.toInt → i.toInt < (16777216 : Int) →
      CountConversion.Exact.fromInt32 (i + 1) =
        FP.add .ieeeValue (CountConversion.Exact.fromInt32 i)
          (CountConversion.Exact.fromNat 1)

/-- The proposition is determined by the source index, never supplied by a
caller. All operands, conversions, order, output casts and guards remain in
the source-specific certificate types. -/
def StrictSource.Sound : StrictSource → Prop
  | .scalar p r => ScalarProbeExact.Certificate p r
  | .maximum p r => MaximumProbeExact.Certificate p r
  | .precision i => PrecisionProbeExact.Certificate i
  | .count a => CountSourceSound a
  | .exponential p b r => ExponentialProbeExact.Certificate p b r

theorem StrictSource.certificate (source : StrictSource) : source.Sound := by
  cases source with
  | scalar p r => exact ScalarProbeExact.all_certificates p r
  | maximum p r => exact MaximumProbeExact.all_certificates p r
  | precision i => exact PrecisionProbeExact.all_certificates i
  | count a =>
    refine ⟨CountConversion.strict_sound a, ?_⟩
    cases a with
    | zero => exact CountConversion.Exact.count_int32_zero
    | successor => exact CountConversion.Exact.count_int32_successor
  | exponential p b r => exact ExponentialProbeExact.certificate p b r

def StrictSource.signature : StrictSource → Signature
  | .scalar p r => profileSignature p r.ruleID r.guards
  | .maximum p r => profileSignature p r.ruleID r.guards
  | .precision i =>
    let p := match i with
      | .roundIdemFp32 | .accWidenFp32 => Profile.fp32
      | .roundIdemBf16 _ false | .castMoveBf16 false => Profile.bf16
      | .roundIdemBf16 _ true | .castMoveBf16 true => Profile.bf16Fp32
    profileSignature p i.ruleID (finiteGuards i.operands)
  | .count .zero =>
    ⟨"COUNT-ZERO", "int32", "fp32", "fp32", "fp32", [], "constant int32 zero"⟩
  | .count .successor =>
    ⟨"COUNT-SUCCESSOR", "int32", "fp32", "fp32", "fp32", [],
      "0 <= i < 16777216; int32 i+1 cannot overflow"⟩
  | .exponential p b r =>
    let ruleID := match b, r with
      | .tl, .zero => "EXP-ZERO"
      | .tl, .negInfSub => "EXP-NEG-INF-SUB"
      | .libdevice, .zero => "EXP-ZERO-LIBDEVICE"
      | .libdevice, .negInfSub => "EXP-NEG-INF-SUB-LIBDEVICE"
    profileSignature p ruleID (match r with | .zero => [] | .negInfSub => finiteGuards ["a"])

def StrictSource.modelID : StrictSource → String
  | .scalar _ _ => "VeriTile.ScalarProbeExact/ieeeValue/typed-load-rnd-store"
  | .maximum _ _ => "VeriTile.MaximumProbeExact/ieeeValue/maxnum/fp32-nodes-typed-store"
  | .precision (.roundIdemFp32) | .precision (.accWidenFp32) =>
    "VeriTile.PrecisionIdentityLaws/sourceCopies/opaque-ordered-arithmetic/typed-store"
  | .precision _ =>
    "VeriTile.PrecisionIdentityLaws/ieeeValue-bf16-fp32-casts/opaque-ordered-arithmetic/typed-store"
  | .count _ => "VeriTile.CountConversion/ieeeValue/signed-int32/bounded-2^24"
  | .exponential _ .tl _ =>
    "VeriTile.ExponentialProbeExact/ieeeValue/tl-exp-log2e-ptx-ex2-special-values"
  | .exponential _ .libdevice _ =>
    "VeriTile.ExponentialProbeExact/ieeeValue/libdevice-expf-special-values"

def StrictSource.certificateDeclaration : StrictSource → String
  | .scalar _ _ => "VeriTile.Triton.FP.ScalarProbeExact.all_certificates"
  | .maximum _ _ => "VeriTile.Triton.FP.MaximumProbeExact.all_certificates"
  | .precision _ => "VeriTile.Triton.FP.PrecisionProbeExact.all_certificates"
  | .count _ => "VeriTile.Triton.FP.CurrentAdmission.StrictSource.certificate"
  | .exponential _ _ _ => "VeriTile.Triton.FP.ExponentialProbeExact.certificate"

structure StrictEntry where
  source : StrictSource
  proof : source.Sound

private def profiles : List Profile := [.fp32, .bf16, .bf16Fp32]

def strictSources : List StrictSource :=
  (profiles.flatMap fun p =>
    [StrictSource.scalar p .addCommute, .scalar p .mulCommute,
      .scalar p .mulOne, .scalar p .divOne,
      .maximum p .commute, .maximum p .associate,
      .maximum p .idempotent, .maximum p .negativeInfinity]) ++
  (PrecisionProbeExact.all.map StrictSource.precision) ++
  [.count .zero, .count .successor] ++
  (profiles.flatMap fun p =>
    [StrictSource.exponential p .tl .zero, .exponential p .tl .negInfSub]) ++
  [.exponential .fp32 .libdevice .zero, .exponential .fp32 .libdevice .negInfSub]

def currentStrict : List StrictEntry :=
  strictSources.map fun source => ⟨source, source.certificate⟩

structure HistoricalRow where
  table : String
  report : ReportedRule
  guards : List OperandGuard
  guardOrigin : String

/-- Adapter domains from the current formal source/registry policy. The
legacy report itself contains neither these fields nor a proof of sampling
the whole guarded domain. Unknown relations receive no invented guards. -/
def legacyGuards (ruleID : String) : List OperandGuard :=
  let operands := match ruleID with
    | "ADD-COMMUTE" | "MUL-COMMUTE" | "CANCEL" | "DIV-RCP" | "CAST-MOVE" => ["a", "b"]
    | "ADD-ASSOC" | "MUL-ASSOC" | "MUL-DISTRIB" | "FMA-CONTRACT"
      | "CAST-REMOVE" | "ACC-WIDEN" => ["a", "b", "c"]
    | "ROUND-IDEM" | "BF16-WIDEN-RETURN" => ["a"]
    | _ => []
  finiteGuards operands ++ if ruleID = "DIV-RCP" then [⟨"b", .nonzero⟩] else []

private def scalarRows (table : String) (rows : List ReportedScalarRule) :
    List HistoricalRow :=
  rows.map fun row => ⟨table, row.report, row.guards, "explicit Admission row"⟩

def historical : List HistoricalRow :=
  (ReportedAdmission.all.map fun row =>
    ⟨"ReportedAdmission", row, legacyGuards row.ruleID,
      "current source/policy adapter; raw historical ReportedRule has no guards field"⟩) ++
  scalarRows "SupplementalAdmission" SupplementalAdmission.all ++
  (CountAdmission.all.map fun row =>
    ⟨"CountAdmission", row, [], "count domain is encoded in source syntax"⟩) ++
  scalarRows "ExpAdmission" ExpAdmission.all ++ scalarRows "LogAdmission" LogAdmission.all

def HistoricalRow.signature (row : HistoricalRow) : Signature :=
  ⟨row.report.ruleID, row.report.input, row.report.compute,
    row.report.accumulator, row.report.output, row.guards,
    if row.report.ruleID = "COUNT-ZERO" then "constant int32 zero"
    else if row.report.ruleID = "COUNT-SUCCESSOR" then
      "0 <= i < " ++ toString CountAdmission.upperExclusive ++
        "; int32 i+1 cannot overflow" else ""⟩

/-- A row can be removed from the current empirical view only by finding a
proof-carrying source with exactly its relation, four precision fields and
domain. Historical evidence is returned unchanged for every retained row. -/
def HistoricalRow.strict? (row : HistoricalRow) : Option StrictEntry :=
  currentStrict.find? fun entry => decide (entry.source.signature = row.signature)

def currentEmpiricalRows : List HistoricalRow := historical.filter fun row => row.strict?.isNone
def currentStrictRows : List HistoricalRow := historical.filter fun row => row.strict?.isSome
def currentEmpiricalInstances : List Signature :=
  (currentEmpiricalRows.map HistoricalRow.signature).eraseDups
def historicalInstances : List Signature := (historical.map HistoricalRow.signature).eraseDups

theorem strict_entry_sound (entry : StrictEntry) : entry.source.Sound := entry.proof

set_option maxRecDepth 8192 in
set_option maxHeartbeats 8000000 in
theorem inventory_counts :
    (historical.length, historicalInstances.length, currentStrict.length,
      currentStrictRows.length, currentEmpiricalRows.length, currentEmpiricalInstances.length) =
    (81, 78, 42, 44, 37, 36) := by decide +kernel

private def guardJson (guard : OperandGuard) : Lean.Json :=
  Lean.Json.mkObj [("operand", Lean.toJson guard.operand), ("predicate", Lean.toJson (match guard.kind with
    | .finite => "finite" | .nonzero => "nonzero" | .positive => "positive"))]

def Signature.toJson (sig : Signature) : Lean.Json := Lean.Json.mkObj [
  ("rule_id", Lean.toJson sig.ruleID),
  ("precision", Lean.Json.mkObj [("input", Lean.toJson sig.input), ("compute", Lean.toJson sig.compute),
    ("accumulator", Lean.toJson sig.accumulator), ("output", Lean.toJson sig.output)]),
  ("guards", .arr (sig.guards.map guardJson).toArray),
  ("syntax_domain", if sig.syntaxDomain = "" then .null else Lean.toJson sig.syntaxDomain)]

def StrictEntry.toJson (entry : StrictEntry) : Lean.Json := Lean.Json.mkObj [
  ("signature", entry.source.signature.toJson), ("model_id", Lean.toJson entry.source.modelID),
  ("certificate_declaration", Lean.toJson entry.source.certificateDeclaration)]

def HistoricalRow.toJson (row : HistoricalRow) : Lean.Json := Lean.Json.mkObj [
  ("table", Lean.toJson row.table), ("report_key", Lean.toJson row.report.key),
  ("artifact", Lean.toJson row.report.artifact), ("format", Lean.toJson row.report.format),
  ("signature", row.signature.toJson), ("guard_origin", Lean.toJson row.guardOrigin),
  ("evidence_kind", Lean.toJson (if row.strict?.isSome then "strict-model-proof" else "empirical"))]

/-- Exported metadata is derived from the proof-carrying Lean values. The
Python exporter must first check the module and its dependency build receipts;
this JSON alone is never a certificate. It contains references, not copies of
the frozen experimental payloads. -/
def exportJson : Lean.Json := Lean.Json.mkObj [
  ("schema_version", Lean.toJson (1 : Nat)),
  ("scope", Lean.toJson "selected-model scalar rule pool; actual kernel usage is audited separately"),
  ("strict", .arr (currentStrict.map StrictEntry.toJson).toArray),
  ("historical_rows", .arr (historical.map HistoricalRow.toJson).toArray),
  ("counts", Lean.Json.mkObj [("historical_rows", Lean.toJson historical.length),
    ("historical_instances", Lean.toJson historicalInstances.length),
    ("strict_instances", Lean.toJson currentStrict.length),
    ("strict_historical_rows", Lean.toJson currentStrictRows.length),
    ("empirical_historical_rows", Lean.toJson currentEmpiricalRows.length),
    ("empirical_instances", Lean.toJson currentEmpiricalInstances.length)])]

#axiomsClean StrictSource.certificate
#axiomsClean strict_entry_sound
#axiomsClean inventory_counts

end VeriTile.Triton.FP.CurrentAdmission
