import bench.tritonbench_g.attention_fwd_triton3.Proofs.Mathematical_RealContract

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
noncomputable section

/-- Source parameters: query/KV head counts; independent batch/head and
row/inner strides; Q/K/V/M/L/Out regions and context lengths; scale; INIT,
even-row flags and signed window selection. Every use is visible below. -/
abbrev Parameters := RealPrologue.Parameters
abbrev Inputs := RealInputs.Inputs

/-- The exact decimal is the source's log2 scale conversion, retained in the
real projection. Inactive checked Q rows contribute zero. -/
def score (BM BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (k : Fin p.layout.kvContext) : ℝ :=
  (∑ d : Fin BD,
    (if p.evenM ≠ 0 ∨ start * BM + r.val < p.layout.queryContext then xs.query r d else 0) *
      xs.key k d) * (p.scale * 1.4426950408889634)

def initial (BM BD : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => if p.init ≠ 0 then ⊥ else some (xs.center r)
  denominator := fun r => if p.init ≠ 0 then 1 else xs.denominator r
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1

/-- For selected (score,value) pairs, m is their maximum together with the
initial center. The denominator and numerator are rescaled initial carry plus
Σ 2^(score-m) and Σ 2^(score-m)*value. Bottom centers have weight zero. -/
def summarize (seed : RealFormula.RowState) (pairs : List (ℝ × ℝ)) : RealFormula.RowState :=
  let m := seed.center ⊔ RealFormula.maxScore pairs
  { center := m,
    denominator := seed.denominator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m (fun _ => 1),
    numerator := seed.numerator * RealFormula.weight seed.center m + RealFormula.sumAt pairs m Prod.snd }

/-- Regular windows keep 0 ≤ query-key+offset < size; the complement arm
keeps query-key+offset ≥ size. No loop iterations preserve the initial state,
which differs from executing a block whose row has no kept keys. -/
def row (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (initialValue : ℝ) (v : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    summarize seed
      (((List.finRange p.layout.kvContext).filter
        (fun k => RealKeys.keep start BM r.val p.window p.complement p.offset p.size k.val)).map
        (fun k => (score BM BD start p xs r k, v k)))
  else seed

def carry (BM BN BD start : Nat) (p : Parameters) (xs : Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (row BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (row BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun k => xs.value k i.2.1)).numerator

/-- END stores M+log2(L) and Out/L, preserving L memory. Continuation stores
all three carry values. Real division/log are total; bottom stored M reads as
zero under the framework's finite real-memory view. -/
def expected (BM BN BD finish start : Nat) (p : Parameters)
    (xs : Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD :=
  let c := carry BM BN BD start p xs
  { center := fun r => (c.center r).elim 0 (fun m =>
      if decide (finish ≠ 0) then m + Real.log (c.denominator r) / Real.log 2 else m),
    denominator := c.denominator,
    output := fun i => if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i }


end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
