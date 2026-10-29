import bench.tritonbench_g.attention_fwd_triton3.Proofs.MathematicalSpecification
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealIO
noncomputable section

/-- Unchecked KV loads require whole key blocks. Positive inner stride and
sufficient row stride prevent output overlap; M/L/Out are distinct regions.
Active address bounds are supplied by the displayed IO windows. -/
specification attention_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (originalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs) := by
  exact RealContract.implements BM BN BD Z finish p Bool.false hN hflag hd hr hm hlm hlo

specification attention_reciprocal_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (reciprocalIO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs) := by
  exact RealContract.implements BM BN BD Z finish p Bool.true hN hflag hd hr hm hlm hlo

set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Reuse of the complete six-input Real formula, finite execution, safety,
readback and frame at the selected FP32 source profile. This remains a physical
per-program contract. The original host's unrestricted logical tensor target
is disproved by original_fp32_logical_view_defect in FPEquiv: a lazy-negative
V view has an unrepresented logical sign. No whole-host acceptance is claimed. -/
specification attention_fp32_correctness (BM BN BD Z finish : Nat) (p : Parameters)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    Spec.Real (fp32IO BM BN BD Z finish p ⊨ fun start _ xs => expected BM BN BD finish start p xs) := by
  exact attention_correctness BM BN BD Z finish p hN hflag hd hr hm hlm hlo

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.Correctness
