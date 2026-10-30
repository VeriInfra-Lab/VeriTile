import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import bench.tritonbench_g.chunk_gla_fwd.Proofs.PairWhere
import VeriTile.Meta.StatementAudit

/-! Reuse the existing complete-source commutation proof on the selected
zero-padded quotation. No original-source execution alias is used here.

Complete split intra-subblock equivalence. Finite checks apply to the actual
reduction result and scale inside each reachable column iteration. All lanes
are checked because where evaluates both arms. Only mul_commute is used;
source loads, fp32 casts, sum order, causal mask and pointer advances remain.
The host's intermediate A has fp32 storage. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.PairSplit
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 18000
set_option maxHeartbeats 8000000

def requirements (kernel : ComputeKernel) (BC : Nat) (scale : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody (fun code => (Source.dynamicRange? code "j").isSome)
    let loop ← Source.dynamicRange? active.yes "j"
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "j" loop.start loop.stop loop.stride loop.body
        (.mulScalarFP32 (Source.prefixBefore loop.body "b_A" 2) BC []
          (.ref .real [BC] "b_A") (.const scale)))).toList
section
variable (qs ks gs : TypedSplit.Storage) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
def original : Program where
  kernel := TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := requirements (TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC) BC scale
def scaleFirst : Program :=
  { original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedSplit.scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC }
theorem split_equiv (R : Spec.Assumptions FP.GuardedFragment) :
    original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC := by
  cases qs <;> cases ks <;> cases gs <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions split_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaFwd.PairSplit
