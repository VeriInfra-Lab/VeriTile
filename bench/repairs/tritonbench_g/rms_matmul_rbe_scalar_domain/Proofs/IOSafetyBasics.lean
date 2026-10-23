import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOByteQKV
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

structure Bounds (p : Arguments) (s : State α) (capacity : RegionName → Nat) : Prop where
  x : ∀ m : Fin 16,∀ k,k<p.K →
    s.pids 0*p.sxb+((programRow p s*16+m.val)%p.M)*p.sxm+k*p.sxk<capacity p.X
  w : ∀ n : Fin 64,∀ k,k<p.K →
    k*p.swk+((programCol p s*64+n.val)%p.N)*p.swn<capacity p.W
  rms : ∀ k,k<p.K → k*p.srms<capacity p.RMS
  out : ∀ i : TileIndex [16,64],outputActive p (programRow p s) (programCol p s) i →
    outputOffset p (s.pids 0) (programRow p s) (programCol p s) i<capacity p.OUT

theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (hs : ∀ st∈code,∀ s : State α,stepSafe M bounds st s)
    (s : State α) : traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil=>exact traceSafeList_nil _ _ _
  | cons st rest ih=>
    apply traceSafeList_cons_intro (hs st (by simp) s)
    intro t _
    exact ih (fun st h=>hs st (by simp [h])) t

theorem prefix_safe (M : Algebra α) (fp8 outputFP8 : Bool) (p : Arguments)
    (capacity : RegionName → Nat) (s : State α) :
    traceSafeList M capacity (prelude fp8 outputFP8 p) s := by
  apply pure_trace
  intro st member t
  cases fp8 <;> cases outputFP8 <;>
    simp only [prelude,source,TypedSource.kernel,ComputeKernel.surfaceBody,
      List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
