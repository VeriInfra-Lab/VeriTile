import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIOResults
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndices
open VeriTile Triton FP.Structural Safety
open KernelProducerGrid (Geometry)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def original (p : Geometry) := Source.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p))
def candidate (p : Geometry) := Source.unwrappedQuantize (KernelProducer.parameters (KernelProducerGrid.parameters p))

theorem modulo_values (v : Values α .nat [1]) (rows : Nat) (h : ∀ i,v i<rows) :
    bop (fun x y : Nat=>x%y) .scalarR v (fun _=>rows)=v := by
  funext i
  exact Nat.mod_eq_of_lt (h i)

theorem head_run (M : Algebra α) (p : Geometry) (s : State α) (valid : s.pids 0<p.rows) :
    run M ((original p).surfaceBody.take 3) s=run M ((candidate p).surfaceBody.take 3) s := by
  simp [original,candidate,Source.quantize,Source.unwrappedQuantize,KernelProducer.parameters,
    KernelProducerGrid.parameters,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,numeric,bop,Option.bind,integralMod]
  rw [modulo_values]
  intro i
  have hz : i.1.val=0:=by omega
  simpa [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,hz] using valid

theorem pure_safe (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (h : ∀ st∈code,∀ s : State α,stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil M bounds s
  | cons st rest ih =>
    exact (traceSafeList_cons M bounds st rest s).mpr
      ⟨h st (by simp) s,fun next _=>ih (fun stmt hs=>h stmt (by simp [hs])) next⟩

theorem head_safe (M : Algebra α) (p : Geometry) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList M bounds ((original p).surfaceBody.take 3) s ∧
    traceSafeList M bounds ((candidate p).surfaceBody.take 3) s := by
  constructor <;> apply pure_safe
  all_goals simp [original,candidate,Source.quantize,Source.unwrappedQuantize,KernelProducer.parameters,
    KernelProducerGrid.parameters,ComputeKernel.surfaceBody]
  all_goals simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem execution (M : Algebra α) (p : Geometry) (s : State α) (valid : s.pids 0<p.rows) :
    exec M (original p) s=exec M (candidate p) s := by
  have left : (original p).surfaceBody=(original p).surfaceBody.take 3 ++ (original p).surfaceBody.drop 3 := rfl
  have right : (candidate p).surfaceBody=(candidate p).surfaceBody.take 3 ++ (original p).surfaceBody.drop 3 := rfl
  change run M (original p).surfaceBody s=run M (candidate p).surfaceBody s
  rw [left,right,run_append,run_append,head_run M p s valid]

theorem safety (M : Algebra α) (p : Geometry) (bounds : RegionName → Nat) (s : State α)
    (valid : s.pids 0<p.rows) : TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s := by
  have left : (original p).surfaceBody=(original p).surfaceBody.take 3 ++ (original p).surfaceBody.drop 3 := rfl
  have right : (candidate p).surfaceBody=(candidate p).surfaceBody.take 3 ++ (original p).surfaceBody.drop 3 := rfl
  unfold TraceSafe
  rw [left,right,traceSafeList_append,traceSafeList_append]
  simp only [(head_safe M p bounds s).1,(head_safe M p bounds s).2,true_and,head_run M p s valid]

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndices
