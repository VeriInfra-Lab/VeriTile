import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightLoop
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightMemory
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPWeightProgram FPWeightStep FPWeightLoop FPSource FPMemory
noncomputable section
set_option maxRecDepth 25000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def expectedHeap (M : Algebra α) (h : Call) (K pid : Nat) (u : State α) :=
  commit "dw" (fun i : TileIndex [K] => pid*K+i.1.val)
    (fun i => Cell.mk .real (carry M h K pid u h.rows i))
    (fun i => pid*K+i.1.val<h.cols) (TileShape.allIndices [K]) u.mem

theorem tail_executes (M : Algebra α) (h : Call) (K pid : Nat) (initial u : State α)
    (inv : Holds M h K pid initial h.rows u) :
    ∃ v, run (engine M) (tailCode h K) u=some v ∧ v.mem=expectedHeap M h K pid initial := by
  simp [tailCode,Kernels.weightBackward,Kernels.weightSource,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,FP.Structural.store,
    inv.col,inv.mask,inv.acc,inv.mem,expectedHeap,columns,masks,FPReference.cast,fold_mem]
  <;> (try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write,update])
  <;> rfl

theorem source_memory (M : Algebra α) (h : Call) (K : Nat) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.weightBackward h K) u=some v ∧
      v.mem=expectedHeap M h K (u.pids 0) u := by
  obtain ⟨v,he,inv⟩ := loop_executes M h K (u.pids 0) u 0 (FPWeightStep.initial M h K u)
    (initial_holds M h K u)
  obtain ⟨w,hw,hmem⟩ := tail_executes M h K (u.pids 0) u v inv
  refine ⟨w,?_,hmem⟩
  change run (engine M) (Kernels.weightBackward h K).surfaceBody u=some w
  rw [source_body,run_append,run_append,prefix_executes]
  simp [run,step,FP.Structural.evalOp]
  change (rangeInt (engine M) "row" 0 (h.rows : Int) 1 (loopBody h K)
    (FPWeightStep.initial M h K u)).bind (fun a => run (engine M) (tailCode h K) a) = some w
  simp only [Int.natCast_zero] at he
  rw [he]
  exact hw

def footprint (h : Call) (K pid : Nat) (q : RegionName) (o : Nat) : Prop :=
  q="dw" ∧ ∃ j : Fin K,pid*K+j.val<h.cols ∧ o=pid*K+j.val

theorem complete (M : Algebra α) (h : Call) (K : Nat) (u : State α) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.weightBackward h K) u=some v ∧
      (∀ j : Fin K,u.pids 0*K+j.val<h.cols → v.mem "dw" (u.pids 0*K+j.val)=
        FPWeightReference.output M h.gemma
          (fun row => FPWeightStep.raw M h K (u.pids 0) u "x" row (j,PUnit.unit))
          (fun row => FPWeightStep.raw M h K (u.pids 0) u "dx" row (j,PUnit.unit))
          (saved u) h.rows) ∧
      (∀ q o,¬footprint h K (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm⟩ := source_memory M h K u
  refine ⟨v,he,?_,?_⟩
  · intro j hj
    rw [hm]
    exact commit_readback "dw" (fun i : TileIndex [K] => u.pids 0*K+i.1.val)
      (fun i => Cell.mk .real (carry M h K (u.pids 0) u h.rows i))
      (fun i => u.pids 0*K+i.1.val<h.cols) u (j,PUnit.unit) hj (row_injective _ _)
  · intro q o outside
    rw [hm]
    unfold expectedHeap
    apply commit_frame
    by_cases same : q="dw"
    · right
      intro i _ active eq
      exact outside ⟨same,⟨i.1,active,eq⟩⟩
    · exact Or.inl same

#axiomsClean source_memory
#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightMemory
