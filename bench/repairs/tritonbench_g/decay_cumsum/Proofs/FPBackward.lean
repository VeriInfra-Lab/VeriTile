import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPBackwardIteration

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry FPBackwardBase
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_executes (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog initial n u) :
    ∃ v, FP.Structural.rangeInt (engine M) "__rev_j" (Int.ofNat n-Int.ofNat (p.span-1)) 1 1 (backwardBody h p) u=some v ∧
      Holds M h p prog initial p.span v := by
  by_cases hn : n<p.span
  · obtain ⟨v,he,hv⟩ := FPBackwardIteration.iteration M h p prog initial n
      (u.setReg "__rev_j" .int [] (fun _ => Int.ofNat n-Int.ofNat (p.span-1))) hn (by simp)
      (bind_holds M h p prog initial n u inv)
    obtain ⟨w,hw,hf⟩ := loop_executes M h p positive prog initial (n+1) v hv
    have lt : Int.ofNat n-Int.ofNat (p.span-1)<1 := by simp only [Int.ofNat_eq_natCast]; omega
    have inc : Int.ofNat n-Int.ofNat (p.span-1)+1=Int.ofNat (n+1)-Int.ofNat (p.span-1) := by
      simp only [Int.ofNat_eq_natCast,Nat.cast_add,Nat.cast_one]
      omega
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.rangeInt]
    simpa only [Nat.one_ne_zero,if_false,lt,if_true,he,bind,Option.bind_some,Nat.cast_one,inc] using hw
  · have bound := inv.bound
    have eq : n=p.span := by omega
    have done : ¬Int.ofNat n-Int.ofNat (p.span-1)<1 := by simp only [Int.ofNat_eq_natCast]; omega
    refine ⟨u,?_,by simpa [eq] using inv⟩
    rw [FP.Structural.rangeInt]
    simp only [Nat.one_ne_zero,if_false,done]
termination_by p.span-n

theorem source_correct (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (scale : ℝ) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.source .backward h p scale) u=some v ∧
      (∀ r∈outputRegions, ∀ (t : Fin p.span) (j : Fin h.BK),
        v.mem r (off h p prog t.val j)=FPReference.backwardStored M r (output M h p prog u r t.val (j,PUnit.unit))) ∧
      FPMemory.Frame h p prog outputRegions u v := by
  obtain ⟨v,he,inv⟩ := loop_executes M h p positive prog u 0 (FPPrefix.initial M .backward h p prog u)
    (initial_holds M h p positive prog u)
  refine ⟨v,?_,fun r member t j => inv.output r member t.val (by omega) t.isLt j,inv.frame⟩
  have start : (0:Int)-(Int.ofNat p.span-1)=Int.ofNat 0-Int.ofNat (p.span-1) := by
    simp only [Int.ofNat_eq_natCast]
    omega
  change run (engine M) (Kernels.source .backward h p scale).surfaceBody u=some v
  rw [FPProgram.source_body,run_append,FPPrefix.source_init,FPPrefix.executes M .backward h p prog u hf hc hr]
  simp only [Option.bind_some,run,FPProgram.loopCode,step,FP.Structural.evalOp]
  change (FP.Structural.rangeInt (engine M) "__rev_j" ((0:Int)-(Int.ofNat p.span-1)) 1 1
    (backwardBody h p) (FPPrefix.initial M .backward h p prog u)).bind (fun w => some w)=some v
  rw [start,he]
  simp [run]

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackward
