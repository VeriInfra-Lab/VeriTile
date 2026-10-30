import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardLoop
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardPost
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardResults
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
variable {α : Type} [Inhabited α]

def result (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α) :=
  let middle := KernelForwardReferenceMemory.memory M s f scale bh iv u s.T
  if f.final then KernelStateMemory.store M s "ht" bh iv
    (KernelForwardLoop.carry M s f bh iv u s.T) middle else middle

theorem executes (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,FP.Structural.exec (engine M) (source .forward s f scale) u=some v ∧
      v.mem=(result M s f scale bh iv u).mem := by
  let prelude := KernelPrefix.forward M s f scale u
  obtain ⟨middle,loopSuccess,inv⟩ := KernelForwardLoop.executes M s f scale bh iv u 0 prelude.val
    (KernelForwardLoop.initial M s f scale bh iv u hv hk hh)
  let post := KernelForwardPost.execute M s f scale bh iv
    (KernelForwardLoop.carry M s f bh iv u s.T) middle inv.regs.carry inv.static
  have loopRun : run (engine M) [KernelProgram.loop .forward s f scale] prelude.val=some middle := by
    simp only [KernelProgram.loop,run,step,loopSuccess]
    rfl
  refine ⟨post.val,?_,?_⟩
  · change run (engine M) (source .forward s f scale).surfaceBody u=some post.val
    rw [source_body,List.append_assoc,run_append,prelude.property]
    simp only [Option.bind_some]
    rw [run_append,loopRun]
    exact post.property
  · change (KernelForwardPost.execute M s f scale bh iv _ middle inv.regs.carry inv.static).val.mem=_
    rw [KernelForwardPost.memory]
    unfold result
    cases hf : f.final <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact inv.memory
    · exact KernelStateMemory.mem_congr M s "ht" bh iv _ _ _ inv.memory

structure Facts (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u v : State α) : Prop where
  success : FP.Structural.exec (engine M) (source .forward s f scale) u=some v
  outputs : ∀ (t : Fin s.T) (j : Fin s.FBV),iv*s.FBV+j.val<s.V →
    v.mem "o" (valAddress s bh iv s.FBV t.val j.val)=KernelReference.stored M
      (KernelReference.output M s (KernelReference.forwardInput M s f bh iv u) scale t.val j)
  residuals : ∀ (t : Fin s.T) (j : Fin s.FBV),iv*s.FBV+j.val<s.V →
    v.mem "v" (valAddress s bh iv s.FBV t.val j.val)=KernelReference.stored M
      (KernelReference.residual M s (KernelReference.forwardInput M s f bh iv u) t.val j)
  terminal : f.final=Bool.true → ∀ (i : Fin s.BK) (j : Fin s.FBV),i.val<s.K → iv*s.FBV+j.val<s.V →
    v.mem "ht" (stateAddress s bh iv s.FBV i.val j.val)=KernelReference.stored M
      (KernelForwardLoop.carry M s f bh iv u s.T (j,i,PUnit.unit))
  frame : ∀ r off,¬KernelForwardResults.Writes s f bh iv r off → v.mem r off=u.mem r off

theorem complete (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State α)
    (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,Facts M s f scale bh iv u v := by
  obtain ⟨v,success,memory⟩ := executes M s f scale bh iv u hv hk hh
  refine ⟨v,success,?_,?_,?_,?_⟩
  · intro t j active
    rw [memory]
    unfold result
    cases hf : f.final <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact KernelForwardResults.output M s f scale bh iv u s.T t.val t.isLt j active
    · rw [KernelStateMemory.frame M s "ht" _ _ _ _ "o" _ (Or.inl (by decide))]
      exact KernelForwardResults.output M s f scale bh iv u s.T t.val t.isLt j active
  · intro t j active
    rw [memory]
    unfold result
    cases hf : f.final <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact KernelForwardResults.residual M s f scale bh iv u s.T t.val t.isLt j active
    · rw [KernelStateMemory.frame M s "ht" _ _ _ _ "v" _ (Or.inl (by decide))]
      exact KernelForwardResults.residual M s f scale bh iv u s.T t.val t.isLt j active
  · intro final i j hi hj
    rw [memory]
    simp only [result,final,ite_true]
    exact KernelStateMemory.cell M s "ht" bh iv _ _ i j hi hj
  · intro r off outside
    rw [memory]
    have rowFrame := KernelForwardResults.frame M s f scale bh iv u s.T r off (fun h => outside (Or.inl h))
    unfold result
    cases hf : f.final <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact rowFrame
    · rw [KernelStateMemory.frame]
      · exact rowFrame
      · by_cases different : r≠"ht"
        · exact Or.inl different
        · right
          intro i j hi hj same
          exact outside (Or.inr ⟨hf,not_ne_iff.mp different,i,j,hi,hj,same⟩)

#axiomsClean executes
#axiomsClean complete
#specNonCircular result avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForward
