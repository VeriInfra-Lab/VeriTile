import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Loop
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeLoop

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

theorem body_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    (bodyExecution a fp8 outputFP8 p s v h bytes t).val.pids=s.pids := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem loopExecutionFull {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (initial : State α) (v : LoopValues α)
    (bytes : Int8Memory initial) (count cur : Nat) (current : State α)
    (bound : cur≤count) (same : current.mem=initial.mem)
    (h : LoopRegisters (loopValues a fp8 p initial v cur) current) :
    ∃ final, range a.model "step" cur count 1 (body fp8 outputFP8 p) current=some final ∧
      LoopRegisters (loopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
      HeaderPreserved current final ∧ final.pids=current.pids := by
  have aux : ∀ left cur (current : State α),count-cur=left → cur≤count →
      current.mem=initial.mem →
      LoopRegisters (loopValues a fp8 p initial v cur) current →
      ∃ final,range a.model "step" cur count 1 (body fp8 outputFP8 p) current=some final ∧
        LoopRegisters (loopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
        HeaderPreserved current final ∧ final.pids=current.pids := by
    intro left
    induction left using Nat.strong_induction_on with
    | h left ih =>
      intro cur current hc hb hm hr
      by_cases less : cur<count
      · have byteCurrent : Int8Memory current := by
          intro r o
          simpa only [hm] using bytes r o
        let next := bodyExecution a fp8 outputFP8 p current
          (loopValues a fp8 p initial v cur) hr byteCurrent cur
        have nextMem : next.val.mem=initial.mem :=
          (body_frame a fp8 outputFP8 p current _ hr byteCurrent cur).trans hm
        have nextRegs : LoopRegisters (loopValues a fp8 p initial v (cur+1)) next.val := by
          rw [loopValues,←nextValues_memory a fp8 p initial current hm]
          exact body_values a fp8 outputFP8 p current _ hr byteCurrent cur
        obtain ⟨final,he,hv,hf,hh,hp⟩ := ih (count-(cur+1)) (by omega) (cur+1) next.val
          rfl (by omega) nextMem nextRegs
        refine ⟨final,?_,hv,hf,HeaderPreserved.trans current next.val final
          (body_header a fp8 outputFP8 p current _ hr byteCurrent cur) hh,
          hp.trans (body_pids a fp8 outputFP8 p current _ hr byteCurrent cur)⟩
        rw [range,if_neg (by decide),if_pos less,next.property]
        exact he
      · have equal : cur=count := by omega
        subst cur
        exact ⟨current,by rw [range,if_neg (by decide),if_neg (by omega)],hr,hm,
          HeaderPreserved.refl current,rfl⟩
  exact aux _ _ _ rfl bound same h

#axiomsClean loopExecutionFull

theorem native_body_pids {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    (nativeBodyExecution a fp8 outputFP8 p s v h bytes t).val.pids=s.pids := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem nativeLoopExecutionFull {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (initial : State α) (v : LoopValues α)
    (bytes : Int8Memory initial) (count cur : Nat) (current : State α)
    (bound : cur≤count) (same : current.mem=initial.mem)
    (h : LoopRegisters (nativeLoopValues a fp8 p initial v cur) current) :
    ∃ final, range a.model "step" cur count 1 (selectedBody fp8 outputFP8 p) current=some final ∧
      LoopRegisters (nativeLoopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
      HeaderPreserved current final ∧ final.pids=current.pids := by
  have aux : ∀ left cur (current : State α),count-cur=left → cur≤count →
      current.mem=initial.mem →
      LoopRegisters (nativeLoopValues a fp8 p initial v cur) current →
      ∃ final,range a.model "step" cur count 1 (selectedBody fp8 outputFP8 p) current=some final ∧
        LoopRegisters (nativeLoopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
        HeaderPreserved current final ∧ final.pids=current.pids := by
    intro left
    induction left using Nat.strong_induction_on with
    | h left ih =>
      intro cur current hc hb hm hr
      by_cases less : cur<count
      · have byteCurrent : Int8Memory current := by
          intro r o
          simpa only [hm] using bytes r o
        let next := nativeBodyExecution a fp8 outputFP8 p current
          (nativeLoopValues a fp8 p initial v cur) hr byteCurrent cur
        have nextMem : next.val.mem=initial.mem :=
          (native_body_frame a fp8 outputFP8 p current _ hr byteCurrent cur).trans hm
        have nextRegs : LoopRegisters (nativeLoopValues a fp8 p initial v (cur+1)) next.val := by
          rw [nativeLoopValues,←nativeNextValues_memory a fp8 p initial current hm]
          exact native_body_values a fp8 outputFP8 p current _ hr byteCurrent cur
        obtain ⟨final,he,hv,hf,hh,hp⟩ := ih (count-(cur+1)) (by omega) (cur+1) next.val
          rfl (by omega) nextMem nextRegs
        refine ⟨final,?_,hv,hf,HeaderPreserved.trans current next.val final
          (native_body_header a fp8 outputFP8 p current _ hr byteCurrent cur) hh,
          hp.trans (native_body_pids a fp8 outputFP8 p current _ hr byteCurrent cur)⟩
        rw [range,if_neg (by decide),if_pos less,next.property]
        exact he
      · have equal : cur=count := by omega
        subst cur
        exact ⟨current,by rw [range,if_neg (by decide),if_neg (by omega)],hr,hm,
          HeaderPreserved.refl current,rfl⟩
  exact aux _ _ _ rfl bound same h

#axiomsClean nativeLoopExecutionFull

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
