import bench.tritonbench_g.rms_matmul_rbe.Proofs.LoopFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def loopValues {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (initial : State α) (v : LoopValues α) : Nat → LoopValues α
  | 0 => v
  | t+1 => nextValues a fp8 p initial (loopValues a fp8 p initial v t) t

theorem nextValues_memory {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (initial current : State α) (same : current.mem=initial.mem)
    (v : LoopValues α) (t : Nat) :
    nextValues a fp8 p current v t=nextValues a fp8 p initial v t := by
  have hx : xTile a p current v t=xTile a p initial v t := by
    funext i
    simp only [xTile,same]
  have hr : rmsTile a fp8 p current v t=rmsTile a fp8 p initial v t := by
    funext i
    simp only [rmsTile,same]
  have hw : weightTile a fp8 p current v t=weightTile a fp8 p initial v t := by
    funext i
    simp only [weightTile,same]
  have hs : scaledTile a fp8 p current v t=scaledTile a fp8 p initial v t := by
    funext i
    simp only [scaledTile,hx,hr]
  simp only [nextValues,hx,hw,hs]

/-- Constructive execution of every K64 iteration of the shared source.
The recurrence uses the original memory because no loop statement stores. -/
theorem loopExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (initial : State α) (v : LoopValues α)
    (bytes : Int8Memory initial) (count cur : Nat) (current : State α)
    (bound : cur≤count) (same : current.mem=initial.mem)
    (h : LoopRegisters (loopValues a fp8 p initial v cur) current) :
    ∃ final, range a.model "step" cur count 1 (body fp8 outputFP8 p) current=some final ∧
      LoopRegisters (loopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
      HeaderPreserved current final := by
  have aux : ∀ left cur (current : State α),count-cur=left → cur≤count →
      current.mem=initial.mem →
      LoopRegisters (loopValues a fp8 p initial v cur) current →
      ∃ final,range a.model "step" cur count 1 (body fp8 outputFP8 p) current=some final ∧
        LoopRegisters (loopValues a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
        HeaderPreserved current final := by
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
        obtain ⟨final,he,hv,hf,hh⟩ := ih (count-(cur+1)) (by omega) (cur+1) next.val
          rfl (by omega) nextMem nextRegs
        refine ⟨final,?_,hv,hf,HeaderPreserved.trans current next.val final
          (body_header a fp8 outputFP8 p current _ hr byteCurrent cur) hh⟩
        rw [range,if_neg (by decide),if_pos less,next.property]
        exact he
      · have equal : cur=count := by omega
        subst cur
        exact ⟨current,by rw [range,if_neg (by decide),if_neg (by omega)],hr,hm,
          HeaderPreserved.refl current⟩
  exact aux _ _ _ rfl bound same h

#axiomsClean loopExecution
#specNonCircular loopValues avoiding [bodyExecution,run,range]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
