import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOByteExecution
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.PrefixValues

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def loop (native : Bool) (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : LoopValues α :=
  if native then nativeLoopValues a fp8 p s v t else loopValues a fp8 p s v t

theorem loop_succ (native : Bool) (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) :
    loop native a fp8 p s v (t+1)=values native a fp8 p s (loop native a fp8 p s v t) t := by
  cases native <;> rfl

theorem values_memory (native : Bool) (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s t : State α) (same : t.mem=s.mem) (v : LoopValues α) (n : Nat) :
    values native a fp8 p t v n=values native a fp8 p s v n := by
  cases native
  · exact nextValues_memory a fp8 p s t same v n
  · exact nativeNextValues_memory a fp8 p s t same v n

theorem loop_regions (native : Bool) (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (regions : PointerRegions p v) (n : Nat) :
    PointerRegions p (loop native a fp8 p s v n) := by
  induction n with
  | zero=>cases native <;> exact regions
  | succ n ih=>
    rw [loop_succ]
    cases native <;> exact ⟨ih.weight,ih.rms⟩

theorem initial_regions (a : Arithmetic α) (p : Arguments) (s : State α) :
    PointerRegions p (initialValues a p s) := ⟨fun _=>rfl,fun _=>rfl⟩

theorem execution (a : Arithmetic α) (native fp8 outputFP8 : Bool)
    (p : Arguments) (initial : State α) (v : LoopValues α)
    (bytes : InputBytes fp8 p initial) (regions : PointerRegions p v)
    (count cur : Nat) (current : State α) (bound : cur≤count) (same : current.mem=initial.mem)
    (h : LoopRegisters (loop native a fp8 p initial v cur) current) :
    ∃ final,range a.model "step" cur count 1 (code native fp8 outputFP8 p) current=some final ∧
      LoopRegisters (loop native a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
      HeaderPreserved current final ∧ final.pids=current.pids := by
  have aux : ∀ left cur (current : State α),count-cur=left → cur≤count →
      current.mem=initial.mem → LoopRegisters (loop native a fp8 p initial v cur) current →
      ∃ final,range a.model "step" cur count 1 (code native fp8 outputFP8 p) current=some final ∧
        LoopRegisters (loop native a fp8 p initial v count) final ∧ final.mem=initial.mem ∧
        HeaderPreserved current final ∧ final.pids=current.pids := by
    intro left
    induction left using Nat.strong_induction_on with
    | h left ih=>
      intro cur current hc hb hm hr
      by_cases less : cur<count
      · have byteCurrent : InputBytes fp8 p current := by
          intro fp r member off
          simpa only [hm] using bytes fp r member off
        have pointerRegions:=loop_regions native a fp8 p initial v regions cur
        let next:=iteration a native fp8 outputFP8 p current
          (loop native a fp8 p initial v cur) hr byteCurrent pointerRegions cur
        have nextMem : next.val.mem=initial.mem :=
          (iteration_frame a native fp8 outputFP8 p current _ hr byteCurrent pointerRegions cur).trans hm
        have nextRegs : LoopRegisters (loop native a fp8 p initial v (cur+1)) next.val := by
          rw [loop_succ,←values_memory native a fp8 p initial current hm]
          exact iteration_values a native fp8 outputFP8 p current _ hr byteCurrent pointerRegions cur
        obtain ⟨final,he,hv,hf,hh,hp⟩:=ih (count-(cur+1)) (by omega) (cur+1) next.val
          rfl (by omega) nextMem nextRegs
        refine ⟨final,?_,hv,hf,HeaderPreserved.trans current next.val final
          (iteration_header a native fp8 outputFP8 p current _ hr byteCurrent pointerRegions cur) hh,
          hp.trans (iteration_pids a native fp8 outputFP8 p current _ hr byteCurrent pointerRegions cur)⟩
        rw [range,if_neg (by decide),if_pos less,next.property]
        exact he
      · have equal : cur=count := by omega
        subst cur
        exact ⟨current,by rw [range,if_neg (by decide),if_neg (by omega)],hr,hm,
          HeaderPreserved.refl current,rfl⟩
  exact aux _ _ _ rfl bound same h

#axiomsClean execution
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
