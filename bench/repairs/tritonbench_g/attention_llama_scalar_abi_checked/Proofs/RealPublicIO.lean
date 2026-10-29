import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteRows
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteSafety

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealPublicIO
open VeriTile Triton Source RealFiniteContext RealFiniteGeometry
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult MaskCoverage
open Backing
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Whole ordinary grid composition. All CTAs read the same original memory;
the unique-owner merge preserves every exterior raw cell. Actual access safety
is attached to the very same source and prepared predecessor. -/
theorem implements (B : Nat) (c : Config) (scale : ℝ)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ c.Out≠c.K ∧ c.Out≠c.V)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    (RealIO.io B c scale).ImplementsOn (RealLayout.domain B c)
      (RealMathematical.attention B c scale) := by
  intro vals domain original hin
  let initial:=(RealIO.io B c scale).initial vals original
  let s:=initial.withNumPids (RealIO.grid B c).toNumPids
  have memory : s.mem=original.mem := by
    simp only [s,initial,BlockState.withNumPids_mem,ParameterizedAlgorithmProgramIO.initial_mem]
  have input : (RealLayout.memory B c).InputBinding s vals :=
    inputBinding_of_mem B c vals original s memory hin
  obtain ⟨states,he,hv,hf⟩:=RealFiniteRows.rows B c scale hD hN hSP sep hinj vals s input domain
    (RealIO.initial_bound B c scale vals original) (RealIO.initial_undef B c scale vals original)
  let frames : Kernel.GridFrames (Source.kernel c).eraseDType.toAlgKernel (RealIO.grid B c) initial :=
    fun idx=>{
      final:=states (program B c idx)
      writes:=writes B c (program B c idx)
      h_exec:=by
        simpa only [s,context_grid] using he (program B c idx)
      h_writeWithin:=by
        intro r o miss
        exact (hf (program B c idx) r o miss).symm }
  have disjoint : Kernel.GridWritesDisjoint frames := by
    intro i j ne a hi hj
    apply ne
    exact program_injective B c (writes_disjoint B c hinj (program B c i) (program B c j) a hi hj)
  let final:=Kernel.mergeFrames (RealIO.grid B c) initial frames
  let launch : Kernel.GridLaunchedOrdinary (Source.kernel c).eraseDType.toAlgKernel
      (RealIO.grid B c) initial final := ⟨frames,disjoint,rfl⟩
  let trace : AlgorithmExecution ((RealIO.io B c scale).stages vals) initial final :=
    .cons ⟨launch⟩ (.nil final)
  refine ⟨final,trace,?_,?_,?_⟩
  · intro bounds hb
    exact ⟨RealFiniteSafety.stage_safe B c scale bounds original vals hb hD hN hSP,trivial⟩
  · intro o j _
    have ho : o=0 := Fin.eq_zero o
    subst o
    let q:=RealLayout.queryIndex B c j
    let idx:=gridIndex B c (owner (index q))
    have hw : (frames idx).writes (c.Out,IOLayout.output c q) := by
      change writes B c (program B c (gridIndex B c (owner (index q)))) _
      rw [program_gridIndex]
      exact owner_writes B c q
    have hm:=Kernel.mergeFrames_mem_written disjoint idx c.Out (IOLayout.output c q) hw
    change final.readMem c.Out (IOLayout.output c q)=_
    have read : final.readMem c.Out (IOLayout.output c q)=
        (states (owner (index q))).readMem c.Out (IOLayout.output c q) := by
      unfold BlockState.readMem
      rw [hm]
      rfl
    rw [read]
    have slot : RealLayout.querySlot B c q=j := Equiv.apply_symm_apply _ _
    simpa only [slot] using hv q
  · intro r o outside _
    have miss : ¬Kernel.GridWriteFootprint frames (r,o) := by
      rintro ⟨idx,q,_,hr,ho⟩
      have out:=outside 0 (RealLayout.querySlot B c q) trivial
      change r≠c.Out ∨ o≠IOLayout.output c
        (RealLayout.queryIndex B c (RealLayout.querySlot B c q)) at out
      rw [RealLayout.queryIndex_slot] at out
      exact out.elim (fun h=>h hr) (fun h=>h ho)
    exact (Kernel.mergeFrames_mem_unwritten (frames:=frames) r o miss).trans
      (congrFun (congrFun (ParameterizedAlgorithmProgramIO.initial_mem
        (RealIO.io B c scale) vals original) r) o)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealPublicIO
