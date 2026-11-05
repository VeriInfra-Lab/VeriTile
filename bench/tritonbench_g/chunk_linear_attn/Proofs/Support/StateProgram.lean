import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRange
import bench.tritonbench_g.chunk_linear_attn.Proofs.Support.StateFinish
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def stateLoopStmt (p : Parameters) (scale : ℝ) (rev seed final : Bool) : ComputeStmt :=
  if rev then
    let actual := stateIntParts p scale seed final
    .forRangeInt "__rev_i_t" actual.start actual.stop actual.stride actual.body
  else .forRange "i_t" 0 p.NT 1 (stateParts p scale rev seed final).body

theorem state_parts (p : Parameters) (scale : ℝ) (rev seed final : Bool) :
    (stateKernel p scale rev seed final).surfaceBody=
      (stateParts p scale rev seed final).before ++
      [stateLoopStmt p scale rev seed final] ++
      (stateParts p scale rev seed final).after := by cases rev <;> rfl

theorem stateStep (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) (hn : n<p.NT) :
    ∃ t, run (engine (withDot M D)) (stateParts p scale rev seed final).body
      (stateIndex p rev n s)=some t ∧
      StateReady pid input t ∧ HasCarry p rev (stateNext M D p pid input scale rev n h) t ∧
      t.mem=writeHistory M p pid rev n h s.mem := by
  let start := stateStoreExecution M D p scale rev seed final pid input s n h ctx hh
  have hs := stateStore_ready M D p scale rev seed final pid input s n h ctx hh hn
  have hp := stateStore_pointers M D p scale rev seed final pid input s n h ctx hh hn
  let finish := stateUpdateExecution M D p scale rev seed final pid input start.val n h hs.1 hs.2.1 hs.2.2 hp
  have hf := stateUpdate_values M D p scale rev seed final pid input start.val n h hs.1 hs.2.1 hs.2.2 hp
  refine ⟨finish.val,?_,hf.1,hf.2.1,?_⟩
  · have split : (stateParts p scale rev seed final).body=
        stateStorePrefix p scale rev seed final ++ stateUpdateSuffix p scale rev seed final := by
      exact (List.take_append_drop _ _).symm
    rw [split,run_append]
    simpa only [start.property,Option.bind_some] using finish.property
  · exact hf.2.2.trans (stateStore_memory M D p scale rev seed final pid input s n h ctx hh hn)

theorem stateLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α)
    (ctx : StateReady pid input s) (hh : HasCarry p rev (stateInitial M p pid input rev seed) s) :
    ∃ t, step (engine (withDot M D)) (stateLoopStmt p scale rev seed final) s=some t ∧
      StateReady pid input t ∧ HasCarry p rev (stateValues M D p pid input scale rev seed p.NT) t ∧
      t.mem=foldRange (fun n mem => writeHistory M p pid rev n (stateValues M D p pid input scale rev seed n) mem)
        0 p.NT 1 s.mem := by
  let stepMemory := fun n mem => writeHistory M p pid rev n (stateValues M D p pid input scale rev seed n) mem
  let I := fun n (t : State α) => StateReady pid input t ∧
    HasCarry p rev (stateValues M D p pid input scale rev seed n) t ∧
    t.mem=foldRange stepMemory 0 n 1 s.mem
  have hs : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (stateParts p scale rev seed final).body
        (stateIndex p rev n t)=some u ∧ I (n+1) u := by
    intro n hn t hi
    obtain ⟨u,he,hc,hv,hm⟩ := stateStep M D p scale rev seed final pid input t n
      (stateValues M D p pid input scale rev seed n) hi.1 hi.2.1 hn
    refine ⟨u,he,hc,hv,?_⟩
    rw [hm,hi.2.2]
    exact (foldRange_succ stepMemory n s.mem).symm
  have hi : I 0 s := ⟨ctx,hh,by simp [foldRange_zero]⟩
  cases rev with
  | «false» =>
    obtain ⟨t,ran,done⟩ := range_invariant (engine (withDot M D)) "i_t"
      (stateParts p scale Bool.false seed final).body p.NT I
      (by intro n hn t h; simpa only [stateIndex,Bool.false_eq_true,if_false] using hs n hn t h)
      0 (Nat.zero_le _) s hi
    refine ⟨t,?_,done⟩
    simpa only [stateLoopStmt,Bool.false_eq_true,if_false,step] using ran
  | «true» =>
    have advance : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
        run (engine (withDot M D)) (stateParts p scale Bool.true seed final).body
          (t.setReg "__rev_i_t" .int [] (fun _ => (1-(p.NT : Int))+(n : Int)))=some u ∧ I (n+1) u := by
      intro n hn t h
      simpa [stateIndex,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using hs n hn t h
    obtain ⟨t,ran,done⟩ := SignedRange.clock (engine (withDot M D)) "__rev_i_t"
      (stateParts p scale Bool.true seed final).body (1-(p.NT : Int)) p.NT I advance 0 (Nat.zero_le _) s hi
    refine ⟨t,?_,done⟩
    have first : (stateIntParts p scale seed final).start =
        .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat p.NT)) (.constInt 1)) := rfl
    have last : (stateIntParts p scale seed final).stop = .constInt 1 := rfl
    have stride : (stateIntParts p scale seed final).stride = .constNat 1 := rfl
    simp only [stateLoopStmt,↓reduceIte,step,first,last,stride,evalOp_unfold,Option.bind,
      Option.bind_some,numeric,bop]
    have beginEq : (0 : Int)-((p.NT : Int)-1)=(1-(p.NT : Int))+(0 : Nat) := by omega
    have endEq : (1-(p.NT : Int))+(p.NT : Int)=1 := by omega
    change rangeInt (engine (withDot M D)) "__rev_i_t" (0-((p.NT : Int)-1)) 1 1
      (stateIntParts p scale seed final).body s=some t
    rw [beginEq]
    simpa only [endEq] using ran

theorem stateInitial_inputs (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (rev seed : Bool) (h : StateInputs a b) : stateInitial M p pid a rev seed=stateInitial M p pid b rev seed := by
  have hi := h "initial" (by simp)
  have hr := h "terminalGradient" (by simp)
  funext i
  cases rev <;> cases seed <;> simp [stateInitial,seedName,load,rawLoad,hi,hr]

/-- Entire forward/reverse state source, including both optional endpoints.
The source executes against current memory. The snapshot is justified only by
preservation of the explicitly listed read regions. -/
theorem stateProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (input : Memory α) (s : State α) (hin : StateInputs s.mem input) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (stateKernel p scale rev seed final) s=some t ∧
      t.mem=stateMemory M D p ⟨s.pids 0,s.pids 1,s.pids 2⟩ input s.mem scale rev seed final ∧
      StateInputs t.mem input := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := stateEntry M D p scale rev seed final s
  have he := stateEntry_ready M D p scale rev seed final s
  have hc : StateReady pid input start.val := ⟨he.1.inputs.trans hin,he.1.ik,he.1.iv,he.1.bh⟩
  have hh : HasCarry p rev (stateInitial M p pid input rev seed) start.val := by
    rw [← stateInitial_inputs M p pid s.mem input rev seed hin]
    exact he.2
  obtain ⟨mid,hm,hc2,hh2,hmem⟩ := stateLoop M D p scale rev seed final pid input start.val hc hh
  let finish := stateFinish M D p scale rev seed final pid input mid
    (stateValues M D p pid input scale rev seed p.NT) hc2 hh2
  refine ⟨finish.val,?_,?_,?_⟩
  · have hl := hm
    change run (engine (withDot M D)) (stateKernel p scale rev seed final).surfaceBody s=some finish.val
    rw [state_parts,List.append_assoc,run_append]
    simp only [start.property,Option.bind_some,List.singleton_append,run,hl,bind,finish.property]
  · change (stateFinish M D p scale rev seed final pid input mid _ hc2 hh2).val.mem=_
    rw [stateFinish_memory,hmem]
    dsimp only [start]
    rw [stateEntry_memory]
    rfl
  · change StateInputs (stateFinish M D p scale rev seed final pid input mid _ hc2 hh2).val.mem input
    rw [stateFinish_memory]
    exact (final_inputs _ _ _ _ _ _ _).trans hc2.inputs

#axiomsClean stateStep
#axiomsClean stateLoop
#axiomsClean stateProgram
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
