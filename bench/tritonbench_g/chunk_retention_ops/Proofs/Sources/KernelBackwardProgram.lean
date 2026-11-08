import bench.tritonbench_g.chunk_retention_ops.Proofs.Sources.KernelBackwardFinish
import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRange
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (range_invariant foldRange_succ foldRange_zero)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (scale : ℝ) (terminal : Bool) : Nat → Carry p α
  | 0 => initialValue M p pid input terminal
  | n+1 => next M D p pid input scale n (values M D p pid input scale terminal n)

def memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input current : Memory α) (scale : ℝ) (terminal initial : Bool) : Memory α :=
  writeInitial M p pid initial (values M D p pid input scale terminal p.NT)
    (foldRange (fun n mem=>writeHistory M p pid n
      (values M D p pid input scale terminal n) mem) 0 p.NT 1 current)

theorem InputEq.refl (m : Memory α) : InputEq m m := by intro r hr o; rfl
theorem InputEq.trans {a b c : Memory α} (ab : InputEq a b) (bc : InputEq b c) : InputEq a c := by
  intro r hr o; exact (ab r hr o).trans (bc r hr o)

def loopStmt (p : Parameters) (scale : ℝ) (terminal initial : Bool) : ComputeStmt :=
  let actual:=intParts p scale terminal initial
  .forRangeInt "__rev_i_t" actual.start actual.stop actual.stride actual.body

theorem source_parts (p : Parameters) (scale : ℝ) (terminal initial : Bool) :
    (kernel p scale terminal initial).surfaceBody=(parts p scale terminal initial).before ++
      [loopStmt p scale terminal initial] ++ (parts p scale terminal initial).after := by rfl

theorem bodyStep (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α) (n : Nat)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) (hn : n<p.NT) :
    ∃ t, run (engine (withDot M D)) (parts p scale terminal initial).body
      (indexState p n s)=some t ∧
      Ready M p pid input t ∧ HasCarry p (next M D p pid input scale n carry) t ∧
      t.mem=writeHistory M p pid n carry s.mem := by
  let start:=storeExecution M D p scale terminal initial pid input s n carry ctx hc
  have hs:=store_ready M D p scale terminal initial pid input s n carry ctx hc hn
  have hp:=store_pointers M D p scale terminal initial pid input s n carry ctx hc hn
  let finish:=updateExecution M D p scale terminal initial pid input start.val n carry hs.1 hs.2.1 hs.2.2 hp
  have hf:=update_values M D p scale terminal initial pid input start.val n carry hs.1 hs.2.1 hs.2.2 hp
  refine ⟨finish.val,?_,hf.1,hf.2.1,?_⟩
  · have split : (parts p scale terminal initial).body=
        storePrefix p scale terminal initial ++ updateSuffix p scale terminal initial :=
      (List.take_append_drop _ _).symm
    rw [split,run_append]
    simpa only [start.property,Option.bind_some] using finish.property
  · exact hf.2.2.trans (store_memory M D p scale terminal initial pid input s n carry ctx hc hn)

theorem loop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α)
    (ctx : Ready M p pid input s) (hc : HasCarry p (initialValue M p pid input terminal) s) :
    ∃ t, step (engine (withDot M D)) (loopStmt p scale terminal initial) s=some t ∧
      Ready M p pid input t ∧ HasCarry p (values M D p pid input scale terminal p.NT) t ∧
      t.mem=foldRange (fun n mem=>writeHistory M p pid n (values M D p pid input scale terminal n) mem)
        0 p.NT 1 s.mem := by
  let stepMemory:=fun n mem=>writeHistory M p pid n (values M D p pid input scale terminal n) mem
  let I:=fun n (t : State α)=>Ready M p pid input t ∧
    HasCarry p (values M D p pid input scale terminal n) t ∧
    t.mem=foldRange stepMemory 0 n 1 s.mem
  have hs : ∀ n,n<p.NT → ∀ t,I n t → ∃ u,
      run (engine (withDot M D)) (parts p scale terminal initial).body
        (indexState p n t)=some u ∧ I (n+1) u := by
    intro n hn t hi
    obtain ⟨u,he,hr,hv,hm⟩:=bodyStep M D p scale terminal initial pid input t n
      (values M D p pid input scale terminal n) hi.1 hi.2.1 hn
    refine ⟨u,he,hr,hv,?_⟩
    rw [hm,hi.2.2]
    exact (foldRange_succ stepMemory n s.mem).symm
  have hi : I 0 s:=⟨ctx,hc,by simp [foldRange_zero]⟩
  have advance : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (parts p scale terminal initial).body
        (t.setReg "__rev_i_t" .int [] (fun _ => (1-(p.NT : Int))+(n : Int)))=some u ∧ I (n+1) u := by
    intro n hn t ht
    simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using hs n hn t ht
  obtain ⟨t,ran,done⟩ := VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange.clock
    (engine (withDot M D)) "__rev_i_t" (parts p scale terminal initial).body
    (1-(p.NT : Int)) p.NT I advance 0 (Nat.zero_le _) s hi
  refine ⟨t,?_,done⟩
  have first : (intParts p scale terminal initial).start =
      .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat p.NT)) (.constInt 1)) := rfl
  have last : (intParts p scale terminal initial).stop = .constInt 1 := rfl
  have stride : (intParts p scale terminal initial).stride = .constNat 1 := rfl
  simp only [loopStmt,step,first,last,stride,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
  change rangeInt (engine (withDot M D)) "__rev_i_t" (0-((p.NT : Int)-1)) 1 1
    (intParts p scale terminal initial).body s=some t
  have beginEq : (0 : Int)-((p.NT : Int)-1)=(1-(p.NT : Int))+(0 : Nat) := by omega
  have endEq : (1-(p.NT : Int))+(p.NT : Int)=1 := by omega
  rw [beginEq]
  simpa only [endEq] using ran

theorem initialValue_inputs (M : Algebra α) (p : Parameters) (pid : PID) (a b : Memory α)
    (terminal : Bool) (h : InputEq a b) : initialValue M p pid a terminal=initialValue M p pid b terminal := by
  have hi:=h "terminal" (by simp)
  funext i
  cases terminal <;> simp [initialValue,VeriTile.Bench.Repairs.ChunkLinearAttn.load,
    VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,hi]

/-- The complete selected backward-state quotation succeeds and has the independently
specified reverse recurrence and memory effects. No execution-success premise is used. -/
theorem program (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (input : Memory α) (s : State α) (hin : InputEq s.mem input)
    (hBT : p.BT≤2^31) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel p scale terminal initial) s=some t ∧
      t.mem=memory M D p ⟨s.pids 0,s.pids 1,s.pids 2⟩ input s.mem scale terminal initial ∧
      InputEq t.mem input := by
  let pid : PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start:=entry M D p scale terminal initial s
  have he:=entry_ready M D p scale terminal initial s hBT
  have hc : Ready M p pid input start.val:=
    ⟨he.1.inputs.trans hin,he.1.ik,he.1.iv,he.1.bh,he.1.rate,he.1.lane,he.1.weights⟩
  have hh : HasCarry p (initialValue M p pid input terminal) start.val := by
    rw [← initialValue_inputs M p pid s.mem input terminal hin]
    exact he.2
  obtain ⟨mid,hm,hc2,hh2,hmem⟩:=loop M D p scale terminal initial pid input start.val hc hh
  let finish:=finishExecution M D p scale terminal initial pid input mid
    (values M D p pid input scale terminal p.NT) hc2 hh2
  refine ⟨finish.val,?_,?_,?_⟩
  · have hl:=hm
    change run (engine (withDot M D)) (kernel p scale terminal initial).surfaceBody s=some finish.val
    rw [source_parts,List.append_assoc,run_append]
    simp only [start.property,Option.bind_some,List.singleton_append,run,hl,bind,finish.property]
  · change (finishExecution M D p scale terminal initial pid input mid _ hc2 hh2).val.mem=_
    rw [finish_memory,hmem]
    dsimp only [start]
    rw [entry_memory]
    rfl
  · change InputEq (finishExecution M D p scale terminal initial pid input mid _ hc2 hh2).val.mem input
    rw [finish_memory]
    exact (initial_inputs _ _ _ _ _ _).trans hc2.inputs

#axiomsClean bodyStep
#axiomsClean loop
#axiomsClean program
#specNonCircular values avoiding [run,FP.Structural.exec,entry,storeExecution,updateExecution,finishExecution]
#specNonCircular memory avoiding [run,FP.Structural.exec,entry,storeExecution,updateExecution,finishExecution]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
