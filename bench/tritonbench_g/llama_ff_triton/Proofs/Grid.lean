import bench.tritonbench_g.llama_ff_triton.Proofs.Complete
import bench.tritonbench_g.llama_ff_triton.Proofs.Preservation
import bench.tritonbench_g.llama_ff_triton.Proofs.Geometry

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Grid
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (atProgram Outside)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

def target {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial : State α) (x : Index p) : α :=
  if native then logicalNative a g p initial x.1.val x.2.val
  else logicalSource a p initial x.1.val x.2.val
def one {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (pid : Nat) (s : State α) :=
  if native then nativeExecution a g p (atProgram s pid)
  else FP.Structural.exec a.model (source p) (atProgram s pid)
def grid {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (s : State α) :=
  GridExecution.sequence (one native a g p) (List.range (count p)) s

theorem single {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial current : State α)
    (bytes : ByteValues initial) (hout : Outside "OUT" initial current) (pid : Nat) :
    ∃ final, one native a g p pid current=some final ∧
      GridExecution.Post "OUT" (address p) (fun x => Cell.mk .fp16 (target native a g p initial x))
        (Owns p pid) current final := by
  let s := atProgram current pid
  have hb : ByteValues s := byteValues_preserved initial s hout bytes
  have per : ∃ final, one native a g p pid current=some final ∧
      Execution.Post p s final (fun i => if native then nativeOutput a g p s i else sourceOutput a p s i) := by
    cases native
    · simpa only [one,Bool.false_eq_true,ite_false] using sourceComplete a p s hb
    · simpa only [one,ite_true] using nativeComplete a g p s hb
  obtain ⟨final,he,hread,hframe⟩ := per
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := by
      simpa only [active,s,row_at,column_at,hr,hc] using And.intro x.1.isLt x.2.isLt
    have ho : outputOffset p s i=address p x := by
      simp only [outputOffset,address,s,row_at,column_at,hr,hc]
    have hir : inputRow p s i.1=x.1.val := by
      change row p pid i.1%p.rows=x.1.val
      rw [hr,Nat.mod_eq_of_lt x.1.isLt]
    have hic : inputColumn p s i.2.1=x.2.val := by
      change column p pid i.2.1%p.columns=x.2.val
      rw [hc,Nat.mod_eq_of_lt x.2.isLt]
    rw [←ho,hread i hi]
    simp only [sourceOutput_logical,nativeOutput_logical,hir,hic,target]
    rw [(logical_preserved a g p initial s hout x.1.val x.2.val).1,
      (logical_preserved a g p initial s hout x.1.val x.2.val).2]
  · intro r o outside
    apply hframe r o
    rcases outside with h|h
    · exact Or.inl h
    · right
      intro i hi
      obtain ⟨x,hx,ho⟩ := active_index p current pid i hi
      rw [ho]
      exact h x hx

theorem whole {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial : State α) (bytes : ByteValues initial) :
    ∃ final, grid native a g p initial=some final ∧
      GridExecution.Post "OUT" (address p) (fun x => Cell.mk .fp16 (target native a g p initial x))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose "OUT" (address p) (address_injective p)
    (fun x => Cell.mk .fp16 (target native a g p initial x)) (Owns p)
    (one native a g p) initial (count p)
    (fun pid _ current hout => single native a g p initial current bytes hout pid)
    (List.range (count p)) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := covers p x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

theorem empty_grid {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) (initial : State α) (h : p.rows=0) :
    grid native a g p initial=some initial := by
  simp [grid,empty p h,GridExecution.sequence]

#axiomsClean single
#axiomsClean whole
#axiomsClean empty_grid
#specNonCircular target avoiding [grid,one,source,run,nativeExecution,sourceOutput,nativeOutput]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Grid
