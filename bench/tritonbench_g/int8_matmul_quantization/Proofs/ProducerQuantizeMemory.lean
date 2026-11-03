import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerQuantize
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.QuantizePass
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
abbrev Memory (α : Type) := RegionName→Nat→Cell α

def quotient {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (original : State α)
    (scale : α) (k : Nat) (memory : Memory α) (i : Index p) : α :=
  M.binary (some .fp32) .real .div
    (M.fp32Load (if i.2.1.val<p.columns-k*p.width then
      (memory p.input (inputOffset p original k i)).read .real else M.literal (some .fp32) .real 0)) scale

/-- Every operand is loaded from the pre-store block snapshot. The lane fold
writes exactly the source store order and does not reload modified lanes. -/
def writeBlock {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original : State α) (scale : α) (k : Nat) (memory : Memory α) : Memory α :=
  (TileShape.allIndices [1,p.width]).foldl (fun current i =>
    if i.2.1.val<p.columns-k*p.width then
      fun region offset => if region=p.output ∧ offset=outputOffset p original k i then
        .mk .int (integer8 (quotient M p original scale k memory i)).toInt else current region offset
    else current) memory

def writeIterations {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original : State α) (scale : α) : Nat→Nat→Memory α→Memory α
  | _,0,memory => memory
  | k,remaining+1,memory => writeIterations M integer8 p original scale (k+1) remaining
      (writeBlock M integer8 p original scale k memory)
private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem fold_expanded {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then
      { t with mem := fun r' o => if r' = r ∧ o = off i then value i else t.mem r' o }
      else t) s).mem =
    l.foldl (fun mem i => fun r' o => if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem :=
  fold_memory l r off value active s

theorem iteration_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (k : Nat)
    (h : Invariant p original scale k current) :
    (iteration M integer8 p original current scale k h).val.mem =
      writeBlock M integer8 p original scale k current.mem := by
  simp +contextual (config := {maxSteps:=1000000}) [iteration,writeBlock,quotient,
    fold_memory,fold_expanded,State.write,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    h.input,h.output,h.offsk,h.scale,ofFloat,toFloat,show (0.0:ℝ)=0 by norm_num]
  apply congrArg (fun f => (TileShape.allIndices [1,p.width]).foldl f current.mem)
  funext memory i region offset
  rcases i with ⟨row,column,⟨⟩⟩
  by_cases active : column.val<p.columns-k*p.width
  all_goals simp [active]

theorem loop_memory {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (start stop : Nat)
    (ordered : start≤stop) (h : Invariant p original scale start current) (final : State α)
    (ran : rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (quantizeBody p) current=some final) :
    final.mem=writeIterations M integer8 p original scale start (stop-start) current.mem := by
  have aux : ∀ distance start (current : State α), stop-start=distance → start≤stop →
      Invariant p original scale start current → ∀ final,
      rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (quantizeBody p) current=some final →
      final.mem=writeIterations M integer8 p original scale start (stop-start) current.mem := by
    intro distance
    induction distance using Nat.strong_induction_on with
    | h distance ih =>
      intro start current count ordered inv final executed
      by_cases active : start<stop
      · let next:=iteration M integer8 p original current scale start inv
        have nextInv:=iteration_invariant M integer8 p original current scale start inv
        rw [rangeInt,if_neg (by decide),if_pos (by exact_mod_cast active),next.property] at executed
        change rangeInt (backend M integer8) "k" ((start+1 : Nat) : Int) (stop : Int) 1 (quantizeBody p) next.val=some final at executed
        have tail:=ih (stop-(start+1)) (by omega) (start+1) next.val rfl (by omega) nextInv final executed
        have remaining : stop-start=(stop-(start+1))+1 := by omega
        rw [remaining,writeIterations]
        rw [iteration_memory] at tail
        exact tail
      · have finished : start=stop := by omega
        subst start
        rw [rangeInt,if_neg (by decide),if_neg (by omega)] at executed
        cases Option.some.inj executed
        simp [writeIterations]
  exact aux _ _ _ rfl ordered h final ran

#axiomsClean iteration_memory
#axiomsClean loop_memory

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.QuantizePass
