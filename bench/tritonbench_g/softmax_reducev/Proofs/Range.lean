import VeriTile.Triton.Float.RangeInduction
import bench.tritonbench_g.softmax_reducev.Proofs.FPSource
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range
open VeriTile.Triton FP.Structural
noncomputable section

/-- Pure value recurrence over the exact source index sequence, independent
of registers, statements, or execution success. -/
def foldRange {β : Type} (update : Nat → β → β) (cur stop stride : Nat) (initial : β) : β :=
  if hzero : stride = 0 then initial
  else if hlt : cur  <  stop then foldRange update (cur + stride) stop stride (update cur initial)
  else initial
termination_by stop - cur
decreasing_by omega



/-- A numerical recurrence may be carried through a source loop without any
associativity or reduction law. The premise proves one actual body execution;
the conclusion follows the original dynamic range indices exactly. -/
theorem range_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n  <  stop →
      ∃ u, run M body (t.setReg idx .nat [] (fun _ => n)) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, FP.Structural.range M idx cur stop stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  exact FP.Structural.range_fold M idx body update I Done cur stop stride initial s
    (fun n a => foldRange update n stop stride a)
    (fun n a => by simpa only [dite_eq_ite] using foldRange.eq_def update n stop stride a)
    hstep hdone h

/-- A loop starting at a natural index uses the signed source bound
exactly, including an empty interval for every negative bound. -/
theorem range_int_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur : Nat) (stop : Int) (stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t,I n a t → n<stop.toNat →
      ∃ u,run M body (t.setReg idx .int [] (fun _=>(n : Int)))=some u ∧ I (n+stride) (update n a) u)
    (hdone : ∀ n a t,I n a t → Done a t) (h : I cur initial s) :
    ∃ final,FP.Structural.rangeInt M idx cur stop stride body s=some final ∧
      Done (foldRange update cur stop.toNat stride initial) final := by
  exact FP.Structural.rangeInt_fold M idx body update I Done cur stop stride initial s
    (fun n a => foldRange update n stop.toNat stride a)
    (fun n a => by simpa only [dite_eq_ite] using foldRange.eq_def update n stop.toNat stride a)
    hstep hdone h

#axiomsClean range_int_values
#axiomsClean range_values
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range
