import bench.tritonbench_g.softmax_triton3.SourceIO
import bench.tritonbench_g.softmax_triton3.Proofs.Pair
import bench.tritonbench_g.softmax_triton3.Proofs.PairSafety
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
open VeriTile.Triton FP.Structural FP.Guarded
open VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open scoped VeriTile.Spec
noncomputable section
set_option linter.unusedSimpArgs false

/-- Full existing kernel IO scope: both actual traces are safe, both runs
succeed, active complete typed output cells agree and both exterior frames hold.
The independent logical input binding is retained even though the memory proof
is stronger and applies to every initial scalar payload. -/
def IOEquivalent (R : Spec.Assumptions FP.GuardedFragment) (lhs rhs : IOProgram) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    Ports M D → (sourceIO M lhs).Equivalent (sourceVariant M lhs rhs)

instance : Spec.ProgramSyntax IOProgram where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × Layout
  signature p := (match p.kernel with | .mk inputs outputs _ => (inputs,outputs),p.layout)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := IOEquivalent
  sameContext := Eq

/-- Connect the existing row numerical theorem to every original IO obligation.
No safety, execution or whole-program equality is supplied as an assumption. -/
theorem io_equivalent (R : FP.ScalarArithmetic.Rules)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) (p : Nat)
    (hB : 0 < B) (countBound : B < 2^32) :
    IOEquivalent R.assumptions
      ⟨softmax_kernel out inp mask stride n B hasMask,⟨out,inp,mask,stride,n,B,hasMask,p⟩⟩
      ⟨guarded_reciprocal_kernel out inp mask stride n B hasMask,⟨out,inp,mask,stride,n,B,hasMask,p⟩⟩ := by
  intro α _ M D model ports vals _ bounds bounded initial _
  let left : IOProgram := ⟨softmax_kernel out inp mask stride n B hasMask,
    ⟨out,inp,mask,stride,n,B,hasMask,p⟩⟩
  let right : IOProgram := ⟨guarded_reciprocal_kernel out inp mask stride n B hasMask,
    ⟨out,inp,mask,stride,n,B,hasMask,p⟩⟩
  let start := (invocation M left).initial initial
  have sameStart : (invocation M right).initial initial = start := rfl
  have readBounds (j : Fin B) (hj : j.val < n) :
      start.pids 0*stride+j.val < bounds inp := by
    simpa [sourceIO,start,invocation,left,Invocation.initial,bindRegisters] using
      bounded.2.1 ⟨0,by simp [sourceIO]⟩ (j,PUnit.unit) (by simpa [sourceIO] using hj)
  have maskBounds (hm : hasMask = Bool.true) (j : Fin B) (hj : j.val < n) :
      start.pids 0*stride+j.val < bounds mask := by
    simpa [sourceIO,start,invocation,left,Invocation.initial,bindRegisters] using
      bounded.2.1 ⟨1,by simp [sourceIO]⟩ (j,PUnit.unit) (by simpa [sourceIO] using And.intro hj hm)
  have writeBounds (j : Fin B) (hj : j.val < n) :
      start.pids 0*stride+j.val < bounds out := bounded.2.2.1 0 (j,PUnit.unit) hj
  let a := originalExecution M out inp mask stride n B hasMask hB start
  let b := candidateExecution M D ports out inp mask stride n B hasMask hB start
  have same := row_memory R M D model ports out inp mask stride n B hasMask hB countBound start
  have frame : (sourceIO M left).Frame vals initial a.val := by
    intro r offset outside
    apply original_frame M out inp mask stride n B hasMask hB start r offset
    intro j hj hit
    have miss := outside.1 0 (j,PUnit.unit) hj
    exact miss.elim (fun h => h hit.1.symm) (fun h => h hit.2.symm)
  refine ⟨?_,?_,a.val,b.val,?_,?_,?_,frame,?_⟩
  · apply (scheduleSafe_singleton ..).mpr
    exact original_safe M out inp mask stride n B hasMask hB bounds start readBounds maskBounds writeBounds
  · apply (scheduleSafe_singleton ..).mpr
    exact candidate_safe M D ports out inp mask stride n B hasMask hB bounds start readBounds maskBounds writeBounds
  · exact (runSchedule_singleton ..).trans a.property
  · exact (runSchedule_singleton ..).trans b.property
  · intro o j active
    change mapCell id (a.val.mem _ _) = mapCell id (b.val.mem _ _)
    rw [same]
  · intro r offset outside
    rw [←same]
    exact frame r offset outside

#axiomsClean io_equivalent
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
