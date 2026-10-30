import bench.repairs.tritonbench_g.softmax_triton2.IO
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.Pair
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.PairSafety
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
open scoped VeriTile.Spec
noncomputable section

/-- Connect the existing row numerical theorem to every original IO obligation.
No safety, execution or whole-program equality is supplied as an assumption. -/
theorem io_equivalent (R : FP.ScalarArithmetic.Rules)
    (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat)
    (hB : 0 < B) (countBound : B < 2^32) :
    IOEquivalent R.assumptions
      ⟨softmax_kernel out inp ib ob ir ic orow oc n B,⟨out,inp,ib,ob,ir,ic,orow,oc,n,B,p⟩⟩
      ⟨guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B,⟨out,inp,ib,ob,ir,ic,orow,oc,n,B,p⟩⟩ := by
  intro α _ M D model ports vals _ bounds bounded initial _
  let left : IOProgram := ⟨softmax_kernel out inp ib ob ir ic orow oc n B,
    ⟨out,inp,ib,ob,ir,ic,orow,oc,n,B,p⟩⟩
  let right : IOProgram := ⟨guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B,
    ⟨out,inp,ib,ob,ir,ic,orow,oc,n,B,p⟩⟩
  let start := (invocation M left).initial initial
  have sameStart : (invocation M right).initial initial = start := rfl
  have readBounds (j : Fin B) (hj : j.val < n) :
      ib+start.pids 0*ir+j.val*ic < bounds inp := bounded.2.1 0 (j,PUnit.unit) hj
  have writeBounds (j : Fin B) (hj : j.val < n) :
      ob+start.pids 0*orow+j.val*oc < bounds out := bounded.2.2.1 0 (j,PUnit.unit) hj
  let a := originalExecution M out inp ib ob ir ic orow oc n B hB start
  let b := candidateExecution M D ports out inp ib ob ir ic orow oc n B hB start
  have same := row_memory R M D model ports out inp ib ob ir ic orow oc n B hB countBound start
  have frame : (sourceIO M left).Frame vals initial a.val := by
    intro r offset outside
    apply original_frame M out inp ib ob ir ic orow oc n B hB start r offset
    intro j hj hit
    have miss := outside.1 0 (j,PUnit.unit) hj
    exact miss.elim (fun h => h hit.1.symm) (fun h => h hit.2.symm)
  refine ⟨?_,?_,a.val,b.val,?_,?_,?_,frame,?_⟩
  · apply (scheduleSafe_singleton ..).mpr
    exact original_safe M out inp ib ob ir ic orow oc n B hB bounds start readBounds writeBounds
  · apply (scheduleSafe_singleton ..).mpr
    exact candidate_safe M D ports out inp ib ob ir ic orow oc n B hB bounds start readBounds writeBounds
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
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
