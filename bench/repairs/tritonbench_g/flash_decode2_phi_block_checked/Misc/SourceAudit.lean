import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Misc.RawResults
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelIdealReadback

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Audit
open VeriTile.Triton FP.Structural FlashDecode2 KernelTensor KernelModel
noncomputable section

/-- Current Algebra field layout is checked definitionally against the refreshed
source cache; no cached proof with a stale load interpretation is accepted. -/
example (x : Native.Term) : Native.algebra.fp32Load x=Native.Term.load x := rfl
example (x y : Native.Term) : Native.algebra.binary (some .fp32) .real .mul x y=
    Native.Term.binary (some .fp32) .real .mul x y := rfl
example (x : Option ℝ) : Reference.ideal.fp32Load x=x := rfl

theorem native_empty (p : KernelNative.Ops) (S : Native.Schedule) (c : Call)
    (heap : Heap) (other : State Native.Term) (h : InputContract c (lengths c heap))
    (empty : Checked.batches c.val=0 ∨ Checked.heads c.val=0)
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    KernelReadback.execute p S c heap other h.metadata order=heap := by
  have zero : size c=0 := by
    rcases empty with hb|hh
    · simp [size,rows,hb]
    · simp [size,rows,hh]
  funext a
  apply KernelReadback.exterior p S c heap other h.metadata order all
  simp only [zero,Nat.mul_zero,Nat.add_zero]
  omega

#axiomsClean VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.real
#axiomsClean VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.native
#axiomsClean KernelModel.empty
#axiomsClean native_empty
#axiomsClean KernelSafety.offsets
#axiomsClean KernelSafety.formed
#axiomsClean KernelSafety.pointers
#axiomsClean KernelSafety.length_add
#axiomsClean arange_bounds
#axiomsClean empty_domain_unchanged
#axiomsClean check_arguments_iff
#axiomsClean KernelTensor.mid_read
#axiomsClean KernelTensor.log_read
#specNonCircular KernelReal.target avoiding [KernelModel.run,Grid.source,FP.Structural.exec,values,Reference.recurrence]
#specNonCircular KernelNative.target avoiding [KernelModel.run,Grid.source,FP.Structural.exec,values]
#specNonCircular InputContract avoiding [KernelModel.run,Grid.source,FP.Structural.exec,Host.Preparation,Host.invoke]
#specNonCircular KernelTensor.Arguments avoiding [KernelModel.run,Grid.source,FP.Structural.exec,Host.Preparation,Host.invoke]
#specNonCircular KernelReadback.execute avoiding [KernelNative.target,KernelEndpoints.native]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Audit
