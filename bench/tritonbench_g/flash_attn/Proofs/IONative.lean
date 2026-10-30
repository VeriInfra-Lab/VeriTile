import bench.tritonbench_g.flash_attn.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

/-- Complete native source observation for every raw layout and scalar ABI.
Output collisions use the actual source's last writer, followed by the original
replica/owner readback. All exterior cells retain their complete typed payload. -/
theorem native_implements (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI) :
    (nativeIO I x scale).Implements (Logical.nativeFormula I x scale) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    exact schedule_safe KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
      (Logical.nativeObservation I x) xs bounds hb _ (fun j hj => (program_valid x j).mp hj) s
  · intro xs s hin
    obtain ⟨final,he,hm⟩ := schedule_memory KernelNative.algebra KernelNative.dot x (KernelNative.scalar scale)
      (programs (parameters x)) s
    refine ⟨final,he,?_,?_⟩
    · intro port i _
      fin_cases port
      · change mapCell _ (final.mem "O" (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val))=_
        rw [hm,grid_output_tensor KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
          (Logical.nativeObservation I x) xs s hin i]
        rfl
      · change mapCell _ (final.mem "L" (FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val))=_
        rw [hm,grid_log_tensor KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
          (Logical.nativeObservation I x) xs s hin i]
        rfl
    · intro r off outside
      rw [hm]
      apply grid_frame
      exact (outside_iff KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
        (Logical.nativeObservation I x) xs r off).mp outside

#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
