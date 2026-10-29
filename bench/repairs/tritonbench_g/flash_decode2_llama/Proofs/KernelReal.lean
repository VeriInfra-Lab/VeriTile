import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelReal
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

def realRead (read : Nat → ℝ) (c : Call) (r : RegionName) (i : Nat) : ℝ :=
  read ((KernelTensor.descriptor c r).metadata.dataPtr.toNat+4*i)

def entry (interpret : Nat → ℝ) (c : Call) (heap : KernelTensor.Heap)
    (other : State (Option ℝ)) :=
  KernelTensor.lift (fun r i _ => some (realRead interpret c r i)) c heap other

def target (interpret : Nat → ℝ) (c : Call) (heap : KernelTensor.Heap) (row d : Nat) : ℝ :=
  KernelModel.realTarget c (KernelTensor.lengths c heap)
    (realRead interpret c "E")
    (realRead interpret c "M") row d

/-- Arbitrary real input observations at the actual Tensor addresses.
Ideal-real execution of the complete online recurrence equals the
unshifted weighted-summary formula. This is separate from native rounding. -/
theorem value (interpret : Nat → ℝ) (c : Call) (heap : KernelTensor.Heap)
    (other : State (Option ℝ)) (h : InputContract c (KernelTensor.lengths c heap))
    (row : Nat) (hr : row<KernelModel.rows c) (k : TileIndex [Checked.block c.val]) :
    Grid.value Reference.ideal (entry interpret c heap other) c.val.masked
      (Checked.batches c.val) (Checked.heads c.val) (Checked.capacity c.val)
      (Checked.width c.val) c.val.blockSeq (Checked.block c.val) row k =
    some (target interpret c heap row (lane k)) := by
  let before := entry interpret c heap other
  let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row before
  let x := fun j => realRead interpret c "E" (row*Checked.capacity c.val+j)
  let v := fun j (k : TileIndex [Checked.block c.val]) =>
    realRead interpret c "M"
      (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+lane k)
  have counted := KernelModel.count_bound c _ h before (KernelTensor.lift_length _ c heap other) row hr
  have hx : ∀ j,j<count s c.val.blockSeq →
      logvalue Reference.ideal s (Checked.heads c.val) (Checked.capacity c.val) j=some (x j) := by
    intro j _
    simp only [s,logvalue,row_log_offset]
    simp [Reference.ideal,before,entry,KernelTensor.lift,Grid.setup,Cell.read_same,x]
  have hv : ∀ j,j<count s c.val.blockSeq → ∀ k,
      FlashDecode2.value Reference.ideal s c.val.masked (Checked.heads c.val)
        (Checked.capacity c.val) (Checked.width c.val) j k=some (v j k) := by
    intro j _ k
    simp only [s,FlashDecode2.value,row_mid_offset]
    simp [active,c.property,Reference.ideal,
      before,entry,KernelTensor.lift,Grid.setup,Cell.read_same,v]
  have result := Reference.source_correct s c.val.masked (Checked.heads c.val)
    (Checked.capacity c.val) (Checked.width c.val) c.val.blockSeq (Checked.block c.val) x v
    (counted.1.symm ▸ counted.2.1) hx hv k
  change output Reference.ideal s _ _ _ _ _ _ k = _
  rw [result]
  change some (Reference.target x v (count s c.val.blockSeq) k)=_
  rw [show count s c.val.blockSeq=KernelModel.iterations c (KernelTensor.lengths c heap) row from counted.1]
  rfl

#axiomsClean value
#specNonCircular target avoiding [KernelModel.run,Grid.source,FP.Structural.exec,values,Reference.recurrence]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelReal
