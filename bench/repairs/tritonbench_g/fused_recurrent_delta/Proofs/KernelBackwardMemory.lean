import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardValues
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelStore
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardMemory
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def keyStore (M : Algebra α) (s : Shape) (r : RegionName) (bh t : Nat)
    (values : Fin s.BK → α) (u : State α) : State α :=
  (TileShape.allIndices [s.BK]).foldl (fun v i =>
    if decide (i.1.val<s.K) then
      v.write r (keyAddress s bh t i.1.val) (KernelReference.stored M (values i.1)) else v) u

def betaStore (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (values : Fin s.BBV → α) (u : State α) : State α :=
  if f.vectorBeta then KernelMemory.rowStore M s "dbeta" (plane s bh iv) iv s.BBV t values u
  else u.write "dbeta" (plane s bh iv*s.T+t)
    (KernelReference.stored M (KernelBackwardReference.betaSum M s values))

theorem key_cell (M : Algebra α) (s : Shape) (r : RegionName) (bh t : Nat)
    (values : Fin s.BK → α) (u : State α) (i : Fin s.BK) (hi : i.val<s.K) :
    (keyStore M s r bh t values u).mem r (keyAddress s bh t i.val)=
      KernelReference.stored M (values i) := by
  apply masked_readback r (fun i : TileIndex [s.BK] => keyAddress s bh t i.1.val)
    (fun i => KernelReference.stored M (values i.1)) (fun i => decide (i.1.val<s.K))
    (TileShape.allIndices [s.BK]) u (i,PUnit.unit)
    (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ _) (by simpa)
  rintro ⟨k,⟨⟩⟩ _ same
  have equal : i=k := Fin.ext (by simpa [keyAddress] using same)
  simp [equal]

theorem key_frame (M : Algebra α) (s : Shape) (r : RegionName) (bh t : Nat)
    (values : Fin s.BK → α) (u : State α) (r' : RegionName) (off : Nat)
    (outside : r'≠r ∨ ∀ i : Fin s.BK,i.val<s.K → off≠keyAddress s bh t i.val) :
    (keyStore M s r bh t values u).mem r' off=u.mem r' off := by
  apply State.masked_scatter_frame
  rcases outside with different | miss
  · exact Or.inl different
  · right
    intro i _ enabled
    exact miss i.1 (by simpa using enabled)

theorem beta_frame (M : Algebra α) (s : Shape) (f : Flags) (bh iv t : Nat)
    (values : Fin s.BBV → α) (u : State α) (r : RegionName) (off : Nat) (different : r≠"dbeta") :
    (betaStore M s f bh iv t values u).mem r off=u.mem r off := by
  unfold betaStore
  split
  · exact KernelMemory.rowStore_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl different)
  · simp [State.write,different]

theorem write_mem (r : RegionName) (off : Nat) (value : Cell α) (u v : State α) (same : u.mem=v.mem) :
    (u.write r off value).mem=(v.write r off value).mem := by
  simp only [State.write,same]

theorem memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : Reverse s f bh iv t h u) :
    (KernelBackwardStep.reverse M s f scale bh iv t h u regs).val.mem=
      (betaStore M s f bh iv t (KernelBackwardValues.dbeta M s f scale bh iv t h u)
        (KernelMemory.rowStore M s "dv" bh iv s.BBV t (KernelBackwardValues.dv M s f scale bh iv t h u)
          (keyStore M s "dk" (plane s bh iv) t (KernelBackwardValues.dk M s f scale bh iv t h u) u))).mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases regs
  all_goals
    simp only [KernelBackwardStep.reverse,State.setReg_mem]
    simp only [KernelBackwardValues.dbeta,KernelBackwardValues.dv,KernelBackwardValues.dk,
      KernelBackwardValues.enhanced,KernelBackwardReference.enhance,KernelBackwardReference.dk,
      KernelBackwardReference.dv,KernelBackwardReference.dbeta,KernelBackwardReference.rawDv,
      KernelBackwardReference.input,KernelBackwardReference.keySum,KernelBackwardReference.valueSum,
      KernelBackwardReference.betaSum,KernelReference.add,KernelReference.mul,
      KernelReference.read,KernelReference.cast,KernelReference.zero,KernelReference.scalar,
      KernelReference.stored,keyStore,betaStore,KernelMemory.rowStore,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      Nat.add_zero,Bool.false_eq_true,ite_false,ite_true,KernelBackwardValues.eta]
    first | apply KernelStore.masked_mem | apply write_mem
    try simp only [State.setReg_mem]
    first | apply KernelStore.masked_mem | apply write_mem
    try simp only [State.setReg_mem]
    apply KernelStore.masked_mem
    rfl

#axiomsClean key_cell
#axiomsClean key_frame
#axiomsClean beta_frame
#axiomsClean memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardMemory
