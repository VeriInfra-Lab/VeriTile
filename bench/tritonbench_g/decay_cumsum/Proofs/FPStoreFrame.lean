import bench.tritonbench_g.decay_cumsum.Proofs.FPMemory
import bench.tritonbench_g.decay_cumsum.Proofs.FPBackwardStep

namespace VeriTile.Bench.Repairs.DecayCumsum.FPStoreFrame
open VeriTile.Triton FP.Structural FPOps
noncomputable section
variable {α : Type} [Inhabited α]

theorem scatter_mem_congr (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u v : State α) (he : u.mem=v.mem) (explicitCast : Bool := Bool.true) :
    (scatter M BK r off value u explicitCast).mem=(scatter M BK r off value v explicitCast).mem := by
  unfold scatter
  generalize TileShape.allIndices [BK]=indices
  induction indices generalizing u v with
  | nil => exact he
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [State.write,he]

@[simp] theorem scatter_setReg (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u : State α) (name : RegName) (dtype : TileDType) (shape : TileShape)
    (v : Values α dtype shape) (explicitCast : Bool := Bool.true) :
    (scatter M BK r off value (u.setReg name dtype shape v) explicitCast).mem=(scatter M BK r off value u explicitCast).mem :=
  scatter_mem_congr M BK r off value (u.setReg name dtype shape v) u rfl explicitCast

theorem backward_memory (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (g last acc : Values α .real [h.BK]) (u : State α) :
    (FPBackwardStep.result M h off g last acc u).mem=
      (scatter M h.BK "dg" off (FPBackwardStep.nextCarry M h off g last acc u)
        (scatter M h.BK "dko" off (FPBackwardStep.kvalue M h off g last u)
          (scatter M h.BK "dqo" off (FPBackwardStep.qvalue M h off g u) u Bool.false) Bool.false)).mem := by
  simp only [FPBackwardStep.result,State.setReg_mem]
  rw [scatter_setReg,scatter_setReg,scatter_setReg,scatter_setReg]
  apply scatter_mem_congr
  rw [scatter_setReg (explicitCast := Bool.false),scatter_setReg (explicitCast := Bool.false),
    scatter_setReg (explicitCast := Bool.false),scatter_setReg (explicitCast := Bool.false)]
  apply scatter_mem_congr (explicitCast := Bool.false)
  rw [scatter_setReg (explicitCast := Bool.false),scatter_setReg (explicitCast := Bool.false),
    scatter_setReg (explicitCast := Bool.false),scatter_setReg (explicitCast := Bool.false)]

#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.DecayCumsum.FPStoreFrame
