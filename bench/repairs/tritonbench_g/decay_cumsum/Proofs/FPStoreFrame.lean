import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPMemory
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPBackwardStep

namespace VeriTile.Bench.Repairs.DecayCumsum.FPStoreFrame
open VeriTile.Triton FP.Structural FPOps
noncomputable section
variable {α : Type} [Inhabited α]

theorem scatter_mem_congr (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u v : State α) (he : u.mem=v.mem) :
    (scatter M BK r off value u).mem=(scatter M BK r off value v).mem := by
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
    (v : Values α dtype shape) :
    (scatter M BK r off value (u.setReg name dtype shape v)).mem=(scatter M BK r off value u).mem :=
  scatter_mem_congr M BK r off value _ _ rfl

theorem directScatter_mem_congr (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u v : State α) (he : u.mem=v.mem) :
    (directScatter M BK r off value u).mem=(directScatter M BK r off value v).mem := by
  unfold directScatter
  generalize TileShape.allIndices [BK]=indices
  induction indices generalizing u v with
  | nil => exact he
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [State.write,he]

@[simp] theorem directScatter_setReg (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u : State α) (name : RegName) (dtype : TileDType) (shape : TileShape)
    (v : Values α dtype shape) :
    (directScatter M BK r off value (u.setReg name dtype shape v)).mem=(directScatter M BK r off value u).mem :=
  directScatter_mem_congr M BK r off value _ _ rfl

theorem backward_memory (M : Algebra α) (h : Call) (off : Fin h.BK → Nat)
    (g last acc : Values α .real [h.BK]) (u : State α) :
    (FPBackwardStep.result M h off g last acc u).mem=
      (scatter M h.BK "dg" off (FPBackwardStep.nextCarry M h off g last acc u)
        (directScatter M h.BK "dko" off (FPBackwardStep.kvalue M h off g last u)
          (directScatter M h.BK "dqo" off (FPBackwardStep.qvalue M h off g u) u))).mem := by
  simp only [FPBackwardStep.result,State.setReg_mem]
  rw [scatter_setReg,scatter_setReg,scatter_setReg,scatter_setReg]
  apply scatter_mem_congr
  rw [directScatter_setReg,directScatter_setReg,directScatter_setReg,directScatter_setReg]
  apply directScatter_mem_congr
  rw [directScatter_setReg,directScatter_setReg,directScatter_setReg,directScatter_setReg]

#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.DecayCumsum.FPStoreFrame
