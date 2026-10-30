import bench.repairs.tritonbench_g.fused_recurrent_delta.IO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardResults
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardFrame
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOFrame
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters Interface
noncomputable section
variable {α : Type}

theorem forward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : ForwardInputs α s) (r : RegionName) (off : Nat)
    (outside : (forwardIO M s f scale bh iv).Outside vals r off) :
    ¬KernelForwardResults.Writes s f bh iv r off := by
  intro writes
  rcases writes with ⟨role,t,j,ht,hj,same⟩ | ⟨enabled,rfl,i,j,hi,hj,same⟩
  · rcases role with rfl | rfl
    · exact (outside.1 (1 : Fin 3) (⟨t,ht⟩,j,()) hj).elim (fun bad => bad rfl) (fun bad => bad same)
    · exact (outside.1 (0 : Fin 3) (⟨t,ht⟩,j,()) hj).elim (fun bad => bad rfl) (fun bad => bad same)
  · exact (outside.1 (2 : Fin 3) (i,j,()) ⟨enabled,hi,hj⟩).elim (fun bad => bad rfl) (fun bad => bad same)

theorem backward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : BackwardInputs α s) (r : RegionName) (off : Nat)
    (outside : (backwardIO M s f scale bh iv).Outside vals r off) :
    ¬KernelBackwardFrame.Writes s f bh iv r off := by
  intro writes
  rcases writes with ⟨role,t,i,ht,hi,same⟩ | ⟨rfl,t,j,ht,hj,same⟩ | ⟨rfl,beta⟩ | ⟨enabled,rfl,i,j,hi,hj,same⟩
  · rcases role with rfl | rfl
    · exact (outside.1 (1 : Fin 6) (⟨t,ht⟩,i,()) hi).elim (fun bad => bad rfl) (fun bad => bad same)
    · exact (outside.1 (0 : Fin 6) (⟨t,ht⟩,i,()) hi).elim (fun bad => bad rfl) (fun bad => bad same)
  · exact (outside.1 (2 : Fin 6) (⟨t,ht⟩,j,()) hj).elim (fun bad => bad rfl) (fun bad => bad same)
  · cases flag : f.vectorBeta
    · simp only [flag,Bool.false_eq_true,ite_false] at beta
      obtain ⟨t,ht,same⟩ := beta
      exact (outside.1 (4 : Fin 6) (⟨t,ht⟩,()) flag).elim (fun bad => bad rfl) (fun bad => bad same)
    · simp only [flag,ite_true] at beta
      obtain ⟨t,j,ht,hj,same⟩ := beta
      exact (outside.1 (3 : Fin 6) (⟨t,ht⟩,j,()) ⟨flag,hj⟩).elim (fun bad => bad rfl) (fun bad => bad same)
  · exact (outside.1 (5 : Fin 6) (i,j,()) ⟨enabled,hi,hj⟩).elim (fun bad => bad rfl) (fun bad => bad same)

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOFrame
