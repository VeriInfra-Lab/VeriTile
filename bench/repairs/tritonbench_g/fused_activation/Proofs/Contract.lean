import bench.repairs.tritonbench_g.fused_activation.Contract
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedActivation

theorem checkTensorInputs_iff (x b i : TensorMetadata) :
    checkTensorInputs x b i = .ok () ↔ InputContract x b i := by
  by_cases empty : x.numel = 0
  · simp [checkTensorInputs, InputContract, empty, VeriTile.Torch.TensorMetadata.isCuda,
      ite_eq_iff, and_assoc]
  · have pos : 0 < x.numel := Nat.pos_of_ne_zero empty
    simp [checkTensorInputs, InputContract, empty, pos, VeriTile.Torch.TensorMetadata.isCuda,
      ite_eq_iff, and_assoc, Nat.pos_iff_ne_zero]

theorem checkInputs_iff (x b i : ArgumentMetadata) :
    checkInputs x b i = .ok () ↔ AdmittedArguments x b i := by
  cases x <;> cases b <;> cases i <;> simp [checkInputs, AdmittedArguments, checkTensorInputs_iff]

theorem block_positive (n : Nat) : 0 < block n :=
  lt_min (by decide) (Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo n))

theorem block_power (n : Nat) : ∃ k, k ≤ 11 ∧ block n = 2^k := by
  by_cases h : 2048 ≤ n.nextPowerOfTwo
  · exact ⟨11, by decide, by simp [block, Nat.min_eq_left h]⟩
  · obtain ⟨k,hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo n
    have bound : 2^k < 2048 := by omega
    refine ⟨k, ?_, by rw [block,hk,Nat.min_eq_right (Nat.le_of_lt bound)]⟩
    by_contra bad
    have hp : (2:Nat)^12 ≤ 2^k := Nat.pow_le_pow_right (by decide) (by omega)
    norm_num at hp
    omega

theorem supported_block (n : Nat) : block n ∈ TritonMul2.supportedBlocks := by
  obtain ⟨k,hk,hb⟩ := block_power n
  rw [hb]
  interval_cases k <;> decide

theorem native_indices (n pid lane : Nat) (hn : n < 2^31)
    (hp : pid < (n+block n-1)/block n) (hl : lane < block n) :
    (TritonMul2Finite.offset32 (block n) pid lane).toInt =
      (pid*block n+lane : Nat) ∧
    TritonMul2Finite.active32 n (block n) pid lane =
      decide (pid*block n+lane<n) :=
  TritonMul2.native_indices n (block n) pid lane hn (supported_block n) hp hl

#axiomsClean checkTensorInputs_iff
#axiomsClean checkInputs_iff
#axiomsClean block_power
#axiomsClean native_indices
end VeriTile.Bench.Repairs.FusedActivation
