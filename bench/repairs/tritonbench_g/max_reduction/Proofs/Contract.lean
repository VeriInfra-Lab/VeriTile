import bench.repairs.tritonbench_g.max_reduction.Contract
import bench.repairs.tritonbench_g.max_reduction.Proofs.Address

namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Torch
noncomputable section

theorem check_global_iff (x : TensorMetadata) : checkGlobal (.tensor x)=.ok () ↔ GlobalContract x := by
  classical
  simp only [checkGlobal,GlobalContract]
  split_ifs <;> simp_all <;> omega

theorem check_dimension_iff (a : DimArguments) : checkDimension a=.ok () ↔ DimContract a := by
  classical
  simp only [checkDimension,DimContract]
  split_ifs <;> simp_all

theorem nextPower_positive (n : Nat) : 0<nextPower n := by
  unfold nextPower
  split_ifs
  · omega
  · positivity

theorem nextPower_covers (n : Nat) : n≤nextPower n := by
  unfold nextPower
  split_ifs with h
  · exact h
  · have := Nat.lt_log2_self (n:=n-1)
    omega

theorem nextPower_bound (n k : Nat) (h : n≤2^k) : nextPower n≤2^k := by
  unfold nextPower
  split_ifs with hsmall
  · exact Nat.one_le_pow k 2 (by decide)
  · have hn : 0<n-1 := by omega
    have hl : (n-1).log2<k := (Nat.log2_lt (Nat.ne_of_gt hn)).mpr (by omega)
    exact Nat.pow_le_pow_right (by decide) (by omega)

theorem global_block_bound (x : TensorMetadata) (h : GlobalContract x) :
    0<globalBlock x.numel ∧ globalBlock x.numel≤65536 := by
  refine ⟨nextPower_positive _,?_⟩
  apply nextPower_bound _ 16
  have hsize := h.1.2.2.2.2.2.1
  have hs : Nat.sqrt (x.numel-1)<65536 := Nat.sqrt_lt.mpr (by omega)
  omega

theorem axis_bound (a : DimArguments) (h : DimContract a) : a.axis<a.input.shape.length := by
  have hr : 0<(a.input.shape.length : Int) := by have := h.2.1; have := h.2.2.1; omega
  have lo := Int.emod_nonneg a.dim (ne_of_gt hr)
  have hi := Int.emod_lt_of_pos a.dim hr
  unfold DimArguments.axis
  omega

theorem shape_decomposition (a : DimArguments) (h : DimContract a) :
    a.input.shape=a.input.shape.take a.axis++[a.N]++a.input.shape.drop (a.axis+1) := by
  have hb := axis_bound a h
  have hd := List.drop_eq_getElem_cons hb
  have ht := List.take_append_drop a.axis a.input.shape
  rw [hd] at ht
  simpa only [DimArguments.N,List.getD,List.getElem?_eq_getElem hb,Option.getD_some,List.singleton_append,List.append_assoc] using ht.symm

theorem dimension_numel (a : DimArguments) (h : DimContract a) :
    a.M*a.N*a.K=a.input.numel := by
  have hs := congrArg List.prod (shape_decomposition a h)
  simpa [TensorMetadata.numel,DimArguments.M,DimArguments.K,List.prod_append,mul_assoc] using hs.symm

theorem output_numel (a : DimArguments) : a.outputShape.prod=a.M*a.K := by
  simp [DimArguments.outputShape,DimArguments.M,DimArguments.K,List.prod_append]
  split <;> simp

theorem block_choice (b : Nat) (h : b∈blockChoices) : 0<b ∧ b≤32 := by
  simp only [blockChoices,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with rfl|rfl|rfl <;> decide

/-- Metadata contiguity and the negative-view check connect direct kernel
loads to logical Tensor elements through the shared storage API laws. The
kernel IO statements additionally bind these values to their input regions. -/
theorem checked_contiguous_read (read : Nat → VeriTile.Torch.FP32.Word)
    (t : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws t)
    (h : TensorValid t.metadata Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    read ((VeriTile.Torch.FP32.dataView t laws (h.2.2.2.2.1 rfl)).address i) =
      VeriTile.Torch.FP32.logicalValue read t laws i :=
  VeriTile.Torch.FP32.direct_read_eq_logical read t laws
    (h.2.2.2.2.1 rfl) h.2.2.2.1 i hi

#axiomsClean checked_contiguous_read
#axiomsClean check_global_iff
#axiomsClean check_dimension_iff
#axiomsClean dimension_numel
#axiomsClean global_block_bound
end
end VeriTile.Bench.Repairs.MaxReduction
