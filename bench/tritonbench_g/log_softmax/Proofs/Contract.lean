import bench.tritonbench_g.log_softmax.Contract
import bench.tritonbench_g.log_softmax.Proofs.Grid
import bench.tritonbench_g.log_softmax.Proofs.Specification

namespace VeriTile.Bench.Repairs.LogSoftmax
open VeriTile.Torch
noncomputable section

theorem checkInputs_iff (a : Arguments) : checkInputs a=.ok () ↔ InputContract a := by
  classical
  simp only [checkInputs,InputContract,ShapeContract]
  split_ifs <;> simp_all

theorem tensor_contiguous (x : TensorMetadata) :
    MaxReduction.TensorValid x Bool.true ↔ MaxReduction.TensorValid x Bool.false ∧ x.isContiguous=Bool.true := by
  simp [MaxReduction.TensorValid]
  tauto

theorem checkBackward_iff (a : BackwardArguments) : checkBackward a=.ok () ↔ BackwardContract a := by
  classical
  simp only [checkBackward,BackwardContract,tensor_contiguous]
  split_ifs <;> simp_all

theorem axis_bound (a : Arguments) (h : ShapeContract a) : a.axis<a.input.shape.length := by
  have hr : 0<(a.input.shape.length : Int) := by have := h.1; have := h.2.1; omega
  have lo := Int.emod_nonneg a.dim (ne_of_gt hr)
  have hi := Int.emod_lt_of_pos a.dim hr
  unfold Arguments.axis
  omega

theorem shape_decomposition (a : Arguments) (h : ShapeContract a) :
    a.input.shape=a.input.shape.take a.axis++[a.N]++a.input.shape.drop (a.axis+1) := by
  have hb := axis_bound a h
  have hd := List.drop_eq_getElem_cons hb
  have ht := List.take_append_drop a.axis a.input.shape
  rw [hd] at ht
  simpa only [Arguments.N,List.getD,List.getElem?_eq_getElem hb,Option.getD_some,List.singleton_append,List.append_assoc] using ht.symm

theorem input_numel (a : Arguments) (h : ShapeContract a) : a.M*a.N*a.K=a.input.numel := by
  have hs := congrArg List.prod (shape_decomposition a h)
  simpa [TensorMetadata.numel,Arguments.M,Arguments.K,List.prod_append,mul_assoc] using hs.symm

theorem block_positive (N : Nat) : 0<Grid.blockColumns N := MaxReduction.nextPower_positive N
theorem block_covers (N : Nat) : N≤Grid.blockColumns N := MaxReduction.nextPower_covers N

/-- Empty calls require no successful compiler configuration. Selected
specification graphs still exist for every finite width, so empty accepted
shapes do not become vacuous merely because no tile would be compiled. -/
theorem plan_exists (N C : Nat) (positive : 0<C) (covers : N≤C) : Nonempty (Plan N C) := by
  have length : (List.finRange C).length=C := by simp
  cases h : List.finRange C with
  | nil => rw [h] at length; simp at length; omega
  | cons first rest =>
    let tree := specificationTree first rest
    have complete : tree.leaves.Perm (List.finRange C) := by rw [h,specificationTree_leaves]
    exact ⟨⟨fun _ => tree,fun _ => tree,fun _ => tree,
      fun _ => complete,fun _ => complete,fun _ => complete,covers,.separate⟩⟩

theorem selected_exists (M N K BM : Nat) (retained : Grid.RetainedBlock BM) :
    Nonempty (Grid.Selected M N K) := by
  obtain ⟨plan⟩ := plan_exists N (Grid.blockColumns N) (block_positive N) (block_covers N)
  exact ⟨⟨BM,Grid.blockColumns N,retained,rfl,Grid.warps N,rfl,plan,
    Grid.cells M N K BM (Grid.blockColumns N),List.Perm.refl _⟩⟩

theorem launch_bounds (a : Arguments) (h : InputContract a) (nonempty : 0<a.input.numel)
    (BM : Nat) (retained : Grid.RetainedBlock BM) :
    0<a.M ∧ 0<a.N ∧ 0<a.K ∧ a.K≤65535 ∧ BM*Grid.blockColumns a.N≤1048576 ∧
      Grid.groups a.M BM*a.K<2^31 := by
  have size := input_numel a h.2.2
  have ext := h.2.2.2.2.resolve_left (by omega)
  have bm := Grid.block_bounds retained
  have capacity : Grid.blockColumns a.N≤131072 := MaxReduction.nextPower_bound _ 17 ext.1
  have nz : a.M*a.N*a.K≠0 := by omega
  have positive : 0<a.M ∧ 0<a.N ∧ 0<a.K := by
    simp only [Nat.mul_ne_zero_iff] at nz
    exact ⟨Nat.pos_of_ne_zero nz.1.1,Nat.pos_of_ne_zero nz.1.2,Nat.pos_of_ne_zero nz.2⟩
  have groups : Grid.groups a.M BM≤a.M := by
    apply Nat.le_of_lt_succ
    apply (Nat.div_lt_iff_lt_mul bm.1).mpr
    have hm : a.M≤a.M*BM := Nat.le_mul_of_pos_right _ bm.1
    rw [Nat.succ_mul]
    omega
  have flat : a.M*a.K≤a.M*a.N*a.K :=
    Nat.mul_le_mul_right _ (Nat.le_mul_of_pos_right _ positive.2.1)
  refine ⟨positive.1,positive.2.1,positive.2.2,ext.2,by nlinarith,?_⟩
  exact lt_of_le_of_lt ((Nat.mul_le_mul_right _ groups).trans flat)
    (by rw [size]; exact h.1.2.2.2.2.2.1)

#axiomsClean checkInputs_iff
#axiomsClean checkBackward_iff
#axiomsClean input_numel
#axiomsClean selected_exists
#axiomsClean launch_bounds
end
end VeriTile.Bench.Repairs.LogSoftmax
