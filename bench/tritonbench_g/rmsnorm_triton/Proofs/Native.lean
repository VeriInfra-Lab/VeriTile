import bench.tritonbench_g.rmsnorm_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 4000000

private theorem row_le (x t : TensorInfo) (b m : Nat) (hb : b<batches x) (hm : m<rows x) :
    b*t.stride 0+m*t.stride 1≤maxOffset x t := by
  have h := Nat.add_le_add
    (Nat.mul_le_mul_right (t.stride 0) (Nat.le_sub_one_of_lt hb))
    (Nat.mul_le_mul_right (t.stride 1) (Nat.le_sub_one_of_lt hm))
  exact h.trans (Nat.le_add_right _ _)

private theorem column_le (x t : TensorInfo) (k : Nat) (hk : k<padded x) :
    k*t.stride 2≤maxOffset x t := by
  have h := Nat.mul_le_mul_right (t.stride 2) (Nat.le_sub_one_of_lt hk)
  exact h.trans (Nat.le_add_left _ _)

private theorem formed_le (x t : TensorInfo) (b m k : Nat)
    (hb : b<batches x) (hm : m<rows x) (hk : k<padded x) :
    b*t.stride 0+m*t.stride 1+k*t.stride 2≤maxOffset x t := by
  exact Nat.add_le_add (Nat.add_le_add
    (Nat.mul_le_mul_right (t.stride 0) (Nat.le_sub_one_of_lt hb))
    (Nat.mul_le_mul_right (t.stride 1) (Nat.le_sub_one_of_lt hm)))
    (Nat.mul_le_mul_right (t.stride 2) (Nat.le_sub_one_of_lt hk))

private theorem weight_le (x w : TensorInfo) (k : Nat) (hk : k<padded x) :
    k*w.stride 0≤maxWeight x w :=
  Nat.mul_le_mul_right (w.stride 0) (Nat.le_sub_one_of_lt hk)

private theorem allocation_aligned (info : TensorInfo) (t : Tensor) (base : Nat → Nat)
    (observed : Observes info t base) (nonempty : info.numel≠0) (aligned : info.dataPtr.toNat%4=0) : 4∣base t.allocation := by
  rw [←observed.pointer nonempty] at aligned
  simp [Nat.add_mod,Nat.mul_mod] at aligned
  exact Nat.dvd_of_mod_eq_zero aligned

/-- Every source-native side condition is derived from checked scalar maxima.
Only ordinary active-storage bounds and allocator separation come from the API. -/
def Binding.finite (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (api : MemoryAPI p live) (b : Binding x w out eps p api.byteBase)
    (nonempty : x.numel≠0) (output : LaunchOutput x out) :
    FiniteAdmission p (b.domain x w out eps c p api) live := by
  have g := c.launch nonempty
  have weightNonempty : w.numel≠0 := by
    simpa [VeriTile.Torch.TensorMetadata.numel,c.weightShape] using Nat.ne_of_gt c.positive
  have outputNonempty : out.numel≠0 := by
    simpa [VeriTile.Torch.TensorMetadata.numel,b.outputShape] using nonempty
  have hp : trips p*1024=padded x := by simp [trips,padded,b.column]
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,api.byteBase,?_,api.allocationEnd,?_,?_,?_,api.freshBytes⟩
  · rw [hp]; exact g.loop
  · rw [b.batch]; exact g.gridX
  · rw [b.row]; exact g.gridY
  · intro ax hax
    rw [b.input.stride]
    interval_cases ax
    · exact g.strides.1
    · exact g.strides.2.1
    · exact g.strides.2.2.1
  · rw [b.weight.stride]; exact g.strides.2.2.2
  · intro ax hax
    rw [b.output.stride]
    interval_cases ax
    · exact output.strides.1
    · exact output.strides.2.1
    · exact output.strides.2.2
  · intro batch row hb hm
    rw [b.input.stride]
    exact (row_le x x batch row (by simpa [b.batch] using hb) (by simpa [b.row] using hm)).trans_lt g.offsets.1
  · intro k hk
    rw [b.input.stride]
    exact (column_le x x k (by simpa [hp] using hk)).trans_lt g.offsets.1
  · intro k hk
    rw [b.weight.stride]
    exact (weight_le x w k (by simpa [hp] using hk)).trans_lt g.offsets.2
  · intro batch row k hb hm hk
    rw [b.output.stride]
    exact (formed_le x out batch row k (by simpa [b.batch] using hb)
      (by simpa [b.row] using hm) (by simpa [hp] using hk)).trans_lt output.offset
  · intro t ht
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl
    · exact allocation_aligned x p.input api.byteBase b.input nonempty g.alignment.1
    · exact allocation_aligned w p.weights api.byteBase b.weight weightNonempty g.alignment.2
    · exact allocation_aligned out p.output api.byteBase b.output outputNonempty output.alignment
  · intro batch row k hb hm hk
    have bound := formed_le x x batch row k (by simpa [b.batch] using hb)
      (by simpa [b.row] using hm) (by simpa [hp] using hk)
    calc
      api.byteBase p.input.allocation+4*inputOffset p batch row k =
          x.dataPtr.toNat+4*(batch*x.stride 0+row*x.stride 1+k*x.stride 2) := by
        rw [←b.input.pointer nonempty]
        simp only [inputOffset,b.input.stride]
        omega
      _ ≤ x.dataPtr.toNat+4*maxOffset x x := by omega
      _ < 2^64 := g.pointers.1
  · intro k hk
    have bound := weight_le x w k (by simpa [hp] using hk)
    calc
      api.byteBase p.weights.allocation+4*weightOffset p k =
          w.dataPtr.toNat+4*(k*w.stride 0) := by
        rw [←b.weight.pointer weightNonempty]
        simp only [weightOffset,b.weight.stride]
        omega
      _ ≤ w.dataPtr.toNat+4*maxWeight x w := by omega
      _ < 2^64 := g.pointers.2
  · intro batch row k hb hm hk
    have bound := formed_le x out batch row k (by simpa [b.batch] using hb)
      (by simpa [b.row] using hm) (by simpa [hp] using hk)
    calc
      api.byteBase p.output.allocation+4*outputOffset p batch row k =
          out.dataPtr.toNat+4*(batch*out.stride 0+row*out.stride 1+k*out.stride 2) := by
        rw [←b.output.pointer outputNonempty]
        simp only [outputOffset,b.output.stride]
        omega
      _ ≤ out.dataPtr.toNat+4*maxOffset x out := by omega
      _ < 2^64 := output.pointer

theorem native_correspondence (x w out : TensorInfo) (eps : EpsilonInfo) (c : InputContract x w eps)
    (p : Parameters) (api : MemoryAPI p live) (b : Binding x w out eps p api.byteBase)
    (nonempty : x.numel≠0) (checkedOutput : checkOutput x out=.ok ()) :
    IndexCorrespondence p (b.domain x w out eps c p api)
      (b.finite x w out eps c p api nonempty ((checkOutput_iff x out).mp checkedOutput)) :=
  index_correspondence _ _ _

#axiomsClean native_correspondence
end
end VeriTile.Bench.Repairs.RmsnormTriton
