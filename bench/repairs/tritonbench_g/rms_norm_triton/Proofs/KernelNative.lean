import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section
set_option maxHeartbeats 4000000

/-- Ordinary live storage and fresh allocation facts after the source's two
contiguous materializations. No native index/padded-bound premises are supplied. -/
structure PreparedAPI (a : Whole.Prepared α input weight entry live) (n : List Nat) where
  pointers : Pointers
  capacity : Nat → Nat
  byteBase : Nat → Nat
  inputPointer : 0 < Whole.rows input n → byteBase a.xc.tensor.allocation+4*a.xc.tensor.base=pointers.input.toNat
  weightPointer : byteBase a.wc.tensor.allocation+4*a.wc.tensor.base=pointers.weight.toNat
  outputPointer : 0 < Whole.rows input n → byteBase a.output.tensor.allocation+4*a.output.tensor.base=pointers.output.toNat
  inputBounds : ∀ row i,row<Whole.rows input n → i<Whole.columns n →
    a.xc.tensor.base+row*Whole.columns n+i<capacity a.xc.tensor.allocation
  weightBounds : ∀ i,i<Whole.columns n → a.wc.tensor.base+i<capacity a.wc.tensor.allocation
  outputBounds : ∀ row i,row<Whole.rows input n → i<Whole.columns n →
    a.output.tensor.base+row*Whole.columns n+i<capacity a.output.tensor.allocation
  allocationEnd : ∀ t∈[a.xc.tensor,a.wc.tensor,a.output.tensor],byteBase t.allocation+4*capacity t.allocation≤2^64
  separated : ∀ allocation,(live allocation ∨ allocation=a.xc.tensor.allocation ∨ allocation=a.wc.tensor.allocation) →
    byteBase a.output.tensor.allocation+4*capacity a.output.tensor.allocation≤byteBase allocation ∨
    byteBase allocation+4*capacity allocation≤byteBase a.output.tensor.allocation

private theorem aligned (base offset ptr : Nat) (observed : base+4*offset=ptr)
    (checked : ptr%4=0) : 4∣base := by
  rw [←observed] at checked
  simp [Nat.add_mod] at checked
  exact Nat.dvd_of_mod_eq_zero checked

def PreparedAPI.finite (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x w n eps) (binding : Binding x w input weight live)
    (a : Whole.Prepared α input weight entry live) (api : PreparedAPI a n.dimensions)
    (checked : PointerContract x n.dimensions api.pointers) : Whole.FiniteAdmission a n.dimensions := by
  have rowBound (row : Nat) (h : row<Whole.rows input n.dimensions) : row*Whole.columns n.dimensions≤rowMax x n.dimensions := by
    exact Nat.mul_le_mul_right _ (Nat.le_sub_one_of_lt (by simpa [binding.rows] using h))
  have paddedBound (row i : Nat) (hr : row<Whole.rows input n.dimensions)
      (hi : i<Whole.nextPower (Whole.columns n.dimensions)) :
      row*Whole.columns n.dimensions+i≤paddedMax x n.dimensions := by
    exact Nat.add_le_add (rowBound row hr) (Nat.le_sub_one_of_lt (by simpa [block_eq,columns] using hi))
  refine ⟨?_,c.columnsBound,?_,?_,api.capacity,api.inputBounds,api.weightBounds,api.outputBounds,
    api.byteBase,?_,api.allocationEnd,?_,?_,?_,api.separated⟩
  · rw [binding.rows]; exact c.rowsBound
  · simpa only [block_eq,columns] using c.blockBound
  · intro row hr; exact (rowBound row hr).trans_lt c.rowProducts
  · intro positive t ht
    simp only [List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl
    · exact aligned _ _ _ (api.inputPointer positive) checked.aligned.1
    · exact aligned _ _ _ api.weightPointer checked.aligned.2.1
    · exact aligned _ _ _ (api.outputPointer positive) checked.aligned.2.2
  · intro row i hr hi
    have bound := paddedBound row i hr hi
    calc
      api.byteBase a.xc.tensor.allocation+4*(a.xc.tensor.base+row*Whole.columns n.dimensions+i) =
          api.pointers.input.toNat+4*(row*Whole.columns n.dimensions+i) := by rw [←api.inputPointer (Nat.zero_lt_of_lt hr)]; omega
      _ ≤ api.pointers.input.toNat+4*paddedMax x n.dimensions := by omega
      _ < 2^64 := checked.inputEnd
  · intro i hi
    have bound : i≤block (columns n.dimensions)-1 := Nat.le_sub_one_of_lt (by simpa only [block_eq,columns] using hi)
    calc
      api.byteBase a.wc.tensor.allocation+4*(a.wc.tensor.base+i) = api.pointers.weight.toNat+4*i := by
        rw [←api.weightPointer]; omega
      _ ≤ api.pointers.weight.toNat+4*(block (columns n.dimensions)-1) := by omega
      _ < 2^64 := checked.weightEnd
  · intro row i hr hi
    have bound := paddedBound row i hr hi
    calc
      api.byteBase a.output.tensor.allocation+4*(a.output.tensor.base+row*Whole.columns n.dimensions+i) =
          api.pointers.output.toNat+4*(row*Whole.columns n.dimensions+i) := by rw [←api.outputPointer (Nat.zero_lt_of_lt hr)]; omega
      _ ≤ api.pointers.output.toNat+4*paddedMax x n.dimensions := by omega
      _ < 2^64 := checked.outputEnd

end
end VeriTile.Bench.Repairs.RmsNormTriton
