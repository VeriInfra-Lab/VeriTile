import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Kernels

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Safety
open MatmulDequantInt4.Direct
noncomputable section

/-- Every formed padded-lane address retains the original signed-32 bound,
including singleton strides. The axis repair adds no new product bound. -/
theorem formed (p : PackedInfo) (f : Source.Profile) (h : PackedContract p 1 f.hostDtype)
    (i : Fin 8) (k n : Nat)
    (hk : k<MatmulDequantInt4.ceil (p.weight.tensor.rows*8) (tuning i).bk*(tuning i).bk)
    (hn : n<MatmulDequantInt4.ceil p.weight.tensor.columns (tuning i).bn*(tuning i).bn) :
    k/8*p.weight.tensor.strides.row+n*p.weight.tensor.strides.column<2^31 ∧
    k/p.group*p.scale.tensor.strides.row+n*p.scale.tensor.strides.column<2^31 ∧
    k/p.group*p.zero.tensor.strides.row+n/8*p.zero.tensor.strides.column<2^31 ∧
    k*p.weight.tensor.columns+n<2^31 := by
  have envelopes := MatmulDequantInt4.Source.dequant_envelopes (packed p 1).shape (tuning i) (tuning_mem i)
  exact MatmulDequantInt4.Strides.formed (packed p 1).strides (packed p 1).shape h.1.2.1.indices k n
    (lt_of_lt_of_le hk envelopes.1) (lt_of_lt_of_le hn envelopes.2)

/-- Primitive pointer arithmetic for every formed source address. -/
theorem pointers (p : PackedInfo) (f : Source.Profile) (o : Info)
    (h : PackedContract p 1 f.hostDtype) (output : OutputABI p f o)
    (i : Fin 8) (k n : Nat)
    (hk : k<MatmulDequantInt4.ceil (p.weight.tensor.rows*8) (tuning i).bk*(tuning i).bk)
    (hn : n<MatmulDequantInt4.ceil p.weight.tensor.columns (tuning i).bn*(tuning i).bn) :
    p.weight.tensor.base+4*(k/8*p.weight.tensor.strides.row+n*p.weight.tensor.strides.column)<2^64 ∧
    p.scale.tensor.base+f.hostDtype.width*(k/p.group*p.scale.tensor.strides.row+n*p.scale.tensor.strides.column)<2^64 ∧
    p.zero.tensor.base+4*(k/p.group*p.zero.tensor.strides.row+n/8*p.zero.tensor.strides.column)<2^64 ∧
    o.tensor.base+f.hostDtype.width*(k*p.weight.tensor.columns+n)<2^64 := by
  have bounds := formed p f h i k n hk hn
  exact ⟨formed_pointer _ _ _ _ _ h.1.2.2.1 _ bounds.1,
    formed_pointer _ _ _ _ _ h.1.2.2.2.1 _ bounds.2.1,
    formed_pointer _ _ _ _ _ h.1.2.2.2.2 _ bounds.2.2.1,
    formed_pointer _ _ _ _ _ output.1 _ bounds.2.2.2⟩

#axiomsClean formed
#axiomsClean pointers
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Safety
