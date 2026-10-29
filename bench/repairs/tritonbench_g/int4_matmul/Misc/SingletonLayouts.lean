import bench.repairs.tritonbench_g.int4_matmul.Misc.RawNative
namespace VeriTile.Bench.Repairs.Int4Matmul.SingletonLayouts
open Host HelperHost

def q8 : Tensor := ⟨1,1,8,1,1,.int32,0,0,Bool.false,Bool.false,Bool.false⟩
def z3 : Tensor := ⟨2,3,1,1,3,.int32,0,0,Bool.false,Bool.false,Bool.false⟩
theorem q8_exact_return : q8.TransposeContiguous .int32 1 8 0 := by simp [q8,Tensor.TransposeContiguous,transposeRowStride,transposeColumnStride]
theorem q8_old_canonical_false : ¬q8.Canonical .int32 1 8 0 := by simp [q8,Tensor.Canonical]
theorem q8_admitted : q8.PackedLayout .int32 1 8 0 := Or.inr q8_exact_return
theorem z3_exact_return : z3.TransposeContiguous .int32 3 1 0 := by simp [z3,Tensor.TransposeContiguous,transposeRowStride,transposeColumnStride]
theorem z3_old_canonical_false : ¬z3.Canonical .int32 3 1 0 := by simp [z3,Tensor.Canonical]
theorem z3_admitted : z3.PackedLayout .int32 3 1 0 := Or.inr z3_exact_return

/-- The ordinary output-layout contract now composes with packed input admission,
with the same shape, exact caller device and independent allocations. -/
theorem quant_outputs_admitted (s : Shape) (hs : s.Valid) (indices : s.k*s.n<2^31)
    (device input : Nat) (a : QuantAllocation s device input) :
    PackedDomain s a.weights a.scales a.zeros := by
  have hd : a.weights.device=device := a.weightLayout.2.2.2.2.2.1
  obtain ⟨hq,hs',hz⟩ := quant_packed_layout s device input a
  refine ⟨hs,?_,?_,?_,?_,indices⟩
  · simpa [hd] using hq
  · simpa [hd] using hs'
  · simpa [hd] using hz
  · exact (List.nodup_cons.mp a.independent).2

/-- This is the actual source multiplication before division by eight, not only
its quotient. The signed32 transport follows from the new actual-stride guard. -/
theorem raw_advance_signed32 (x : Arguments) (valid : Contract x) (t : Tile)
    (member : t∈sourceTiles) :
    (BitVec.ofNat 32 (t.bk*t.splits*x.packed.rowStride)).toInt=
      (t.bk*t.splits*x.packed.rowStride:Int) :=
  signed32_exact _ (contract_strides x valid t member).rawAdvance

theorem actual_formed_signed32 (x : Arguments) (valid : Contract x) (t : Tile)
    (member : t∈sourceTiles) (k c : Nat) (hk : k<kEnvelope x.shape t) (hc : c<x.shape.n) :
    (BitVec.ofNat 32 (x.packed.elementOffset (k/8) c)).toInt=(x.packed.elementOffset (k/8) c:Int) ∧
    (BitVec.ofNat 32 (x.scale.elementOffset (k/x.shape.group) c)).toInt=
      (x.scale.elementOffset (k/x.shape.group) c:Int) ∧
    (BitVec.ofNat 32 (x.zeros.elementOffset (k/x.shape.group) (c/8))).toInt=
      (x.zeros.elementOffset (k/x.shape.group) (c/8):Int) := by
  obtain ⟨hq,hs,hz⟩ := (contract_strides x valid t member).formed k c hk hc
  exact ⟨signed32_exact _ hq,signed32_exact _ hs,signed32_exact _ hz⟩

def rawOverflowShape : Shape := ⟨1,8,4194304,8⟩
theorem old_shape_guard_admitted : Guard rawOverflowShape := by
  constructor <;> decide +kernel
theorem raw_product_overflow : ¬(1024*4194304<2^31) := by decide +kernel
theorem raw_product_wrap : (BitVec.ofNat 32 (1024*4194304)).toInt=0 := by decide +kernel

#axiomsClean q8_exact_return
#axiomsClean q8_old_canonical_false
#axiomsClean q8_admitted
#axiomsClean z3_exact_return
#axiomsClean z3_old_canonical_false
#axiomsClean z3_admitted
#axiomsClean quant_outputs_admitted
#axiomsClean raw_advance_signed32
#axiomsClean actual_formed_signed32
#axiomsClean old_shape_guard_admitted
#axiomsClean raw_product_overflow
#axiomsClean raw_product_wrap
end VeriTile.Bench.Repairs.Int4Matmul.SingletonLayouts
