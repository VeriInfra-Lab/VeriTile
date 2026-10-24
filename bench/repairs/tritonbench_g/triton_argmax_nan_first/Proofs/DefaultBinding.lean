import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Default
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Global

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Default
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section

set_option maxRecDepth 30000
set_option maxHeartbeats 3000000

def compactKey (word : Word) : Nat :=
  if word.toNat%2147483648=0 then 2147483647 else MachineMax.rank .fp32 word

theorem nan32 (word : Word) : floatNaN .fp32 word=MachineMax.isNaN .fp32 word := by
  have bound : word.toNat%2147483648<2147483648 := Nat.mod_lt _ (by decide)
  change decide (2139095040<word.toNat%2147483648)=
    decide ((word.toNat%2147483648)/8388608=255 ∧ (word.toNat%2147483648)%8388608≠0)
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  omega

theorem key32 (word : Word) : floatKey .fp32 word=
    if MachineMax.isNaN .fp32 word then ⊤ else (Int.ofNat (compactKey word) : WithTop Int) := by
  rw [floatKey,nan32]
  simp only [floatMagnitude,RealScratch.FloatingKind.format,compactKey,MachineMax.rank,BinaryConversion.key32_toNat]
  norm_num
  split_ifs <;> rfl

theorem compact_lt (x y : Word) : compactKey x<compactKey y ↔ MachineMax.rank .fp32 x<MachineMax.rank .fp32 y := by
  have hx : x.toNat<4294967296 := x.isLt
  have hy : y.toNat<4294967296 := y.isLt
  simp only [compactKey,MachineMax.rank,BinaryConversion.key32_toNat]
  split_ifs <;> omega

theorem compact_eq (x y : Word) : compactKey x=compactKey y ↔ MachineMax.rank .fp32 x=MachineMax.rank .fp32 y := by
  have hx : x.toNat<4294967296 := x.isLt
  have hy : y.toNat<4294967296 := y.isLt
  simp only [compactKey,MachineMax.rank,BinaryConversion.key32_toNat]
  split_ifs <;> omega

theorem gt32 (x y : Word) : orderedGt (floatKey .fp32 x) (floatKey .fp32 y)=greater x y := by
  rw [key32,key32,greater_priority]
  cases hx : MachineMax.isNaN .fp32 x <;> cases hy : MachineMax.isNaN .fp32 y <;>
    simp [priority,hx,hy,orderedGt,compact_lt]

theorem eq32 (x y : Word) : orderedEq (floatKey .fp32 x) (floatKey .fp32 y)=equal x y := by
  rw [key32,key32,equal_priority]
  cases hx : MachineMax.isNaN .fp32 x <;> cases hy : MachineMax.isNaN .fp32 y <;>
    simp [priority,hx,hy,orderedEq,compact_eq]

theorem source32 (c : GlobalConfig) (input : Nat → Word) :
    Source.globalIndex (floatingScratch .fp32 id).scalars c input=Source.globalIndex fp32Scalars c input := by
  have gt : (fun x y : Word => orderedGt (floatKey .fp32 x) (floatKey .fp32 y))=greater := funext fun x => funext fun y => gt32 x y
  have eq : (fun x y : Word => orderedEq (floatKey .fp32 x) (floatKey .fp32 y))=equal := funext fun x => funext fun y => eq32 x y
  have pad : floatPadding .fp32=(0xff800000 : Word) := by decide
  have ops : (floatingScratch .fp32 id).scalars=
      (⟨greater,equal,0xff800000,id,greater,equal,0xff800000⟩ : Scalars RealScratch.FloatingKind.fp32.format) := by
    change (⟨greater,equal,0xff800000,id,_,_,floatPadding .fp32⟩ : Scalars RealScratch.FloatingKind.fp32.format)=_
    dsimp only [floatingScratch]
    rw [pad]
    exact congrArg₂ (fun g e : Word → Word → Bool =>
      (⟨greater,equal,0xff800000,id,g,e,0xff800000⟩ : Scalars RealScratch.FloatingKind.fp32.format)) gt eq
  rw [ops]
  rfl

/-- The default FP32 request strengthens the staged target to the original
input's least NaN index, or least numerical maximum when there is no NaN. -/
theorem staged_reference (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    globalReference (floatingScratch .fp32 id) c coverage.blockPositive
      (coverage.blocksPositive.trans_le coverage.midFits) input=
      (reference c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val := by
  rw [←global_reference (floatingScratch .fp32 id) c coverage.blockPositive
    (coverage.blocksPositive.trans_le coverage.midFits) input coverage.firstComplete coverage.secondComplete,
    source32,reference_index c coverage input]

#axiomsClean source32
#axiomsClean staged_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Default
