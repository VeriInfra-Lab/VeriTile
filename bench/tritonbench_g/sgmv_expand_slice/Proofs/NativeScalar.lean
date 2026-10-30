import bench.tritonbench_g.sgmv_expand_slice.Proofs.NativeBackend
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceSuffix

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1200000

@[simp] theorem unpack_pack (f : Native.Format) (x : Native.Word f) :
    unpack f (pack f x) = x := by
  cases f <;> simp [pack,unpack,Native.Format.bits]

@[simp] theorem format_input (f : ScalarFormat) : format f.dtype = scalar f := by cases f <;> rfl
@[simp] theorem format_weight (f : LowFormat) : format f.scalar.dtype = (weight f).format := by cases f <;> rfl
@[simp] theorem format_storage (f : ScalarFormat) : format (SourceSuffix.storageType f) = scalar f := by cases f <;> rfl

theorem typed_zero (rest : Algebra Raw) (ops : Native.Primitives) (profile : Native.Profile)
    (h : ZeroConversions ops) (f : ScalarFormat) :
    (algebra rest ops profile).cast none .real f.dtype
      ((algebra rest ops profile).literal none .real 0) = pack (scalar f) 0 := by
  cases f <;> simp [algebra,format,scalar,ScalarFormat.dtype,pack,unpack,
    Native.convertIfNeeded,Native.Format.bits]
  · simpa only [BitVec.setWidth_zero] using congrArg (fun x : BitVec 16 => x.setWidth 32) h.half
  · simpa only [BitVec.setWidth_zero] using congrArg (fun x : BitVec 16 => x.setWidth 32) h.bfloat

theorem cast_input (rest : Algebra Raw) (ops : Native.Primitives) (abi : SourceABI)
    (value : Native.Word (scalar abi.input)) :
    (if abi.input = abi.weight.scalar then pack (scalar abi.input) value else
      (algebra rest ops (profile abi)).cast none abi.input.dtype abi.weight.scalar.dtype
        (pack (scalar abi.input) value)) =
      pack (weight abi.weight).format
        (Native.convertIfNeeded ops (scalar abi.input) (weight abi.weight).format value) := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases input <;> cases weight <;>
    simp [algebra,format,scalar,NativeBackend.weight,ScalarFormat.dtype,LowFormat.scalar,
      pack,unpack,Native.convertIfNeeded,Native.Format.bits,Native.WeightFormat.format]

theorem finish_matches (rest : Algebra Raw) (ops : Native.Primitives) (abi : SourceABI)
    (acc : Raw) (old : Native.Word (scalar abi.output)) :
    SourceSuffix.finish (algebra rest ops (profile abi)) abi acc (pack (scalar abi.output) old) =
      pack (scalar abi.output) (Native.finish ops (profile abi) acc old) := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases add <;>
    simp [SourceSuffix.finish,SourceSuffix.castIfNeeded,SourceSuffix.storePrecision,
      SourceSuffix.additionType,SourceSuffix.storageType,algebra,profile,format,scalar,
      NativeBackend.weight,ScalarFormat.dtype,LowFormat.scalar,pack,unpack,
      Native.finish,Native.convertIfNeeded,Native.additionFormat,Native.additionOldFirst,
      Native.Format.bits,Native.WeightFormat.format]

#axiomsClean unpack_pack
#axiomsClean typed_zero
#axiomsClean cast_input
#axiomsClean finish_matches
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
