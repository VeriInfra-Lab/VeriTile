import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceBody
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Raw := BitVec 32

def scalar : ScalarFormat → Native.Format
  | .half => .half | .bfloat => .bfloat | .single => .single
def weight : LowFormat → Native.WeightFormat
  | .half => .half | .bfloat => .bfloat
def profile (abi : SourceABI) : Native.Profile :=
  ⟨scalar abi.input,weight abi.weight,scalar abi.output,abi.addInputs⟩

/-- The .real channel carries explicit FP32 operations in this source; it is
not a mathematical-real backend here. Unsupported float8 channels are unused. -/
def format : FloatDType → Native.Format
  | .fp16 => .half | .bf16 => .bfloat | _ => .single
def pack (f : Native.Format) (x : Native.Word f) : Raw := x.setWidth 32
def unpack (f : Native.Format) (x : Raw) : Native.Word f := x.setWidth f.bits

def algebra (rest : Algebra Raw) (ops : Native.Primitives) (p : Native.Profile) : Algebra Raw :=
  {rest with
    literal := fun _ _ _ => 0
    cast := fun _ source target x =>
      pack (format target) (Native.convertIfNeeded ops (format source) (format target)
        (unpack (format source) x))
    binary := fun precision dtype operation a b => match operation with
      | .add =>
          pack (format dtype)
            (if Native.additionOldFirst p then
              ops.add (format dtype) (unpack (format dtype) b) (unpack (format dtype) a)
             else ops.add (format dtype) (unpack (format dtype) a) (unpack (format dtype) b))
      | other => rest.binary precision dtype other a b
    fp32Load := id}

/-- The only selected dot case is assembled from eight local m16n8k16 ports.
Other shapes are a total structural fallback and never occur in this source. -/
def dot (ops : Native.Primitives) (f : Native.WeightFormat) : SourceBody.DotImplementation Raw where
  apply := by
    intro precision left right output options batch m k n a b acc
    by_cases hb : batch = []
    · subst batch
      by_cases hm : m = 32
      · subst m
        by_cases hk : k = 16
        · subst k
          by_cases hn : n = 32
          · subst n
            exact fun i => Native.mmaTile ops f
              (fun r q => unpack f.format (a (r,q,PUnit.unit)))
              (fun q c => unpack f.format (b (q,c,PUnit.unit)))
              (fun r c => acc (r,c,PUnit.unit)) i.1 i.2.1
          · exact fun _ => 0
        · exact fun _ => 0
      · exact fun _ => 0
    · exact fun _ => 0

/-- These are precise local constant-conversion laws, required to match typed
padding zeros to the native reference. They say nothing about program execution. -/
structure ZeroConversions (ops : Native.Primitives) : Prop where
  half : ops.convert .single .half 0 = 0
  bfloat : ops.convert .single .bfloat 0 = 0

theorem zero_conversion_witness : ZeroConversions Native.zeroPrimitives := ⟨rfl,rfl⟩

theorem selected_dot (ops : Native.Primitives) (f : Native.WeightFormat)
    (precision : Option ComputeDType) (left right output : FloatDType) (options : DotOptions)
    (a : TileIndex [32,16] → Raw) (b : TileIndex [16,32] → Raw)
    (acc : TileIndex [32,32] → Raw) (i : TileIndex [32,32]) :
    (dot ops f).apply precision left right output options
      (batch := []) (m := 32) (k := 16) (n := 32) a b acc i =
    Native.mmaTile ops f
      (fun r q => unpack f.format (a (r,q,PUnit.unit)))
      (fun q c => unpack f.format (b (q,c,PUnit.unit)))
      (fun r c => acc (r,c,PUnit.unit)) i.1 i.2.1 := by
  simp [dot]

#axiomsClean selected_dot
#axiomsClean zero_conversion_witness
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
