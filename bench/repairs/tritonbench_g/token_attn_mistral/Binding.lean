import bench.repairs.tritonbench_g.token_attn_mistral.Bounds
import bench.repairs.tritonbench_g.token_attn_mistral.Profiles
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host
open VeriTile.Triton FP.Structural Grid

/-- Triton's recorded integer specialization key: constexpr one, div16, or
ordinary i32. Values themselves are still loaded from the actual Call. -/
def argumentClass (n : Nat) : Nat := if n=1 then 1 else if n%16=0 then 16 else 0

def profileStride : Native.Profile → Nat
  | .d16Strided | .d64Strided => 2
  | _ => 1

def profilePointerClass : Native.Profile → Nat
  | .d16Unaligned => 0
  | _ => 16

def pointerClass (t : Tensor) : Nat := if byteAddress t 0%16=0 then 16 else 0

/-- Exact non-numerical JIT specialization class of each recorded profile.
Together with Layout's dense table/V/Out, these are all dynamic argument
classes and constexpr dimensions in the saved actual binder reports.
This is not a statement about compiler execution or reduction outputs. -/
structure CompilerBinding (c : Call) (profile : Native.Profile) : Prop where
  width : c.value.width=profile.width
  pointers : ∀ t∈[c.probability.toTensor,c.value.toTensor,c.output.toTensor,c.pages.toTensor,
      c.request.toTensor,c.unusedStart.toTensor,c.sequence.toTensor,c.start.toTensor,c.length.toTensor],
    pointerClass t=profilePointerClass profile
  metadata : ∀ t∈[c.request,c.sequence,c.start,c.length],
    argumentClass t.stride=argumentClass (profileStride profile)
  probabilityToken : argumentClass c.probability.colStride=argumentClass (profileStride profile)
  probabilityHead : argumentClass c.probability.rowStride=0
  tableRow : argumentClass c.pages.rowStride=0
  valueBatch : argumentClass c.value.slotStride=argumentClass (2*profile.width)
  valueHead : argumentClass c.value.headStride=argumentClass profile.width
  outputBatch : argumentClass c.output.slotStride=argumentClass (4*profile.width)
  outputHead : argumentClass c.output.headStride=argumentClass profile.width
  group : argumentClass (group c)=0
  window : argumentClass c.window=0

#specNonCircular CompilerBinding avoiding [FP.Structural.exec,Grid.run,Native.eval]
end VeriTile.Bench.Repairs.TokenAttnMistral.Host
