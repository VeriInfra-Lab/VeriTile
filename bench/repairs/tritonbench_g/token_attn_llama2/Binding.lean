import bench.repairs.tritonbench_g.token_attn_llama2.Domain
import bench.repairs.tritonbench_g.token_attn_llama2.Profiles
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole

def argumentClass (n : Nat) : Nat := if n=1 then 1 else if n%16=0 then 16 else 0

def profilePointerClass : Native.Profile → Nat
  | .d16Aligned | .d32Aligned | .d64Aligned | .d128Aligned => 16
  | _ => 0

/-- Exact non-numerical specialization class of every dynamic integer/pointer
argument in the saved real JIT binder. Placement is measured in 32-bit cells.
The scalar host scale is the captured CPython value for the four supported D. -/
structure CompilerBinding (c : Call) (p : Placement c) (profile : Native.Profile) : Prop where
  dimension : c.dim=profile.dim
  pointers : ∀ r, (if (4*p.base r)%16=0 then 16 else 0)=profilePointerClass profile
  maximum : argumentClass c.maxLength=0
  locations : argumentClass c.locBatch=0
  queryBatch : argumentClass c.qBatch=16
  queryHead : argumentClass c.qHead=16
  keyBatch : argumentClass c.kBatch=16
  keyHead : argumentClass c.kHead=16
  outputHead : argumentClass c.outHead=16
  groups : argumentClass (c.heads/c.kvHeads)=0

#specNonCircular CompilerBinding avoiding [VeriTile.Triton.FP.Structural.exec,NativeGrid.runPrograms,Native.eval]
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
