import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource
open VeriTile.Triton
noncomputable section
/-- Payload storage of the existing source family; dot precision follows the
host priority bf16, then fp16, then fp32. -/
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

structure Tuning where
  BM : Nat
  BN : Nat
  BCS : Nat
  stages : Nat
  warps : Nat
  deriving DecidableEq

def tuning (i : Fin 9) : Tuning :=
  match i.val with
  | 0 => ⟨128, 256, 64, 3, 8⟩
  | 1 => ⟨64, 256, 32, 4, 4⟩
  | 2 => ⟨128, 128, 32, 4, 4⟩
  | 3 => ⟨128, 64, 32, 4, 4⟩
  | 4 => ⟨64, 128, 32, 4, 4⟩
  | 5 => ⟨128, 32, 32, 4, 4⟩
  | 6 => ⟨64, 32, 32, 5, 2⟩
  | 7 => ⟨32, 64, 32, 5, 2⟩
  | _ => ⟨64, 64, 32, 4, 2⟩

structure TensorStrides where
  batch : Nat
  sequence : Nat
  head : Nat
  feature : Nat

structure DoutStrides where
  batch : Nat
  chunk : Nat
  head : Nat
  row : Nat
  column : Nat

/-- Actual queried host strides. The retained source layout branches are
inactive under admission. Supplied output is direct; no-residual strides are
selected as zero exactly as in the wrapper. -/
structure HostCall where
  A : RegionName
  Dout : RegionName
  Out : RegionName
  Residual : RegionName
  batch : Nat
  seqlen : Nat
  K : Nat
  nchunks : Nat
  chunk : Nat
  groups : Nat
  hasGroups : Bool
  hasResidual : Bool
  aStrides : TensorStrides
  doutStrides : DoutStrides
  outStrides : TensorStrides
  residualStrides : TensorStrides
  aStorage : Storage := .fp32
  doutStorage : Storage := .fp32
  residualDType : FloatDType := .fp32
  outputDType : FloatDType := .fp32

def HostCall.groupCount (h : HostCall) := if h.hasGroups then h.groups else 1

def HostCall.tensorShape (h : HostCall) : List Nat :=
  if h.hasGroups then [h.batch, h.seqlen, h.groups, h.K] else [h.batch, h.seqlen, h.K]

def HostCall.doutShape (h : HostCall) : List Nat :=
  if h.hasGroups then [h.batch, h.nchunks, h.groups, h.chunk, h.chunk]
  else [h.batch, h.nchunks, h.chunk, h.chunk]

def HostCall.headStride (h : HostCall) (stride : Nat) := if h.hasGroups then stride else 0

def HostCall.resStride (h : HostCall) (stride : Nat) := if h.hasResidual then stride else 0

/-- `out is None` selects empty_like(A); otherwise the returned descriptor is
exactly the supplied out. Its memory is never assumed initialized to zero. -/
def HostCall.returnRegion (h : HostCall) := h.Out

end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource
