import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Source.BlockPaddingView

/-! Proof-only views derived from the current repair and current candidates.
These retain source precision and change only block-load padding for the existing
zero-oracle mathematical lemmas. Proofs/SourceBridge checks each whole program
and discharges this padding change before using its execution result. The
canonical sources, including their explicit zero padding, stay in Kernels.lean.
There are no independently maintained source quotations in this module. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16


def Storage.current : Storage → VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput.Storage
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInter
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def source (query key gate : Storage) (swapped : Bool)
    (out_dtype : FloatDType)
    (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter.source
      query.current key.current gate.current swapped out_dtype q k g A s_k_h s_k_t scale T K BT BC BK NC)

def original (query key gate : Storage) := source query key gate Bool.false
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInter

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntra
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def source (query key gate : Storage) (swapped : Bool)
    (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra.source
      query.current key.current gate.current swapped q k g A s_k_h s_k_t scale T K BT BC BK)

def original (query key gate : Storage) := source query key gate Bool.false
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntra

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit
open VeriTile.Triton
abbrev Storage := TypedOutput.Storage
def source (query key gate : Storage) (swapped : Bool)
    (q k g A : RegionName) (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit.source
      query.current key.current gate.current swapped q k g A s_k_h s_k_t scale T K BT BC BK NC)

def original (query key gate : Storage) := source query key gate Bool.false
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open VeriTile.Triton
def source (query key gate : Storage) (swapped : Bool)
    (dtype out_dtype : FloatDType)
    (q v g h o A : RegionName) (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput.source
      query.current key.current gate.current swapped dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV)

def original (query key gate : Storage) := source query key gate Bool.false
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
open VeriTile.Triton

def original (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge.original out_dtype A A2 T BT BC NK)

def loadFirst (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : ComputeKernel :=
  BlockPaddingView.kernel .unspecified
    (VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge.loadFirst out_dtype A A2 T BT BC NK)
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
