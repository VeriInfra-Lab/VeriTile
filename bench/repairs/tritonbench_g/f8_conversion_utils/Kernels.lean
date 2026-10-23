import bench.tritonbench_g.f8_conversion_utils.Kernels

namespace VeriTile.Bench.Repairs.F8ConversionUtils
open VeriTile.Triton
inductive Direction where
  | decode | encode
  deriving DecidableEq
def Direction.doubleStore : Direction → Bool | .decode => Bool.true | .encode => Bool.false
namespace Source
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
abbrev toHalf := VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toHalf
abbrev toFP8 := VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toFP8
inductive Profile where
  | decodeE5 | encodeHalfE5 | encodeSingleE5 | decodeE4 | encodeHalfE4 | encodeSingleE4
  deriving DecidableEq

def Profile.input : Profile → FloatDType
  | .decodeE5 => .f8e5 | .decodeE4 => .f8e4
  | .encodeHalfE5 | .encodeHalfE4 => .fp16 | .encodeSingleE5 | .encodeSingleE4 => .real
def Profile.output : Profile → FloatDType
  | .decodeE5 | .decodeE4 => .fp16
  | .encodeHalfE5 | .encodeSingleE5 => .f8e5
  | .encodeHalfE4 | .encodeSingleE4 => .f8e4
def Profile.twice : Profile → Bool
  | .decodeE5 | .decodeE4 => Bool.true | _ => Bool.false

def kernel (p : Profile) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  match p with
  | .decodeE5 => toHalf .e5 Bool.false Y X N B
  | .encodeHalfE5 => toFP8 .fp16 .e5 Bool.false Y X N B
  | .encodeSingleE5 => toFP8 .fp32 .e5 Bool.false Y X N B
  | .decodeE4 => toHalf .e4 Bool.false Y X N B
  | .encodeHalfE4 => toFP8 .fp16 .e4 Bool.false Y X N B
  | .encodeSingleE4 => toFP8 .fp32 .e4 Bool.false Y X N B

/-- Existing source candidate: commute only the natural tile-index product.
Both decode stores and every source conversion remain unchanged. -/
def indexCommuted (p : Profile) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  match p with
  | .decodeE5 => toHalf .e5 Bool.true Y X N B
  | .encodeHalfE5 => toFP8 .fp16 .e5 Bool.true Y X N B
  | .encodeSingleE5 => toFP8 .fp32 .e5 Bool.true Y X N B
  | .decodeE4 => toHalf .e4 Bool.true Y X N B
  | .encodeHalfE4 => toFP8 .fp16 .e4 Bool.true Y X N B
  | .encodeSingleE4 => toFP8 .fp32 .e4 Bool.true Y X N B

def Profile.inputWidth : Profile → Nat | .decodeE5 | .decodeE4 => 1 | .encodeHalfE5 | .encodeHalfE4 => 2 | .encodeSingleE5 | .encodeSingleE4 => 4
def Profile.outputWidth : Profile → Nat | .decodeE5 | .decodeE4 => 2 | _ => 1

def Profile.direction : Profile → Direction
  | .decodeE5 | .decodeE4 => .decode | _ => .encode
def Profile.encoder : Profile → Input
  | .encodeSingleE5 | .encodeSingleE4 => .fp32 | _ => .fp16
def Profile.format : Profile → Format
  | .decodeE4 | .encodeHalfE4 | .encodeSingleE4 => .e4 | _ => .e5
end Source
namespace Chunking
def chunkSize : Nat := 2^30
def chunkCount (n : Nat) : Nat := (n+chunkSize-1)/chunkSize
def chunkStart (k : Nat) : Nat := k*chunkSize
def chunkLength (n k : Nat) : Nat := min chunkSize (n-chunkStart k)
end Chunking
/-- Empty tensors return their allocation without a kernel launch. -/
def launchChunks (n : Nat) : List Nat := List.range (Chunking.chunkCount n)
end VeriTile.Bench.Repairs.F8ConversionUtils
