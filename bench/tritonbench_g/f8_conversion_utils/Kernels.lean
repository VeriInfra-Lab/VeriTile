import bench.tritonbench_g.f8_conversion_utils.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature

/-! Complete original Python conversion bodies. Caller metadata selects FP8
format and encoder input storage; decode retains both identical stores. -/
namespace VeriTile.Bench.Repairs.F8ConversionUtils.Body
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource

def decodeSignature (format : Format) : CallSignature where
  pointers := [("Y",.fp16),("X",match format with | .e4 => .f8e4 | .e5 => .f8e5)]
  scalars := [("N",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X"]
  writePorts := ["Y"]

def encodeSignature (input : Input) (format : Format) : CallSignature where
  pointers := [("Y",match format with | .e4 => .f8e4 | .e5 => .f8e5),
    ("X",match input with | .fp16 => .fp16 | .fp32 => .fp32)]
  scalars := [("N",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X"]
  writePorts := ["Y"]

def decodeOriginalKernel (format : Format) (Y X : RegionName) (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (decodeSignature format) call_args [N := N] in triton {
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)
  }

def decodeCandidateKernel (format : Format) (Y X : RegionName) (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (decodeSignature format) call_args [N := N] in triton {
    pid = tl.program_id(0)
    offs = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)
  }

def encodeOriginalKernel (input : Input) (format : Format) (Y X : RegionName) (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (encodeSignature input format) call_args [N := N] in triton {
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
  }

def encodeCandidateKernel (input : Input) (format : Format) (Y X : RegionName) (N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call (encodeSignature input format) call_args [N := N] in triton {
    pid = tl.program_id(0)
    offs = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
  }

/-- Profile adapters select a whole raw body; they generate no statements. -/
def toHalf (format : Format) (swapped : Bool) : sourceType :=
  if swapped then decodeCandidateKernel format else decodeOriginalKernel format

def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  if swapped then encodeCandidateKernel input format else encodeOriginalKernel input format
end VeriTile.Bench.Repairs.F8ConversionUtils.Body

namespace VeriTile.Bench.Repairs.F8ConversionUtils
open VeriTile.Triton
inductive Direction where
  | decode | encode
  deriving DecidableEq
def Direction.doubleStore : Direction → Bool | .decode => Bool.true | .encode => Bool.false
namespace Source
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
abbrev toHalf := Body.toHalf
abbrev toFP8 := Body.toFP8
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
