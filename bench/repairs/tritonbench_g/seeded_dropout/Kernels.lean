import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Complete unchanged original and pre-existing candidate Python bodies.
The three Python-int JIT ABIs are supplied solely by call signatures.
No seed/block assignments or explicit store cast are added to the bodies. -/
namespace VeriTile.Bench.Repairs.SeededDropout
open Lean VeriTile.Triton

inductive SeedKind where
  | i32 | i64 | u64
  deriving DecidableEq

def SeedCarrier : SeedKind → Type
  | .i32 | .i64 => Int
  | .u64 => Nat

def seedValue : (kind : SeedKind) → SeedCarrier kind → Int
  | .i32, s | .i64, s => s
  | .u64, s => Int.ofNat s


def signatureI32 : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32),("p",.fp32),("seed",.i32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr"]
  writePorts := ["output_ptr"]

def sourceI32 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Int) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureI32 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def candidateI32 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Int) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureI32 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def signatureI64 : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32),("p",.fp32),("seed",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr"]
  writePorts := ["output_ptr"]

def sourceI64 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Int) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureI64 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def candidateI64 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Int) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureI64 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def signatureU64 : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32),("p",.fp32),("seed",.u64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr"]
  writePorts := ["output_ptr"]

def sourceU64 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Nat) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureU64 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def candidateU64 (x_ptr output_ptr : RegionName) (n : Nat) (probability : ℝ) (seedValue : Nat) : ComputeKernel :=
  let BLOCK_SIZE := (1024 : Nat)
  with_triton_call signatureU64 call_args [n_elements := n,p := probability,seed := seedValue] in
  triton {
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def kernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => sourceI32
  | .i64 => sourceI64
  | .u64 => sourceU64

def indexCommutedKernel (kind : SeedKind) : RegionName → RegionName → Nat → ℝ → SeedCarrier kind → ComputeKernel :=
  match kind with
  | .i32 => candidateI32
  | .i64 => candidateI64
  | .u64 => candidateU64
def originalKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  kernel kind "x_ptr" "output_ptr" n p seed

def candidateKernel (kind : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier kind) : ComputeKernel :=
  indexCommutedKernel kind "x_ptr" "output_ptr" n p seed

def gridCount (n : Nat) : Nat := (n+1023)/1024
end VeriTile.Bench.Repairs.SeededDropout
