import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Both actual repaired Python function bodies. Metadata is signed int32 and
all six original int64 casts, the complete runtime loop, tail masks and FP32
payload accesses remain. The old typed-region call interface is a thin wrapper. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.Source
open Lean VeriTile.Triton

def blockSize : Nat := 256

def sourceSignature : CallSignature where
  pointers := [("old_a_start",.i32),("old_a_len",.i32),("old_a_location",.fp32),
    ("new_a_start",.i32),("new_a_location",.fp32)]
  scalars := []
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["old_a_len","old_a_start","new_a_start","old_a_location"]
  writePorts := ["new_a_location"]

def source (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_signature sourceSignature in
  triton {
    a_id = tl.cast(tl.program_id(0), tl.int64)
    length = tl.cast(tl.load(old_a_len + a_id), tl.int64)
    old_start = tl.cast(tl.load(old_a_start + a_id), tl.int64)
    new_start = tl.cast(tl.load(new_a_start + a_id), tl.int64)
    old_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    new_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=i + old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=i + new_offset < length)
  }

def commutedSource (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_signature sourceSignature in
  triton {
    a_id = tl.cast(tl.program_id(0), tl.int64)
    length = tl.cast(tl.load(old_a_len + a_id), tl.int64)
    old_start = tl.cast(tl.load(old_a_start + a_id), tl.int64)
    new_start = tl.cast(tl.load(new_a_start + a_id), tl.int64)
    old_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    new_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + i + old_start + old_offset, mask=i + old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=i + new_offset < length)
  }

def kernel (commuted : Bool) : Region .int → Region .int → Region .int →
    RegionName → RegionName → Nat → ComputeKernel :=
  match commuted with
  | Bool.false => fun os ol ns src dst B => source os.cast ol.cast src ns.cast dst B
  | Bool.true => fun os ol ns src dst B => commutedSource os.cast ol.cast src ns.cast dst B

def originalKernel : ComputeKernel :=
  kernel Bool.false ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

def commutedKernel : ComputeKernel :=
  kernel Bool.true ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

end VeriTile.Bench.Repairs.VarLenCopy.Source
