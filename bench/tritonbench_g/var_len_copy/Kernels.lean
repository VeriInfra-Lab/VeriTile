import VeriTile.Triton.DSL.CallSignature

/-! Repaired segment copy. The Python function bodies are pasted unchanged;
metadata storage types and pointer bindings are outside the quotations. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.Source
open Lean VeriTile.Triton

def blockSize : Nat := 256

def callSignature : CallSignature where
  pointers := [("old_a_start",.i32),("old_a_len",.i32),("old_a_location",.fp32),
    ("new_a_start",.i32),("new_a_location",.fp32)]
  scalars := []
  constexpr := ["BLOCK_SIZE"]

def bodyKernel (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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

def commutedBodyPastedKernel (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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

/-- The public interface keeps the existing port order. Only invocation
bindings and port metadata are supplied here; no source statement is changed. -/
def kernel (commuted : Bool) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) : ComputeKernel :=
  let selected := if commuted then commutedBodyPastedKernel os.cast ol.cast src ns.cast dst B
    else bodyKernel os.cast ol.cast src ns.cast dst B
  .mk [ol.cast,os.cast,ns.cast,src] [dst] selected.surfaceBody

def originalKernel : ComputeKernel :=
  kernel Bool.false ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

def commutedKernel : ComputeKernel :=
  kernel Bool.true ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

end VeriTile.Bench.Repairs.VarLenCopy.Source
