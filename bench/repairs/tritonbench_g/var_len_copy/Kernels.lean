import VeriTile.Triton.DSL

/-! Canonical repaired source. FP32 payloads are copied without arithmetic;
metadata storage is signed int32, explicitly widened to signed int64 before
loop, mask and address arithmetic. The candidate commutes the two scalar
source address offsets, retaining the repaired tail masks. -/
namespace VeriTile.Bench.Repairs.VarLenCopy.Source
open Lean VeriTile.Triton

def blockSize : Nat := 256

/-- Complete ordinary quotations for the retained original and commuted
signed address forms; all metadata and original tail masks are explicit. -/
def kernel (commuted : Bool) : Region .int → Region .int → Region .int →
    RegionName → RegionName → Nat → ComputeKernel :=
  match commuted with
  | Bool.false => fun os ol ns src dst B =>
    triton {
      tl.ports inputs=[ol.cast, os.cast, ns.cast, src], outputs=[dst]
      tl.region os = tl.int32, ol = tl.int32, ns = tl.int32, src = tl.float32, dst = tl.float32
      a_id = tl.program_id(0).to(tl.int64)
      length = tl.load(ol + a_id).to(tl.int64)
      old_start = tl.load(os + a_id).to(tl.int64)
      new_start = tl.load(ns + a_id).to(tl.int64)
      old_offset = tl.arange(0, $(B)).to(tl.int64)
      new_offset = tl.arange(0, $(B)).to(tl.int64)
      for i in range($(0), length, $(B)) {
        v = tl.load(src + old_start + i + old_offset, mask=i + old_offset < length)
        tl.store(dst + new_start + i + new_offset, v, mask=i + new_offset < length)
      }
    }
  | Bool.true => fun os ol ns src dst B =>
    triton {
      tl.ports inputs=[ol.cast, os.cast, ns.cast, src], outputs=[dst]
      tl.region os = tl.int32, ol = tl.int32, ns = tl.int32, src = tl.float32, dst = tl.float32
      a_id = tl.program_id(0).to(tl.int64)
      length = tl.load(ol + a_id).to(tl.int64)
      old_start = tl.load(os + a_id).to(tl.int64)
      new_start = tl.load(ns + a_id).to(tl.int64)
      old_offset = tl.arange(0, $(B)).to(tl.int64)
      new_offset = tl.arange(0, $(B)).to(tl.int64)
      for i in range($(0), length, $(B)) {
        v = tl.load(src + i + old_start + old_offset, mask=i + old_offset < length)
        tl.store(dst + new_start + i + new_offset, v, mask=i + new_offset < length)
      }
    }

def originalKernel : ComputeKernel :=
  kernel Bool.false ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

def commutedKernel : ComputeKernel :=
  kernel Bool.true ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" blockSize

end VeriTile.Bench.Repairs.VarLenCopy.Source
