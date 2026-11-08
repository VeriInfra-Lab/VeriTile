import bench.tritonbench_g.masked_select.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature

/-! Literal Python function bodies. Storage profiles and runtime arguments are
supplied by the call site; inclusive prefixes use the actual signed Int64 port.
The retained auxiliary storage profiles share the same selection algorithm. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy (Parameters)

def callSignature (storage : VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.Storage) : CallSignature where
  pointers := [("inp_ptr",match storage with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .i64),
    ("select_mask_ptr",.boolean),
    ("prefix_sum_ptr",.i64),
    ("out_ptr",match storage with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32 | .fp64 => .fp64 | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .i64)]
  scalars := [("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["inp_ptr","select_mask_ptr","prefix_sum_ptr"]
  writePorts := ["out_ptr"]

def sourceForKernel (storage : VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.Storage) (p : Parameters) : ComputeKernel :=
  let inp_ptr := p.input
  let out_ptr := p.output
  let select_mask_ptr := p.selection
  let prefix_sum_ptr := p.prefixSum
  let n_elements := p.size
  let BLOCK_SIZE := p.width
  with_triton_call (callSignature storage) call_args [n_elements := n_elements] in triton {
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))
  }

def kernel (p : Parameters) : ComputeKernel := sourceForKernel .fp32 p

end VeriTile.Bench.Repairs.MaskedSelect
namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open VeriTile.Triton
namespace TypedCopy
open VeriTile.Bench.Repairs.MaskedSelect

abbrev source : Storage → Parameters → ComputeKernel := sourceForKernel
end TypedCopy
namespace Whole
abbrev kernel := TypedCopy.source .fp32
structure Config where
  width : Nat
  warps : Nat
  deriving DecidableEq, Repr

def configs : List Config :=
  [4, 8, 16, 32].flatMap (fun w => [256, 512, 1024, 2048, 4096].map (fun b => ⟨b, w⟩))
def grid (n width : Nat) : Nat := (n + width - 1) / width
end Whole
end VeriTile.Bench.TritonBenchG.MaskedSelect
namespace VeriTile.Bench.Repairs.MaskedSelect
abbrev configs := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.configs
end VeriTile.Bench.Repairs.MaskedSelect
