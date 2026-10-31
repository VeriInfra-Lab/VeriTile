import bench.tritonbench_g.dropout_triton.Sources.Parameters
/-! Shared repaired dropout source, specialized by data and keep-buffer dtype.
Each branch retains identical eager arithmetic, omitted load padding and stores.
The existing candidate commutes only the integer block-start product. -/
namespace VeriTile.Bench.Repairs.DropoutTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n+1023)/1024

def sourceFamily (storage : Storage) (keep : KeepStorage) (swapped : Bool)
    (x_ptr x_keep_ptr output_ptr : RegionName) (n_elements BLOCK_SIZE : Nat) (p : ℝ) : ComputeKernel :=
  match storage,keep,swapped with
  | .fp32, .int32, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, x_keep_ptr = tl.int32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
  }
  | .fp32, .int32, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, x_keep_ptr = tl.int32, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
  }
  | .fp32, .bool, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, x_keep_ptr = tl.int1, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
  }
  | .fp32, .bool, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float32, x_keep_ptr = tl.int1, output_ptr = tl.float32
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float32), mask=mask)
  }
  | .fp16, .int32, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, x_keep_ptr = tl.int32, output_ptr = tl.float16
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .int32, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, x_keep_ptr = tl.int32, output_ptr = tl.float16
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .bool, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, x_keep_ptr = tl.int1, output_ptr = tl.float16
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .fp16, .bool, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.float16, x_keep_ptr = tl.int1, output_ptr = tl.float16
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.float16), mask=mask)
  }
  | .bf16, .int32, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, x_keep_ptr = tl.int32, output_ptr = tl.bfloat16
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .int32, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, x_keep_ptr = tl.int32, output_ptr = tl.bfloat16
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .bool, Bool.false => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, x_keep_ptr = tl.int1, output_ptr = tl.bfloat16
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }
  | .bf16, .bool, Bool.true => triton {
    tl.ports inputs=[x_ptr,x_keep_ptr], outputs=[output_ptr]
    tl.region x_ptr = tl.bfloat16, x_keep_ptr = tl.int1, output_ptr = tl.bfloat16
    pid = tl.program_id(axis=0)
    block_start = $(BLOCK_SIZE) * pid
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - $(p)), 0.0)
    tl.store(output_ptr + offsets, (output).to(tl.bfloat16), mask=mask)
  }

def sourceKernel := sourceFamily .fp32 .int32 Bool.false
def commutedSourceKernel := sourceFamily .fp32 .int32 Bool.true

def kernel (n : Nat) (p : ℝ) : ComputeKernel := sourceKernel "x" "keep" "out" n blockSize p
def indexCommutedKernel (n : Nat) (p : ℝ) : ComputeKernel := commutedSourceKernel "x" "keep" "out" n blockSize p
end VeriTile.Bench.Repairs.DropoutTriton
