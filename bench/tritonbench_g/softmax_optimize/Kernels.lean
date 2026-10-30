import VeriTile.Triton.DSL

/-! Separate source transcription of `softmax_optimize_repaired.py`.
The selected source is FP32. Integer channels below retain mathematical signed
values; `Proofs.Native` separately establishes the actual widened machine
address computations from allocation metadata and the explicit PID port.
The two -infinity guards are actual source statements, not numerical premises.
No equivalence between separate multiply/add and a compiled FMA is asserted. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize
open VeriTile.Triton
set_option linter.unusedVariables false

def kernel (output_ptr input_ptr : RegionName) (M N TILE_N : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[input_ptr], outputs=[output_ptr]
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    pid_m = tl.program_id(0).to(tl.int64)
    cols = $((N : Int))
    m = tl.full([$(TILE_N)], value=-float("inf"), dtype=tl.float32)
    z = tl.full([$(TILE_N)], value=0, dtype=tl.float32)
    prev_multiple = ((cols + $((TILE_N : Int)) - $((1 : Int))) // $((TILE_N : Int))) * $((TILE_N : Int)) - $((TILE_N : Int))
    for start_n in range($(0), prev_multiple, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N)).to(tl.int64)
      offset = pid_m * cols + n_offsets
      input_ptrs = input_ptr + offset
      inp = tl.load(input_ptrs).to(tl.float32)
      new_m = tl.maximum(m, inp)
      old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
      new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
      new_z = old_mass + new_mass
      m = new_m
      z = new_z
    }
    for start_n in range(prev_multiple, cols, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N)).to(tl.int64)
      offset = pid_m * cols + n_offsets
      input_ptrs = input_ptr + offset
      mask = n_offsets < cols
      inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
      new_m = tl.maximum(m, inp)
      old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
      new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
      new_z = old_mass + new_mass
      m = new_m
      z = new_z
    }
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m
    prev_multiple = ((cols + $((TILE_N : Int)) - $((1 : Int))) // $((TILE_N : Int))) * $((TILE_N : Int)) - $((TILE_N : Int))
    for start_n in range($(0), prev_multiple, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N)).to(tl.int64)
      offset = pid_m * cols + n_offsets
      input_ptrs = input_ptr + offset
      inp = tl.load(input_ptrs).to(tl.float32)
      e = tl.exp(inp - m)
      out = e / z
      output_ptrs = output_ptr + offset
      tl.store(output_ptrs, out)
    }
    for start_n in range(prev_multiple, cols, $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N)).to(tl.int64)
      offset = pid_m * cols + n_offsets
      input_ptrs = input_ptr + offset
      mask = n_offsets < cols
      inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
      e = tl.exp(inp - m)
      out = e / z
      output_ptrs = output_ptr + offset
      tl.store(output_ptrs, out, mask=mask)
    }
  }

/-- Existing candidate: commute the exact signed integer row product in all
four complete loops. Floating expressions, casts, masks and store order remain. -/
def indexCommutedKernel (Y X : RegionName) (M N B : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    pid_m = tl.program_id(0).to(tl.int64)
    cols = $((N : Int))
    m = tl.full([$(B)], value=-float("inf"), dtype=tl.float32)
    z = tl.full([$(B)], value=0, dtype=tl.float32)
    prev_multiple = ((cols + $((B : Int)) - $((1 : Int))) // $((B : Int))) * $((B : Int)) - $((B : Int))
    for start_n in range($(0), prev_multiple, $(B)) {
      n_offsets = start_n + tl.arange(0, $(B)).to(tl.int64)
      offset = cols * pid_m + n_offsets
      input_ptrs = X + offset
      inp = tl.load(input_ptrs).to(tl.float32)
      new_m = tl.maximum(m, inp)
      old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
      new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
      new_z = old_mass + new_mass
      m = new_m
      z = new_z
    }
    for start_n in range(prev_multiple, cols, $(B)) {
      n_offsets = start_n + tl.arange(0, $(B)).to(tl.int64)
      offset = cols * pid_m + n_offsets
      input_ptrs = X + offset
      mask = n_offsets < cols
      inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
      new_m = tl.maximum(m, inp)
      old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
      new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
      new_z = old_mass + new_mass
      m = new_m
      z = new_z
    }
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m
    prev_multiple = ((cols + $((B : Int)) - $((1 : Int))) // $((B : Int))) * $((B : Int)) - $((B : Int))
    for start_n in range($(0), prev_multiple, $(B)) {
      n_offsets = start_n + tl.arange(0, $(B)).to(tl.int64)
      offset = cols * pid_m + n_offsets
      input_ptrs = X + offset
      inp = tl.load(input_ptrs).to(tl.float32)
      e = tl.exp(inp - m)
      out = e / z
      output_ptrs = Y + offset
      tl.store(output_ptrs, out)
    }
    for start_n in range(prev_multiple, cols, $(B)) {
      n_offsets = start_n + tl.arange(0, $(B)).to(tl.int64)
      offset = cols * pid_m + n_offsets
      input_ptrs = X + offset
      mask = n_offsets < cols
      inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
      e = tl.exp(inp - m)
      out = e / z
      output_ptrs = Y + offset
      tl.store(output_ptrs, out, mask=mask)
    }
  }

end VeriTile.Bench.Repairs.SoftmaxOptimize
