import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Geometry

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section

def tensorIndex (c : Call) (b head row dim : Nat) : Nat :=
  ((c.start b+row)*c.layout.heads+head)*c.layout.D+dim

theorem output_index (r : Request) (heap : ByteMemory) (h : Contract r heap) (b head row dim : Nat) :
    r.call.output b head row dim=tensorIndex r.call b head row dim := by
  rcases h.checks.2.1.1 with ⟨_,_,_,orow,ohead,odim,_⟩
  simp only [Call.output,tensorIndex,orow,ohead,odim,Nat.mul_one]
  ring

/-- Complete coverage of every valid logical output, including ragged tails. -/
theorem logical_coverage (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (b : Fin r.call.batches) (head : Fin r.call.layout.heads) (row : Nat)
    (hr : row<r.call.length b) (dim : Fin r.call.layout.D) :
    ∃ idx : GridIndex r.call.grid,
      r.call.active idx (r.call.lane row dim) ∧
      flatIndex r.call idx (r.call.lane row dim)=tensorIndex r.call b head row dim ∧
      r.call.offset idx (r.call.lane row dim)=r.call.output b head row dim := by
  obtain ⟨idx,_,on,offset⟩ := Host.covered r.call b head row hr
    (Host.full_coverage r.call b (complete_grid r heap h b b.isLt) row hr) dim
  refine ⟨idx,on,?_,offset⟩
  rw [←(flat_index r.call r.dimensions (extent r) h.checks.2.1 idx _ on).1,
    offset,output_index r heap h]

#axiomsClean logical_coverage
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
