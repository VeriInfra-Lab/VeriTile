import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Contract
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Geometry
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Checked
open VeriTile.Triton FP.Structural Execution
open LlamaFF.Contract
noncomputable section
set_option maxHeartbeats 6000000

def parameters (x w : TensorInfo) : Parameters :=
  ⟨batch x*sequence x,outputs w,columns x,useByte w,(1:ℝ)/1000000⟩

/-- All signed index and formed byte-offset bounds are derived from the
unchanged public checks. No caller-supplied machine bound is added. -/
theorem indices (x w v r : TensorInfo) (ok : Admitted x w v r) (out : Nat) :
    LlamaFF.IndexDomain (shape x w) (LlamaFF.CheckedHost.buffers x w v r out) :=
  LlamaFF.CheckedHost.index_domain ((python_checks_iff x w v r).mp ok) out

theorem input_exact {α : Type} (x w v r : TensorInfo) (ok : Admitted x w v r)
    (s : State α) (live : s.pids 0<Grid.count (parameters x w))
    (t : Nat) (ht : t≤steps (parameters x w)) (i : TileIndex [16,64]) :
    (BitVec.ofNat 64 (aOffset (parameters x w) s t i)).toInt=aOffset (parameters x w) s t i := by
  have hm : 0<(parameters x w).rows := by
    by_contra h
    have hz : (parameters x w).rows=0 := by omega
    rw [Grid.empty _ hz] at live
    omega
  have h := (indices x w v r ok 0).x_exact 0 (inputRow (parameters x w) s i.1) t i.2.1.val
    (by norm_num [shape]) (Nat.mod_lt _ hm) ht i.2.1.isLt
  simpa only [LlamaFF.address64,LlamaFF.CheckedHost.buffers,LlamaFF.View3.offset,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.zero_mul,Nat.mul_zero,Nat.zero_add,Nat.mul_one,
    aOffset,parameters] using h

theorem weight_exact {α : Type} (x w v r : TensorInfo) (ok : Admitted x w v r)
    (s : State α) (t : Nat) (ht : t≤steps (parameters x w)) (i : TileIndex [64,16]) :
    (BitVec.ofNat 64 (wOffset (parameters x w) s t i)).toInt=wOffset (parameters x w) s t i := by
  have valid := (python_checks_iff x w v r).mp ok
  have hn : 0<(parameters x w).columns := valid.shapes.2.1
  have h := (LlamaFF.formed_exact _ _ _ _ _ (indices x w v r ok 0).weightFormed
    0 (inputColumn (parameters x w) s i.2.1) (t*64+i.1.val)
    (by omega) (Nat.mod_lt _ hn).le (LlamaFF.loop_formed (shape x w) t i.1.val ht i.1.isLt)).1
  simpa only [LlamaFF.address64,LlamaFF.weightView,LlamaFF.CheckedHost.buffers,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.zero_mul,Nat.mul_zero,Nat.zero_add,Nat.mul_one,
    wOffset,parameters,Nat.add_comm] using h

#axiomsClean indices
#axiomsClean input_exact
#axiomsClean weight_exact
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Checked
