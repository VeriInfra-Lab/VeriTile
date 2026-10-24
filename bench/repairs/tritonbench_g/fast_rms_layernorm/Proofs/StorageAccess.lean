import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.Storage
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightFlat
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealRows

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.StorageAccess
open VeriTile.Triton Storage
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
noncomputable section

def flat {h : Call} {p : Phase} (s : Placement h p) : FlatAlloc where
  flat := "physical-slots"
  regions := buffers p
  base := fun r => if hr : r∈buffers p then (s.allocation r hr).buffer.base/4 else 0
  extent := fun r => if hr : r∈buffers p then (s.allocation r hr).buffer.extent else 0

theorem flat_closed {h : Call} {p : Phase} (s : Placement h p) (r : RegionName)
    (outside : r∉(flat s).regions) : (flat s).extent r=0 := by
  simp only [flat] at *
  simp [outside]

theorem flat_disjoint {h : Call} {p : Phase} (s : Placement h p) : (flat s).Disjoint := by
  intro r hr q hq ne
  simp only [flat] at hr hq ⊢
  simp only [dif_pos hr,dif_pos hq]
  rcases s.separate r hr q hq ne with sep|sep
  · left; omega
  · right; omega

theorem capacity {h : Call} {p : Phase} (s : Placement h p) (r : RegionName)
    (hr : r∈buffers p) : cells h r≤(flat s).extent r := by
  simpa [flat,hr] using (s.allocation r hr).holds

theorem physical_slot {h : Call} {p : Phase} (s : Placement h p) (r : RegionName)
    (hr : r∈buffers p) (o : Nat) (byte : Fin 4) :
    address s r hr o byte=(flat s).addr r o*4+(s.allocation r hr).buffer.base%4+byte.val := by
  simp only [address,FlatAlloc.addr,flat,dif_pos hr]
  omega

theorem row_access {h : Call} {p : Phase} (s : Placement h p) (r : RegionName)
    (hr : r∈buffers p) (shape : cells h r=h.cells) (row col : Nat)
    (hi : row<h.rows) (hj : col<h.cols) : row*h.cols+col<(flat s).extent r := by
  have bound:=Native.offset_bound h row col hi hj
  have cap:=capacity s r hr
  rw [shape] at cap
  exact bound.trans_le cap

theorem forward_safe (h : Call) (s : Placement h .forward) (u : BlockState) (pid : u.pids 0<h.rows) :
    Kernel.TraceSafe (flat s).extent (Kernels.forward h).eraseDType.toAlgKernel u := by
  apply ForwardRun.trace_safe h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps
  · intro j hj
    exact row_access s "x" (by simp [buffers]) (by simp [cells]) _ _ pid hj
  · intro j hj
    have cap : h.cols≤(flat s).extent "w" := by simpa [cells] using capacity s "w" (by simp [buffers])
    cases h.gemma <;> simpa [ForwardRun.weightOffset] using hj.trans_le cap
  · intro j hj
    exact row_access s "y" (by simp [buffers]) (by simp [cells]) _ _ pid hj
  · have cap : h.rows≤(flat s).extent "r" := by simpa [cells] using capacity s "r" (by simp [buffers])
    simpa using pid.trans_le cap

theorem backward_safe (h : Call) (s : Placement h .backward) (u : BlockState) (pid : u.pids 0<h.rows) :
    Kernel.TraceSafe (flat s).extent (Kernels.backward h).eraseDType.toAlgKernel u := by
  apply BackwardRun.trace_safe h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps
  · intro j hj
    exact row_access s "dx" (by simp [buffers]) (by simp [cells]) _ _ pid hj
  · intro j hj
    exact row_access s "x" (by simp [buffers]) (by simp [cells]) _ _ pid hj
  · intro j hj
    have cap : h.cols≤(flat s).extent "w" := by simpa [cells] using capacity s "w" (by simp [buffers])
    exact hj.trans_le cap
  · have cap : h.rows≤(flat s).extent "r" := by simpa [cells] using capacity s "r" (by simp [buffers])
    simpa using pid.trans_le cap

theorem weight_bounds (h : Call) (s : Placement h .backward) (pid : Nat) :
    RealWeightSafety.Bounds h 256 pid (flat s).extent := by
  constructor
  · intro row hr j hj
    exact row_access s "x" (by simp [buffers]) (by simp [cells]) _ _ hr hj
  · intro row hr j hj
    exact row_access s "dx" (by simp [buffers]) (by simp [cells]) _ _ hr hj
  · intro row hr
    have cap : h.rows≤(flat s).extent "r" := by simpa [cells] using capacity s "r" (by simp [buffers])
    exact hr.trans_le cap
  · intro j hj
    have cap : h.cols≤(flat s).extent "dw" := by simpa [cells] using capacity s "dw" (by simp [buffers])
    exact hj.trans_le cap

/-- All three flat source executions are derived. No source-success premise. -/
theorem forward_flat (h : Call) (s : Placement h .forward) (u : BlockState)
    (pid : u.pids 0<h.rows) (undef : u.undef=fun _ _ => 0) :
    ∃ v,exec ((flat s).flattenKernel (Kernels.forward h).eraseDType.toAlgKernel) ((flat s).flattenState u)=
      some ((flat s).flattenState v) ∧ exec (Kernels.forward h).eraseDType.toAlgKernel u=some v := by
  obtain ⟨v,he,_⟩:=RealRows.forward_complete h u
  refine ⟨v,?_,he⟩
  rw [(flat s).exec_flatten (flat_disjoint s) (flat_closed s) _ u (forward_safe h s u pid)
    (ForwardRun.flatten_ok h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps) undef,he,Option.map_some]

theorem backward_flat (h : Call) (s : Placement h .backward) (u : BlockState)
    (pid : u.pids 0<h.rows) (undef : u.undef=fun _ _ => 0) :
    ∃ v,exec ((flat s).flattenKernel (Kernels.backward h).eraseDType.toAlgKernel) ((flat s).flattenState u)=
      some ((flat s).flattenState v) ∧ exec (Kernels.backward h).eraseDType.toAlgKernel u=some v := by
  obtain ⟨v,he,_⟩:=RealRows.backward_values h u
  refine ⟨v,?_,he⟩
  rw [(flat s).exec_flatten (flat_disjoint s) (flat_closed s) _ u (backward_safe h s u pid)
    (BackwardRun.flatten_ok h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps) undef,he,Option.map_some]

theorem weight_flat (h : Call) (s : Placement h .backward) (u : BlockState)
    (undef : u.undef=fun _ _ => 0) :
    ∃ v,exec ((flat s).flattenKernel (RealWeightProgram.source h 256)) ((flat s).flattenState u)=
      some ((flat s).flattenState v) ∧ exec (RealWeightProgram.source h 256) u=some v := by
  obtain ⟨v,he,_⟩:=RealWeightMemory.complete h 256 u
  refine ⟨v,?_,he⟩
  rw [(flat s).exec_flatten (flat_disjoint s) (flat_closed s) _ u
    (RealWeightTrace.complete h 256 _ u (weight_bounds h s _)) (RealWeightTrace.flatten_ok h 256) undef,he,Option.map_some]

#axiomsClean flat_disjoint
#axiomsClean forward_safe
#axiomsClean backward_safe
#axiomsClean weight_bounds
#axiomsClean forward_flat
#axiomsClean backward_flat
#axiomsClean weight_flat
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.StorageAccess
