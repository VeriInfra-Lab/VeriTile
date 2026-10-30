import bench.tritonbench_g.rmsnorm_triton.IO

namespace VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
open KernelGrid
noncomputable section

 theorem rows_pos (g : Geometry) (q : Fin (rowCount g)) : 0 < g.rows := by
  have := q.isLt
  unfold rowCount at this
  nlinarith

theorem batch_lt (g : Geometry) (q : Fin (rowCount g)) : batch g q.val < g.batches := by
  exact (Nat.div_lt_iff_lt_mul (rows_pos g q)).mpr q.isLt

theorem row_lt (g : Geometry) (q : Fin (rowCount g)) : row g q.val < g.rows :=
  Nat.mod_lt _ (rows_pos g q)

def index (g : Geometry) (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) :
    Fin (rowCount g) := ⟨b*g.rows+m, by
  have := Nat.mul_le_mul_right g.rows (Nat.succ_le_of_lt hb)
  unfold rowCount
  nlinarith⟩

@[simp] theorem batch_index (g : Geometry) (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) :
    batch g (index g b m hb hm).val = b := by
  have hp : 0<g.rows := Nat.lt_of_le_of_lt (Nat.zero_le m) hm
  change (b*g.rows+m)/g.rows=b
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hm, Nat.zero_add]

@[simp] theorem row_index (g : Geometry) (b m : Nat) (hb : b<g.batches) (hm : m<g.rows) :
    row g (index g b m hb hm).val = m := by
  simp [row,index,Nat.add_mod,Nat.mod_eq_of_lt hm]

@[simp] theorem index_coordinate (g : Geometry) (q : Fin (rowCount g)) :
    index g (batch g q.val) (row g q.val) (batch_lt g q) (row_lt g q) = q := by
  apply Fin.ext
  exact Nat.div_add_mod' q.val g.rows

theorem unique_coordinates (g : Geometry) (unique : OutputUnique g.parameters)
    (q q' : Fin (rowCount g)) (k k' : Fin g.width)
    (same : outputAddress g q.val k.val = outputAddress g q'.val k'.val) :
    q=q' ∧ k=k' := by
  have h := unique _ _ _ _ _ _ (batch_lt g q) (row_lt g q) k.isLt
    (batch_lt g q') (row_lt g q') k'.isLt
    (show outputOffset g.parameters (batch g q.val) (row g q.val) k.val =
      outputOffset g.parameters (batch g q'.val) (row g q'.val) k'.val by
      simpa [outputOffset,Geometry.parameters,outputAddress,Nat.add_assoc] using
        congrArg (g.output.base+·) same)
  refine ⟨?_,Fin.ext h.2.2⟩
  apply Fin.ext
  have a := Nat.div_add_mod' q.val g.rows
  have b := Nat.div_add_mod' q'.val g.rows
  change batch g q.val*g.rows+row g q.val=q.val at a
  change batch g q'.val*g.rows+row g q'.val=q'.val at b
  rw [←a,←b,h.1,h.2.1]

@[simp] theorem launch_pid0 (b m : Nat) (s : BlockState) :
    ((launch b m).initial s).pids 0=b := rfl
@[simp] theorem launch_pid1 (b m : Nat) (s : BlockState) :
    ((launch b m).initial s).pids 1=m := rfl

 theorem member_bounds (g : Geometry) (work : Schedule g.parameters) (b m : Nat)
    (h : (b,m) ∈ work.order) : b<g.batches ∧ m<g.rows :=
  (programGrid_mem g.parameters b m).mp (work.complete.mem_iff.mp h)

end
end VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
