import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Host
namespace VeriTile.Bench.Repairs.BlockSparseAttn
set_option maxHeartbeats 10000000

theorem channel_bound (s : Shape) (block lane : Nat) (hb : block<s.blocks) (hl : lane<s.blockD) :
    block*s.blockD+lane<s.dim := by
  have hp:=Nat.mul_le_mul_right s.blockD (show block+1≤s.blocks by omega)
  unfold Shape.dim
  nlinarith

theorem csr_first_pointer (s : Shape) (c : CSR) (l : Lane) (hl : l.Valid s) :
    c.first s (laneWork s l).head (laneWork s l).row=
      c.rows (((l.headProgram%s.heads%s.layouts)*c.rowStride+l.rowProgram : Nat):Int) := by
  have he:=(decode s.blockM l.rowProgram l.rowLane hl.2.2.2.1).1
  simp only [CSR.first,laneWork,he]
theorem csr_last_pointer (s : Shape) (c : CSR) (l : Lane) (hl : l.Valid s) :
    c.last s (laneWork s l).head (laneWork s l).row=
      c.rows (((l.headProgram%s.heads%s.layouts)*c.rowStride+l.rowProgram+1 : Nat):Int) := by
  have he:=(decode s.blockM l.rowProgram l.rowLane hl.2.2.2.1).1
  simp only [CSR.last,laneWork,he]

theorem query_grid_i32 (a : Host.Arguments) (valid : Host.Contract a) (l : Lane)
    (hl : l.Valid a.layout.shape) (db d : Nat) (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD) :
    address a.layout.q (laneWork a.layout.shape l).batch (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row (db*a.layout.shape.blockD+d)<2^31 := by
  have hh : 0<a.layout.shape.heads := valid.shape.2.1
  have hb : (laneWork a.layout.shape l).batch<a.q.batches := by
    rw [valid.qshape.1]
    exact (Nat.div_lt_iff_lt_mul hh).mpr hl.2.1
  have hh' : (laneWork a.layout.shape l).head<a.q.heads := by
    rw [valid.qshape.2.1]
    exact Nat.mod_lt _ hh
  have hr:=Host.row_envelope a.layout.shape l hl
  have hq:=valid.qshape.2.2.1
  have hn : (laneWork a.layout.shape l).row≤a.q.rows-1+(a.layout.shape.blockM+a.layout.shape.blockN) := by
    have hp:=valid.qguard.positive.2.2.1
    change l.rowProgram*a.layout.shape.blockM+l.rowLane≤_
    omega
  have hc : db*a.layout.shape.blockD+d<a.q.dim := by
    rw [valid.qshape.2.2.2]
    exact channel_bound a.layout.shape db d hdb hd
  rw [valid.strides.1]
  exact Host.tensor_formed a.q a.layout.shape valid.qguard _ _ _ _ hb hh' hn hc

theorem output_grid_i32 (a : Host.Arguments) (valid : Host.Contract a) (l : Lane)
    (hl : l.Valid a.layout.shape) : (laneWork a.layout.shape l).target a.layout.shape<2^31 := by
  let s:=a.layout.shape
  have hr:=Host.row_envelope s l hl
  have hc:=channel_bound s l.channelProgram l.colLane hl.2.2.1 hl.2.2.2.2
  have hhead : l.headProgram<s.batches*s.heads := hl.2.1
  have hp:=Nat.mul_le_mul_right s.length (show l.headProgram+1≤s.batches*s.heads by omega)
  have hn : l.headProgram*s.length+(l.rowProgram*s.blockM+l.rowLane)<s.batches*s.heads*s.length+s.blockM := by
    nlinarith
  have he:=Nat.div_add_mod l.headProgram s.heads
  have target : (laneWork s l).target s=
      (l.headProgram*s.length+(l.rowProgram*s.blockM+l.rowLane))*s.dim+
        (l.channelProgram*s.blockD+l.colLane) := by
    unfold laneWork Work.target outputAddress
    rw [show l.headProgram/s.heads*s.heads+l.headProgram%s.heads=l.headProgram by simpa [Nat.mul_comm] using he]
  have hd:=Nat.mul_le_mul_right s.dim
    (show l.headProgram*s.length+(l.rowProgram*s.blockM+l.rowLane)+1≤s.batches*s.heads*s.length+s.blockM by omega)
  have hi:=valid.outputFormed
  change (s.batches*s.heads*s.length+s.blockM)*s.dim<2^31 at hi
  change (laneWork s l).target s<2^31
  rw [target]
  nlinarith

#axiomsClean channel_bound
#axiomsClean csr_first_pointer
#axiomsClean csr_last_pointer
#axiomsClean query_grid_i32
#axiomsClean output_grid_i32
end VeriTile.Bench.Repairs.BlockSparseAttn
