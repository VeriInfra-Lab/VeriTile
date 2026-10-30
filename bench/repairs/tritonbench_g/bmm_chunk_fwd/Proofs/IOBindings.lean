import bench.repairs.tritonbench_g.bmm_chunk_fwd.IO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section
open Classical
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem data_bound (p : Parameters) (b r g k : Nat)
    (hb : b<p.shape.batch) (hr : r<p.shape.seqlen) (hg : g<p.shape.groups) (hk : k<p.shape.K) :
    Host.inputOffset (Host.prepared p.shape (Host.contiguous p.shape)) b r g k<dataExtent p := by
  let v := Host.prepared p.shape (Host.contiguous p.shape)
  have bnd0 := Nat.mul_le_mul_right v.batch (Nat.le_of_lt hb)
  have bnd1 := Nat.mul_le_mul_right v.row (Nat.le_of_lt hr)
  have bnd2 := Nat.mul_le_mul_right v.head (Nat.le_of_lt hg)
  have bnd3 := Nat.mul_le_mul_right v.column (Nat.le_of_lt hk)
  change b*v.batch+r*v.row+g*v.head+k*v.column<
    p.shape.batch*v.batch+p.shape.seqlen*v.row+p.shape.groups*v.head+p.shape.K*v.column+1
  omega

theorem sequence_bound (p : Parameters) (b r : Nat) (hb : b<p.shape.batch) (hr : r<p.shape.seqlen) :
    b*p.seqBatch+r*p.seqRow<seqExtent p := by
  have bnd0 := Nat.mul_le_mul_right p.seqBatch (Nat.le_of_lt hb)
  have bnd1 := Nat.mul_le_mul_right p.seqRow (Nat.le_of_lt hr)
  unfold seqExtent
  omega

theorem groups_positive (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c) :
    0<p.shape.groups := by
  have bound := Host.p2_lt pid
  change Host.p2 pid<p.shape.chunks*p.shape.groups at bound
  nlinarith

theorem a_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (i : TileIndex [c.M,p.shape.K])
    (active : (views p c).aMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i) :
    (views p c).aWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i<dataExtent p := by
  apply lt_of_eq_of_lt (Host.a_window (parameters p) c (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i.1 i.2.1)
  exact data_bound p _ _ _ _ (Host.p1_lt pid)
    (Host.local_to_sequence (parameters p) _ _ active.2)
    (Nat.mod_lt _ (groups_positive p c pid)) i.2.1.isLt

theorem b_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (i : TileIndex [p.shape.K,c.N])
    (active : (views p c).bMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i) :
    (views p c).bWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i<dataExtent p := by
  apply lt_of_eq_of_lt (Host.b_window (parameters p) c (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i.2.1 i.1)
  exact data_bound p _ _ _ _ (Host.p1_lt pid)
    (Host.local_to_sequence (parameters p) _ _ active.2)
    (Nat.mod_lt _ (groups_positive p c pid)) i.1.isLt

theorem row_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (hasSeq : p.hasSeq=Bool.true) (i : TileIndex [c.M])
    (active : (views p c).rowIDMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i) :
    (views p c).rowIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i<seqExtent p := by
  change (Host.p1 pid*(parameters p).parameters.sBatch+
    RealViews.chunk (parameters p).parameters (Host.p2 pid)*(parameters p).parameters.chunk*(parameters p).parameters.sRow)+
      RealViews.row c.M c.N (parameters p).parameters (Host.p0 pid) i.1*(parameters p).parameters.sRow<_
  rw [Host.sequence_window (parameters p) hasSeq (Host.p1 pid) (Host.p2 pid)]
  exact sequence_bound p _ _ (Host.p1_lt pid) (Host.local_to_sequence (parameters p) _ _ active.2)

theorem column_bound (p : Parameters) (c : Host.Config) (pid : Host.PID (parameters p) c)
    (hasSeq : p.hasSeq=Bool.true) (i : TileIndex [c.N])
    (active : (views p c).columnIDMask (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i) :
    (views p c).columnIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) i<seqExtent p := by
  change (Host.p1 pid*(parameters p).parameters.sBatch+
    RealViews.chunk (parameters p).parameters (Host.p2 pid)*(parameters p).parameters.chunk*(parameters p).parameters.sRow)+
      RealViews.column c.N (parameters p).parameters (Host.p0 pid) i.1*(parameters p).parameters.sRow<_
  rw [Host.sequence_window (parameters p) hasSeq (Host.p1 pid) (Host.p2 pid)]
  exact sequence_bound p _ _ (Host.p1_lt pid) (Host.local_to_sequence (parameters p) _ _ active.2)

/-- Public masked channels supply exactly the former raw source pins. -/
theorem input_binding (p : Parameters) (c : Host.Config) (v : RealInputs p) (s : BlockState)
    (nonempty : p.shape.outputSize≠0) (binding : (realIO p c).InputBinding s v) :
    Host.Pinned (parameters p) c (logicalInputs p v) s := by
  intro pid
  constructor
  · intro i k active
    let off := (views p c).aWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) (i,k,PUnit.unit)
    have bound := a_bound p c pid (i,k,PUnit.unit) active
    have pin := binding (0 : Fin 3) ⟨off,bound⟩ ⟨nonempty,pid,(i,k,PUnit.unit),active,rfl⟩
    change s.readMem p.A off=v 0 ⟨off,bound⟩ at pin
    change s.readMem p.A off=array p v 0 _
    have address : off=Host.inputOffset (Host.prepared p.shape (Host.contiguous p.shape))
        (Host.p1 pid) (Host.p2 pid/p.shape.groups*p.shape.chunk+
          RealViews.row c.M c.N (parameters p).parameters (Host.p0 pid) i) (Host.p2 pid%p.shape.groups) k :=
      Host.a_window (parameters p) c _ _ _ i k
    exact pin.trans ((by unfold array; rw [dif_pos (show off<dataExtent p from bound)]; rfl : _=array p v 0 off).trans
      (congrArg (array p v 0) address))
  · intro k j active
    let off := (views p c).bWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) (k,j,PUnit.unit)
    have bound := b_bound p c pid (k,j,PUnit.unit) active
    have pin := binding (1 : Fin 3) ⟨off,bound⟩ ⟨nonempty,pid,(k,j,PUnit.unit),active,rfl⟩
    change s.readMem p.B off=v 1 ⟨off,bound⟩ at pin
    change s.readMem p.B off=array p v 1 _
    have address : off=Host.inputOffset (Host.prepared p.shape (Host.contiguous p.shape))
        (Host.p1 pid) (Host.p2 pid/p.shape.groups*p.shape.chunk+
          RealViews.column c.N (parameters p).parameters (Host.p0 pid) j) (Host.p2 pid%p.shape.groups) k :=
      Host.b_window (parameters p) c _ _ _ j k
    exact pin.trans ((by unfold array; rw [dif_pos (show off<dataExtent p from bound)]; rfl : _=array p v 1 off).trans
      (congrArg (array p v 1) address))
  · intro hasSeq i active
    let off := (views p c).rowIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) (i,PUnit.unit)
    have bound := row_bound p c pid hasSeq (i,PUnit.unit) active
    have pin := binding (2 : Fin 3) ⟨off,bound⟩ ⟨nonempty,hasSeq,Or.inl ⟨pid,(i,PUnit.unit),active,rfl⟩⟩
    change s.readMemValue .int (Region.cast p.Seq) off=v 2 ⟨off,bound⟩ at pin
    change s.readMemValue .int (Region.cast p.Seq) off=sequence p v _
    have address : off=Host.p1 pid*p.seqBatch+(Host.p2 pid/p.shape.groups*p.shape.chunk+
        RealViews.row c.M c.N (parameters p).parameters (Host.p0 pid) i)*p.seqRow :=
      Host.sequence_window (parameters p) hasSeq _ _ _
    exact pin.trans ((by simp [sequence,off,bound] : _=sequence p v off).trans
      (congrArg (sequence p v) address))
  · intro hasSeq j active
    let off := (views p c).columnIDWindow (Host.p0 pid) (Host.p1 pid) (Host.p2 pid) (j,PUnit.unit)
    have bound := column_bound p c pid hasSeq (j,PUnit.unit) active
    have pin := binding (2 : Fin 3) ⟨off,bound⟩ ⟨nonempty,hasSeq,Or.inr ⟨pid,(j,PUnit.unit),active,rfl⟩⟩
    change s.readMemValue .int (Region.cast p.Seq) off=v 2 ⟨off,bound⟩ at pin
    change s.readMemValue .int (Region.cast p.Seq) off=sequence p v _
    have address : off=Host.p1 pid*p.seqBatch+(Host.p2 pid/p.shape.groups*p.shape.chunk+
        RealViews.column c.N (parameters p).parameters (Host.p0 pid) j)*p.seqRow :=
      Host.sequence_window (parameters p) hasSeq _ _ _
    exact pin.trans ((by simp [sequence,off,bound] : _=sequence p v off).trans
      (congrArg (sequence p v) address))

/-- Every declared input envelope is used only at the original active source
reads. Full output backing also covers every main and cleanup store. -/
theorem input_bounds (p : Parameters) (c : Host.Config) (v : RealInputs p) (s : BlockState)
    (nonempty : p.shape.outputSize≠0) (bounds : RegionBounds)
    (bounded : (realIO p c).InBounds bounds s v) (pid : Host.PID (parameters p) c) :
    RealIO.Bounds (views p c) (s.withGridIndex pid) bounds := by
  constructor
  · intro i active
    exact bounded.1 (0 : Fin 3) ⟨_,a_bound p c pid i active⟩ ⟨nonempty,pid,i,active,rfl⟩
  · intro i active
    exact bounded.1 (1 : Fin 3) ⟨_,b_bound p c pid i active⟩ ⟨nonempty,pid,i,active,rfl⟩
  · intro hasSeq i active
    exact bounded.1 (2 : Fin 3) ⟨_,row_bound p c pid hasSeq i active⟩
      ⟨nonempty,hasSeq,Or.inl ⟨pid,i,active,rfl⟩⟩
  · intro hasSeq i active
    exact bounded.1 (2 : Fin 3) ⟨_,column_bound p c pid hasSeq i active⟩
      ⟨nonempty,hasSeq,Or.inr ⟨pid,i,active,rfl⟩⟩
  · intro i active
    exact bounded.2.1 (0 : Fin 1) ⟨_,Host.output_bound (parameters p) c pid i active⟩ trivial

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
