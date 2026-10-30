import bench.tritonbench_g.chunk_gla_fwd.Proofs.WholeReal
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Allocation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def attention (c : Shape) : WriteFootprint :=
  fun x => x.1="attention" ∧ x.2<c.heads*c.T*c.BT
def scratch (c : Shape) : WriteFootprint :=
  fun x => x.1="scratch" ∧ x.2<Geometry.ceilDiv c.K c.BK*c.heads*c.T*16

theorem inter_bound (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (cell : RegionName×Nat) (hc : Kernel.GridWriteFootprint (InterGrid.frames c scale s hu) cell) :
    attention c cell := by
  rcases hc with ⟨q,hr,hen,i,hi,he⟩
  refine ⟨hr,?_⟩
  rw [he]
  have hh : (ProducerGeometry.launchIndex c q).head<c.heads :=
    (q ⟨2,by norm_num [ProducerGeometry.grid,Grid.rank]⟩).isLt
  have hb := Geometry.attention_bound hh hi.1 hi.2
  simpa only [Geometry.attentionAddress,RealInterMath.outputAddress,ProducerGeometry.layout,Nat.add_mul,Nat.mul_assoc] using hb

theorem intra_bound (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (cell : RegionName×Nat) (hc : Kernel.GridWriteFootprint (IntraGrid.frames c BK hK scale s hu) cell) :
    attention c cell := by
  rcases hc with ⟨q,hq⟩
  obtain ⟨hr,i,hi,he⟩ := IntraGrid.footprint_tile c BK q cell hq
  refine ⟨hr,?_⟩
  rw [he]
  have hh : (DiagonalGeometry.launchIndex c q).head<c.heads :=
    (q ⟨2,by norm_num [DiagonalGeometry.grid,Grid.rank]⟩).isLt
  have hb := Geometry.attention_bound hh hi.1 hi.2
  simpa only [Geometry.attentionAddress,DiagonalGeometry.address,Nat.add_mul,Nat.mul_assoc] using hb

theorem split_bound (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (cell : RegionName×Nat) (hc : Kernel.GridWriteFootprint (SplitGrid.frames c scale s hu) cell) :
    scratch c cell := by
  rcases hc with ⟨q,hq⟩
  obtain ⟨hr,i,hi,he⟩ := SplitGrid.footprint_tile c q cell hq
  refine ⟨hr,?_⟩
  rw [he]
  exact Geometry.scratch_bound
    (q ⟨0,by norm_num [SplitGeometry.grid,Grid.rank]⟩).isLt
    (q ⟨2,by norm_num [SplitGeometry.grid,Grid.rank]⟩).isLt hi i.2.1.isLt

theorem merge_bound (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (cell : RegionName×Nat) (hc : Kernel.GridWriteFootprint (MergeGrid.frames c s hu) cell) :
    attention c cell := by
  rcases hc with ⟨q,hr,hen,i,hi,he⟩
  refine ⟨hr,?_⟩
  rw [he,MergeGrid.source_address]
  have hh : (DiagonalGeometry.launchIndex c q).head<c.heads :=
    (q ⟨2,by norm_num [DiagonalGeometry.grid,Grid.rank]⟩).isLt
  have hi' : DiagonalGeometry.active c (DiagonalGeometry.launchIndex c q) i := hi
  have hb := Geometry.attention_bound hh hi'.1 hi'.2
  simpa only [Geometry.attentionAddress,DiagonalGeometry.address,Nat.add_mul,Nat.mul_assoc] using hb

theorem inter_frame (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (attention c) s (InterGrid.result c scale s hu) := by
  intro r off hn
  exact Kernel.mergeFrames_writeWithin (InterGrid.frames c scale s hu) r off
    (fun hw => hn (inter_bound c scale s hu _ hw))

theorem intra_frame (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (attention c) s (IntraGrid.result c BK hK scale s hu) := by
  intro r off hn
  exact Kernel.mergeFrames_writeWithin (IntraGrid.frames c BK hK scale s hu) r off
    (fun hw => hn (intra_bound c BK hK scale s hu _ hw))

theorem split_frame (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (scratch c) s (SplitGrid.result c scale s hu) := by
  intro r off hn
  exact Kernel.mergeFrames_writeWithin (SplitGrid.frames c scale s hu) r off
    (fun hw => hn (split_bound c scale s hu _ hw))

theorem merge_frame (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (attention c) s (MergeGrid.result c s hu) := by
  intro r off hn
  exact Kernel.mergeFrames_writeWithin (MergeGrid.frames c s hu) r off
    (fun hw => hn (merge_bound c s hu _ hw))

theorem direct_frame (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (attention c) s (Producer.direct c BK hK scale s hu) := by
  intro r off hn
  exact (inter_frame c scale s hu r off hn).trans
    (intra_frame c BK hK scale (InterGrid.result c scale s hu) hu r off hn)

theorem large_frame (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    BlockState.WriteWithin (fun cell => attention c cell ∨ scratch (Producer.splitShape c) cell)
      s (Producer.large c scale s hu) := by
  intro r off hn
  have ha : ¬attention c (r,off) := fun h => hn (Or.inl h)
  have hs : ¬scratch (Producer.splitShape c) (r,off) := fun h => hn (Or.inr h)
  exact (inter_frame c scale s hu r off ha).trans
    ((split_frame (Producer.splitShape c) scale (InterGrid.result c scale s hu) hu r off hs).trans
      (merge_frame (Producer.splitShape c)
        (SplitGrid.result (Producer.splitShape c) scale (InterGrid.result c scale s hu) hu) hu r off ha))

/-- Finite erased zero allocation; cells outside the allocated extent stay
unchanged. Fresh physical allocation is handled by the host memory contract. -/
def zeroRange (s : BlockState) (region : RegionName) (size : Nat) : BlockState :=
  {s with mem:=fun r off => if r=region ∧ off<size then MemCell.real 0 else s.mem r off}

theorem zeroRange_read (s : BlockState) (region : RegionName) (size off : Nat) (ho : off<size) :
    (zeroRange s region size).readMem region off=0 := by
  simp [zeroRange,BlockState.readMem,ho]

theorem zeroRange_frame (s : BlockState) (region : RegionName) (size : Nat) :
    BlockState.WriteWithin (fun x => x.1=region ∧ x.2<size) s (zeroRange s region size) := by
  intro r off hn
  simp [zeroRange,hn]

def initialized (c : Shape) (s : BlockState) : BlockState :=
  zeroRange (zeroRange s "attention" (c.heads*c.T*c.BT)) "scratch" (Geometry.ceilDiv c.K 128*c.heads*c.T*16)

theorem initialized_attention (c : Shape) (s : BlockState) : Producer.ZeroAttention c (initialized c s) := by
  intro h t j
  have hb : h.val*c.T*c.BT+t.val*c.BT+j.val<c.heads*c.T*c.BT := by
    simpa only [Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using Geometry.attention_bound h.isLt t.isLt j.isLt
  simp only [initialized,zeroRange,BlockState.readMem]
  simp [hb]

theorem initialized_scratch (c : Shape) (s : BlockState) : Producer.ZeroScratch c (initialized c s) := by
  intro f h t j
  apply zeroRange_read
  exact Geometry.scratch_bound f.isLt h.isLt t.isLt j.isLt

theorem initialized_inputs (c : Shape) (s : BlockState) : InterGrid.inputs (initialized c s)=InterGrid.inputs s := by
  apply StageFrames.inputs_of_frame
  intro r hra hrs off
  simp [initialized,zeroRange,hra,hrs]

#axiomsClean direct_frame
#axiomsClean large_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Allocation
