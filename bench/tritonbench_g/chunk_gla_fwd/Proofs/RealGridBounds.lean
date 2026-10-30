import bench.tritonbench_g.chunk_gla_fwd.Proofs.Allocation
import bench.tritonbench_g.chunk_gla_fwd.Proofs.LocalRealSafety

/-! Geometry turns finite public buffer extents into the previously proved
per-program access conditions. Bounds concern every active source access. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealGridBounds
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

theorem packed {heads T K h t k : Nat} (hh : h<heads) (ht : t<T) (hk : k<K) :
    h*(T*K)+t*K+k<heads*T*K := by
  simpa only [Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound hh ht hk

theorem inter_bounds (c : ProducerGeometry.Shape) (idx : GridIndex (ProducerGeometry.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hk : c.heads*c.T*c.K≤bounds "k") (hg : c.heads*c.T*c.K≤bounds "g")
    (ha : c.heads*c.T*c.BT≤bounds "attention") :
    RealInterSafety.Bounds (ProducerGeometry.layout c) (ProducerGeometry.launchIndex c idx)
      ProducerGeometry.regions bounds := by
  let p := ProducerGeometry.layout c
  let l := ProducerGeometry.launchIndex c idx
  have hh : l.head<c.heads := (idx ⟨2,by norm_num [ProducerGeometry.grid,Grid.rank]⟩).isLt
  have query (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) (kb : Nat) :
      RealInterAccessSafety.QueryBounds p l r kb bounds := by
    intro i j hij
    simpa only [p,ProducerGeometry.layout,Nat.mul_one]
      using (packed hh hij.1 hij.2).trans_le hr
  have key (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) (kb : Nat) :
      RealInterAccessSafety.KeyBounds p l r kb bounds := by
    intro i j hij
    have h := packed hh hij.2 hij.1
    simpa only [p,ProducerGeometry.layout,Nat.mul_one,Nat.add_right_comm]
      using h.trans_le hr
  constructor
  · intro enabled kb _
    refine ⟨query "q" hq kb,query "g" hg kb,key "k" hk kb,key "g" hg kb,?_⟩
    intro e he
    have h := (packed hh enabled.1 he).trans_le hg
    simpa only [RealInterMath.referenceAddress,p,ProducerGeometry.layout] using h
  · intro _ i active
    have h := (Geometry.attention_bound hh active.1 active.2).trans_le ha
    simpa only [RealInterMath.outputAddress,p,ProducerGeometry.layout,
      Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using h

theorem inter_safe (c : ProducerGeometry.Shape) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) (idx : GridIndex (ProducerGeometry.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hk : c.heads*c.T*c.K≤bounds "k") (hg : c.heads*c.T*c.K≤bounds "g")
    (ha : c.heads*c.T*c.BT≤bounds "attention") :
    (InterGrid.kernel c scale).TraceSafe bounds (s.withGridIndex idx) := by
  apply LocalRealSafety.inter _ _ _ _ (by simpa using hu)
  rw [ProducerGeometry.launch_grid]
  exact inter_bounds c idx bounds hq hk hg ha

theorem intra_bounds (c : ProducerGeometry.Shape) (BK : Nat)
    (idx : GridIndex (DiagonalGeometry.grid c)) (bounds : RegionBounds)
    (hq : c.heads*c.T*c.K≤bounds "q") (hk : c.heads*c.T*c.K≤bounds "k")
    (hg : c.heads*c.T*c.K≤bounds "g") (ha : c.heads*c.T*c.BT≤bounds "attention") :
    RealIntraSafety.Bounds (IntraGrid.layout c BK) (DiagonalGeometry.launchIndex c idx)
      IntraGrid.regions bounds := by
  let p := IntraGrid.layout c BK
  let l := DiagonalGeometry.launchIndex c idx
  have hh : l.head<c.heads := (idx ⟨2,by norm_num [DiagonalGeometry.grid,Grid.rank]⟩).isLt
  have matrix (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) :
      RealIntraAccessSafety.MatrixBounds p l r bounds := by
    intro i e hi he
    simpa only [RealIntraMath.matrixAddress,p,IntraGrid.layout]
      using (packed hh hi he).trans_le hr
  refine ⟨fun _=>matrix "q" hq,fun _=>matrix "g" hg,?_⟩
  intro _ j hj
  have hj16 : j<16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
  have ht : RealIntraMath.rowStart p l+j<c.T := by
    have h := lt_of_lt_of_le hj (Nat.min_le_right _ _)
    dsimp [p,l,IntraGrid.layout,RealIntraMath.rowStart,RealIntraMath.columns] at h ⊢
    omega
  have key (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) :
      RealIntraAccessSafety.KeyBounds p l r j bounds := by
    intro e he
    simpa only [RealIntraMath.keyAddress,p,IntraGrid.layout]
      using (packed hh ht he).trans_le hr
  refine ⟨key "k" hk,key "g" hg,?_⟩
  intro i hi
  have hc := DiagonalGeometry.local_bound c idx ⟨j,hj16⟩
  have h := (Geometry.attention_bound hh hi hc).trans_le ha
  simpa only [RealIntraMath.outputAddress,p,IntraGrid.layout,
    Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h

theorem intra_safe (c : ProducerGeometry.Shape) (BK : Nat) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) (idx : GridIndex (DiagonalGeometry.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hk : c.heads*c.T*c.K≤bounds "k") (hg : c.heads*c.T*c.K≤bounds "g")
    (ha : c.heads*c.T*c.BT≤bounds "attention") :
    (IntraGrid.kernel c BK scale).TraceSafe bounds (s.withGridIndex idx) := by
  apply LocalRealSafety.intra _ _ _ _ (by simpa using hu) (by decide) (by decide)
    (IntraGrid.columns_le c BK _) bounds
  rw [IntraGrid.launch_grid]
  exact intra_bounds c BK idx bounds hq hk hg ha

theorem split_bounds (c : ProducerGeometry.Shape) (idx : GridIndex (SplitGeometry.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hk : c.heads*c.T*c.K≤bounds "k") (hg : c.heads*c.T*c.K≤bounds "g")
    (hs : Geometry.ceilDiv c.K c.BK*c.heads*c.T*16≤bounds "scratch") :
    RealSplitSafety.Bounds (SplitGrid.layout c) (SplitGeometry.launchIndex c idx)
      SplitGrid.regions bounds := by
  let p := SplitGrid.layout c
  let l := SplitGeometry.launchIndex c idx
  have hh : l.head<c.heads := (idx ⟨2,by norm_num [SplitGeometry.grid,Grid.rank]⟩).isLt
  have hf : l.feature<Geometry.ceilDiv c.K c.BK :=
    (idx ⟨0,by norm_num [SplitGeometry.grid,Grid.rank]⟩).isLt
  have matrix (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) :
      RealSplitAccessSafety.MatrixBounds p l r bounds := by
    intro i e hi he
    simpa only [RealSplitMath.matrixAddress,p,SplitGrid.layout]
      using (packed hh hi he).trans_le hr
  refine ⟨fun _=>matrix "q" hq,fun _=>matrix "g" hg,?_⟩
  intro _ j hj
  have hj16 : j<16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
  have ht : RealSplitMath.rowStart p l+j<c.T := by
    have h := lt_of_lt_of_le hj (Nat.min_le_right _ _)
    dsimp [p,l,SplitGrid.layout,RealSplitMath.rowStart,RealSplitMath.columns] at h ⊢
    omega
  have key (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) :
      RealSplitAccessSafety.KeyBounds p l r j bounds := by
    intro e he
    simpa only [RealSplitMath.keyAddress,p,SplitGrid.layout]
      using (packed hh ht he).trans_le hr
  refine ⟨key "k" hk,key "g" hg,?_⟩
  intro i hi
  have h := (Geometry.scratch_bound hf hh hi hj16).trans_le hs
  simpa only [RealSplitMath.outputAddress,p,SplitGrid.layout,l,SplitGeometry.launchIndex,
    Geometry.scratchAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm] using h

theorem split_safe (c : ProducerGeometry.Shape) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) (idx : GridIndex (SplitGeometry.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hk : c.heads*c.T*c.K≤bounds "k") (hg : c.heads*c.T*c.K≤bounds "g")
    (hs : Geometry.ceilDiv c.K c.BK*c.heads*c.T*16≤bounds "scratch") :
    (SplitGrid.kernel c scale).TraceSafe bounds (s.withGridIndex idx) := by
  apply LocalRealSafety.split _ _ _ _ (by simpa using hu) (by decide) (by decide) bounds
  rw [SplitGrid.launch_grid]
  exact split_bounds c idx bounds hq hk hg hs

theorem merge_bounds (c : ProducerGeometry.Shape) (idx : GridIndex (DiagonalGeometry.grid c))
    (bounds : RegionBounds) (hs : Geometry.ceilDiv c.K c.BK*c.heads*c.T*16≤bounds "scratch")
    (ha : c.heads*c.T*c.BT≤bounds "attention") :
    RealMergeSafety.Bounds (MergeGrid.layout c) (MergeGrid.launchIndex c idx)
      "scratch" "attention" bounds := by
  let p := MergeGrid.layout c
  let l := MergeGrid.launchIndex c idx
  have hh : l.head<c.heads := (idx ⟨2,by norm_num [DiagonalGeometry.grid,Grid.rank]⟩).isLt
  constructor
  · intro _ kb hk i hi
    have h := (Geometry.scratch_bound hk hh hi i.2.1.isLt).trans_le hs
    simpa only [RealMergeProgram.inputAddress,p,MergeGrid.layout,l,MergeGrid.launchIndex,
      Geometry.scratchAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm] using h
  · intro _ i hi
    have h := (Geometry.attention_bound hh hi.1 hi.2).trans_le ha
    simpa only [RealMergeProgram.outputAddress,p,MergeGrid.layout,
      Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h

theorem merge_safe (c : ProducerGeometry.Shape) (s : BlockState)
    (hu : s.undef=fun _ _=>0) (idx : GridIndex (DiagonalGeometry.grid c))
    (bounds : RegionBounds) (hs : Geometry.ceilDiv c.K c.BK*c.heads*c.T*16≤bounds "scratch")
    (ha : c.heads*c.T*c.BT≤bounds "attention") :
    (MergeGrid.kernel c).TraceSafe bounds (s.withGridIndex idx) := by
  apply LocalRealSafety.merge _ _ _ _ (by simpa using hu) bounds
  rw [MergeGrid.launch_grid]
  exact merge_bounds c idx bounds hs ha

theorem output_bounds (c : OutputGrid.Shape) (s : BlockState) (idx : GridIndex (OutputGrid.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hg : c.heads*c.T*c.K≤bounds "g") (hv : c.heads*c.T*c.V≤bounds "v")
    (hhist : c.heads*Geometry.ceilDiv c.T c.BT*c.K*c.V≤bounds "h")
    (ha : c.heads*c.T*c.BT≤bounds "attention") (ho : c.heads*c.T*c.V≤bounds "o") :
    RealOutputSafety.Bounds (s.withGridIndex idx) "q" "v" "g" "h" "o" "attention"
      (c.T*c.K) c.K (c.T*c.V) c.V (Geometry.ceilDiv c.T c.BT*c.K*c.V) c.V
      c.T c.K c.V c.BT c.BK c.BV bounds := by
  have hh : (s.withGridIndex idx).pids 2<c.heads := by
    simpa only [BlockState.withGridIndex,GridIndex.toPids,OutputGrid.grid,Grid.rank,↓reduceDIte]
      using (idx ⟨2,by norm_num [OutputGrid.grid,Grid.rank]⟩).isLt
  have ht : (s.withGridIndex idx).pids 1<Geometry.ceilDiv c.T c.BT := by
    simpa only [BlockState.withGridIndex,GridIndex.toPids,OutputGrid.grid,Grid.rank,↓reduceDIte]
      using (idx ⟨1,by norm_num [OutputGrid.grid,Grid.rank]⟩).isLt
  have feature (r : RegionName) (hr : c.heads*c.T*c.K≤bounds r) (kb : Nat) :
      RealOutputSafety.MatrixBounds r ((s.withGridIndex idx).pids 2*(c.T*c.K))
        c.T c.K c.BT c.BK c.K 1 ((s.withGridIndex idx).pids 1*c.BT) (kb*c.BK) bounds := by
    intro i j hij
    simpa only [Nat.mul_one] using (packed hh hij.1 hij.2).trans_le hr
  constructor
  · intro kb _
    refine ⟨feature "q" hq kb,feature "g" hg kb,?_⟩
    intro i j hij
    have h := (Geometry.history_bound hh ht hij.1 hij.2).trans_le hhist
    simpa only [Geometry.historyAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.mul_one]
      using h
  · have value (r : RegionName) (hr : c.heads*c.T*c.V≤bounds r) :
        RealOutputSafety.MatrixBounds r ((s.withGridIndex idx).pids 2*(c.T*c.V))
          c.T c.V c.BT c.BV c.V 1 ((s.withGridIndex idx).pids 1*c.BT)
          ((s.withGridIndex idx).pids 0*c.BV) bounds := by
      intro i j hij
      simpa only [Nat.mul_one] using (packed hh hij.1 hij.2).trans_le hr
    refine ⟨value "v" hv,value "o" ho,?_⟩
    intro i j hij
    have h := (packed hh hij.1 hij.2).trans_le ha
    simpa only [Nat.mul_one,Nat.mul_assoc] using h

theorem output_safe (c : OutputGrid.Shape) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _=>0) (idx : GridIndex (OutputGrid.grid c))
    (bounds : RegionBounds) (hq : c.heads*c.T*c.K≤bounds "q")
    (hg : c.heads*c.T*c.K≤bounds "g") (hv : c.heads*c.T*c.V≤bounds "v")
    (hhist : c.heads*Geometry.ceilDiv c.T c.BT*c.K*c.V≤bounds "h")
    (ha : c.heads*c.T*c.BT≤bounds "attention") (ho : c.heads*c.T*c.V≤bounds "o") :
    (OutputGrid.kernel c scale).TraceSafe bounds (s.withGridIndex idx) := by
  apply LocalRealSafety.output _ _ _ _ (by simpa using hu) bounds
  exact output_bounds c s idx bounds hq hg hv hhist ha ho

#axiomsClean inter_safe
#axiomsClean intra_safe
#axiomsClean split_safe
#axiomsClean merge_safe
#axiomsClean output_safe

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealGridBounds
