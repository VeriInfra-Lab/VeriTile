import bench.repairs.tritonbench_g.lightning_attention.Proofs.Geometry
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def RowsFoot (N width head start rows : Nat) (dest r : RegionName) (o : Nat) : Prop :=
  r=dest ∧ ∃ lane, lane<rows ∧ start+lane<N ∧ ∃ feature, feature<width ∧
    o=Geometry.dense N width head (start+lane) feature

def HeadFoot (N width head : Nat) (dest r : RegionName) (o : Nat) : Prop :=
  r=dest ∧ ∃ time, time<N ∧ ∃ feature, feature<width ∧ o=Geometry.dense N width head time feature

def FwdFoot (p : Parameters) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  r="output" ∧ ∃ time, time<p.N ∧ ∃ feature, feature<p.BM ∧
    o=Geometry.dense p.N p.E pid.head time (pid.channel*p.BM+feature)

theorem rows_head (N width head start rows : Nat) (dest r : RegionName) (o : Nat)
    (h : RowsFoot N width head start rows dest r o) : HeadFoot N width head dest r o := by
  obtain ⟨hr,lane,_,ht,feature,hf,he⟩ := h
  exact ⟨hr,start+lane,ht,feature,hf,he⟩

theorem writeRows_foot_frame (M : Algebra α) (dest : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬RowsFoot N width head start rows dest r o) :
    writeRows M dest N width head start rows values m r o=m r o := by
  apply scatter_frame
  by_cases hr : r=dest
  · right
    intro i _ hi he
    apply hs
    refine ⟨hr,i.1.val,i.1.isLt,hi,i.2.1.val,i.2.1.isLt,?_⟩
    rw [←Geometry.row_address N width head start rows i]
    exact he
  · exact Or.inl hr

theorem writeTrans_foot_frame (M : Algebra α) (dest : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬RowsFoot N width head start rows dest r o) :
    writeTrans M dest N width head start rows values m r o=m r o := by
  apply scatter_frame
  by_cases hr : r=dest
  · right
    intro i _ hi he
    apply hs
    refine ⟨hr,i.2.1.val,i.2.1.isLt,hi,i.1.val,i.1.isLt,?_⟩
    rw [←Geometry.trans_address N width head start rows i]
    exact he
  · exact Or.inl hr

theorem writeRows_head_frame (M : Algebra α) (dest : RegionName) (N width head start rows : Nat)
    (values : Matrix α rows width) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬HeadFoot N width head dest r o) :
    writeRows M dest N width head start rows values m r o=m r o :=
  writeRows_foot_frame _ _ _ _ _ _ _ _ _ _ _ (fun hw => hs (rows_head _ _ _ _ _ _ _ _ hw))
theorem writeTrans_head_frame (M : Algebra α) (dest : RegionName) (N width head start rows : Nat)
    (values : Matrix α width rows) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬HeadFoot N width head dest r o) :
    writeTrans M dest N width head start rows values m r o=m r o :=
  writeTrans_foot_frame _ _ _ _ _ _ _ _ _ _ _ (fun hw => hs (rows_head _ _ _ _ _ _ _ _ hw))

theorem writeOutput_frame (M : Algebra α) (p : Parameters) (pid : PID) (block : Nat)
    (values : Matrix α p.BT p.BM) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬FwdFoot p pid r o) : writeOutput M p pid block values m r o=m r o := by
  apply scatter_frame
  by_cases hr : r="output"
  · right
    intro i _ hi he
    exact hs ⟨hr,block*p.BT+i.1.val,hi,i.2.1.val,i.2.1.isLt,
      he.trans (Geometry.fwd_address p pid block i)⟩
  · exact Or.inl hr

theorem fwdMemory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (r : RegionName) (o : Nat) (hs : ¬FwdFoot p pid r o) :
    fwdMemory M D p pid input m r o=m r o := by
  unfold fwdMemory
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro n hn mem
  exact writeOutput_frame _ _ _ _ _ _ _ _ hs

def IntraFoot (p : Parameters) (head block : Nat) (r : RegionName) (o : Nat) : Prop :=
  RowsFoot p.N p.D head (block*p.BT) p.BT "queryGradient" r o ∨
  RowsFoot p.N p.D head (block*p.BT) p.BT "keyGradient" r o ∨
  RowsFoot p.N p.E head (block*p.BT) p.BT "valueGradient" r o

def InterFoot (p : Parameters) (head : Nat) (r : RegionName) (o : Nat) : Prop :=
  HeadFoot p.N p.D head "queryGradient" r o ∨
  HeadFoot p.N p.D head "keyGradient" r o ∨ HeadFoot p.N p.E head "valueGradient" r o

theorem intraMemory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (m : Memory α) (r : RegionName) (o : Nat) (hs : ¬IntraFoot p head block r o) :
    intraMemory M D p head block m r o=m r o := by
  unfold intraMemory
  rw [writeRows_foot_frame _ _ _ _ _ _ _ _ _ _ _ (fun hw => hs (Or.inr (Or.inr hw))),
    writeTrans_foot_frame _ _ _ _ _ _ _ _ _ _ _ (fun hw => hs (Or.inr (Or.inl hw))),
    writeRows_foot_frame _ _ _ _ _ _ _ _ _ _ _ (fun hw => hs (Or.inl hw))]

theorem queryUpdate_head_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬HeadFoot p.N p.D head "queryGradient" r o) :
    queryUpdate M D p head row carry m r o=m r o :=
  writeRows_head_frame _ _ _ _ _ _ _ _ _ _ _ hs

theorem keyValueUpdate_head_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) (r : RegionName) (o : Nat)
    (hk : ¬HeadFoot p.N p.D head "keyGradient" r o)
    (hv : ¬HeadFoot p.N p.E head "valueGradient" r o) :
    keyValueUpdate M D p head row carry m r o=m r o := by
  unfold keyValueUpdate
  rw [writeRows_head_frame _ _ _ _ _ _ _ _ _ _ _ hv,
    writeTrans_head_frame _ _ _ _ _ _ _ _ _ _ _ hk]

theorem queryScanMemory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input m : Memory α) (r : RegionName) (o : Nat)
    (hs : ¬HeadFoot p.N p.D head "queryGradient" r o) :
    queryScanMemory M D p head input m r o=m r o := by
  unfold queryScanMemory
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro block hb mem
  unfold queryBlockMemory
  split
  · rfl
  · apply foldRange_observe _ (fun mem : Memory α => mem r o)
    intro sub hs' mem
    exact queryUpdate_head_frame _ _ _ _ _ _ _ _ _ hs

theorem keyValueScanMemory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input m : Memory α) (r : RegionName) (o : Nat)
    (hk : ¬HeadFoot p.N p.D head "keyGradient" r o)
    (hv : ¬HeadFoot p.N p.E head "valueGradient" r o) :
    keyValueScanMemory M D p head input m r o=m r o := by
  unfold keyValueScanMemory
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro block hb mem
  unfold keyValueBlockMemory
  split
  · rfl
  · apply foldRange_observe _ (fun mem : Memory α => mem r o)
    intro sub hs' mem
    exact keyValueUpdate_head_frame _ _ _ _ _ _ _ _ _ hk hv

theorem interMemory_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input m : Memory α) (r : RegionName) (o : Nat) (hs : ¬InterFoot p head r o) :
    interMemory M D p head input m r o=m r o := by
  unfold interMemory
  rw [keyValueScanMemory_frame _ _ _ _ _ _ _ _ (fun hw => hs (Or.inr (Or.inl hw)))
      (fun hw => hs (Or.inr (Or.inr hw))),
    queryScanMemory_frame _ _ _ _ _ _ _ _ (fun hw => hs (Or.inl hw))]

def Footprint (stage : Stage) (d : Host.Dimensions) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  match stage with
  | .forward => FwdFoot d.parameters pid r o
  | .intra => IntraFoot d.parameters pid.head pid.channel r o
  | .inter => InterFoot d.parameters pid.head r o

theorem stage_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : Memory α) (r : RegionName) (o : Nat) (hs : ¬Footprint stage d pid r o) :
    stageMemory M D stage d pid m r o=m r o := by
  cases stage with
  | forward => exact fwdMemory_frame _ _ _ _ _ _ _ _ hs
  | intra => exact intraMemory_frame _ _ _ _ _ _ _ _ hs
  | inter => exact interMemory_frame _ _ _ _ _ _ _ _ hs

theorem grid_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : Memory α) (r : RegionName) (o : Nat)
    (hs : ∀ pid∈schedule, ¬Footprint stage d pid r o) :
    gridMemory M D stage d schedule m r o=m r o := by
  induction schedule generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    change gridMemory M D stage d rest (stageMemory M D stage d pid m) r o=_
    rw [ih _ (fun id hid => hs id (by simp [hid]))]
    exact stage_frame _ _ _ _ _ _ _ _ (hs pid (by simp))

theorem input_not_written (stage : Stage) (d : Host.Dimensions) (pid : PID) (r : RegionName) (o : Nat)
    (hr : r∈(["query","key","value","incoming"] : List RegionName)) : ¬Footprint stage d pid r o := by
  intro hw
  cases stage with
  | forward => obtain ⟨rfl,_⟩ := hw; simp at hr
  | intra => rcases hw with ⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩ <;> simp at hr
  | inter => rcases hw with ⟨rfl,_⟩|⟨rfl,_⟩|⟨rfl,_⟩ <;> simp at hr

theorem stage_inputs (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : Memory α) : Inputs (stageMemory M D stage d pid m) m := by
  intro r hr
  funext o
  exact stage_frame _ _ _ _ _ _ _ _ (input_not_written _ _ _ _ _ hr)
theorem grid_inputs (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : Memory α) : Inputs (gridMemory M D stage d schedule m) m := by
  intro r hr
  funext o
  apply grid_frame
  intro pid hp
  exact input_not_written _ _ _ _ _ hr
#axiomsClean stage_frame
#axiomsClean grid_frame
#axiomsClean stage_inputs
#axiomsClean grid_inputs
end
end VeriTile.Bench.Repairs.LightningAttention
