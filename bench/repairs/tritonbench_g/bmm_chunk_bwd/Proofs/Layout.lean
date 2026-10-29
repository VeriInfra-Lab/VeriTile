import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Dense

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Successful ordinary allocation metadata. Supplied output is used directly;
only the absent-output branch allocates fresh storage. No execution or layout
safety condition is supplied by this descriptor. -/
structure Allocation (p : Metadata) where
  objects : Role → Nat
  freshObject : Nat
  freshPointer : Nat
  inputFits : ∀ r,Used p r → base p r+4*(p.tensor r).numel≤2^64
  freshFits : p.hasOutput=Bool.false → freshPointer+4*(p.tensor .a).numel≤2^64
  freshIdentity : p.hasOutput=Bool.false → ∀ r,Used p r → freshObject≠objects r
  freshSeparate : p.hasOutput=Bool.false → ∀ r,Used p r →
    freshPointer+4*(p.tensor .a).numel≤base p r ∨ base p r+4*(p.tensor r).numel≤freshPointer

def Allocation.pointer {p : Metadata} (a : Allocation p) := if p.hasOutput then base p .output else a.freshPointer
def Allocation.object {p : Metadata} (a : Allocation p) := if p.hasOutput then a.objects .output else a.freshObject

theorem output_size (p : Metadata) (h : InputContract p) (present : p.hasOutput=Bool.true) :
    (p.tensor .output).numel=(p.tensor .a).numel := by simp [VeriTile.Torch.TensorMetadata.numel,h.output present]
theorem residual_size (p : Metadata) (h : InputContract p) (present : p.hasResidual=Bool.true) :
    (p.tensor .residual).numel=(p.tensor .a).numel := by simp [VeriTile.Torch.TensorMetadata.numel,h.residual present]

theorem output_separate (p : Metadata) (h : InputContract p) (a : Allocation p)
    (r : Role) (used : Used p r) (ne : r≠.output) :
    a.pointer+4*(p.tensor .a).numel≤base p r ∨ base p r+4*(p.tensor r).numel≤a.pointer := by
  cases present : p.hasOutput
  · simpa [Allocation.pointer,present] using a.freshSeparate present r used
  · have hh := h.separate .output r (by simp [Used,present]) used (Ne.symm ne)
    simpa [Allocation.pointer,present,output_size p h present] using hh

theorem output_fits (p : Metadata) (h : InputContract p) (a : Allocation p) :
    a.pointer+4*(p.tensor .a).numel≤2^64 := by
  cases present : p.hasOutput
  · simpa [Allocation.pointer,present] using a.freshFits present
  · simpa [Allocation.pointer,present,output_size p h present] using a.inputFits .output (by simp [Used,present])

def extent (p : Metadata) (r : RegionName) : Nat :=
  if r="A" then (p.tensor .a).numel else if r="Dout" then (p.tensor .dout).numel
  else if r="Residual" then if p.hasResidual then (p.tensor .residual).numel else 0
  else if r="Out" then (p.tensor .a).numel else 0

def flat (p : Metadata) (a : Allocation p) : FlatAlloc where
  flat := "heap"
  regions := ["A","Dout","Residual","Out"]
  base := fun r => if r="A" then base p .a else if r="Dout" then base p .dout
    else if r="Residual" then (if p.hasResidual then base p .residual else 0) else if r="Out" then a.pointer else 0
  extent := extent p

theorem extent_closed (p : Metadata) (r : RegionName)
    (outside : r∉(["A","Dout","Residual","Out"] : List RegionName)) : extent p r=0 := by
  simp only [List.mem_cons,List.not_mem_nil,or_false,not_or] at outside
  simp [extent,outside.1,outside.2.1,outside.2.2.1,outside.2.2.2]

theorem flat_disjoint (p : Metadata) (h : InputContract p) (a : Allocation p) : (flat p a).Disjoint := by
  have usedA : Used p .a := by simp [Used]
  have usedD : Used p .dout := by simp [Used]
  have ad := h.separate .a .dout usedA usedD (by decide)
  have ao := output_separate p h a .a usedA (by decide)
  have d_o := output_separate p h a .dout usedD (by decide)
  intro r hr s hs different
  simp only [flat,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl|rfl|rfl|rfl <;> rcases hs with rfl|rfl|rfl|rfl
  all_goals try { exact False.elim (different rfl) }
  all_goals simp only [flat,extent,ite_true,show ("A":RegionName)≠"Dout" by decide,
    show ("A":RegionName)≠"Residual" by decide,show ("A":RegionName)≠"Out" by decide,
    show ("Dout":RegionName)≠"A" by decide,show ("Dout":RegionName)≠"Residual" by decide,
    show ("Dout":RegionName)≠"Out" by decide,show ("Residual":RegionName)≠"A" by decide,
    show ("Residual":RegionName)≠"Dout" by decide,show ("Residual":RegionName)≠"Out" by decide,
    show ("Out":RegionName)≠"A" by decide,show ("Out":RegionName)≠"Dout" by decide,
    show ("Out":RegionName)≠"Residual" by decide,ite_false]
  all_goals try omega
  all_goals cases res : p.hasResidual
  all_goals simp only [res,Bool.false_eq_true,↓reduceIte,Nat.add_zero]
  all_goals try omega
  all_goals
    have usedR : Used p .residual := by simp [Used,res]
    have ar := h.separate .a .residual usedA usedR (by decide)
    have dr := h.separate .dout .residual usedD usedR (by decide)
    have r_o := output_separate p h a .residual usedR (by decide)
    omega

theorem dense_bounds (p : Metadata) (h : InputContract p) (c : Tuning)
    (q : HostGrid.Pid) (hq : q∈HostGrid.programs (HostContract.grid (dense p) c)) (s : BlockState) :
    RealIO.Bounds (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p)))
      (HostGrid.launch (HostContract.grid (dense p) c) q s) (extent p) := by
  have coords := grid_coordinates p c q hq
  constructor
  · intro i hi
    simp [HostGrid.io,RealViews.io,HostGrid.launch] at hi
    change row p c q i.1<chunk p ∧ i.2.1.val<limit p q at hi
    change (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).doutWindow q.tile q.batch q.chunkHead i<_
    rw [dout_offset p c q i coords.2.2]
    exact dout_bound p h _ _ _ _ _ coords.1 coords.2.1 coords.2.2 i.2.1.isLt hi.1
  · intro i hi
    simp [HostGrid.io,RealViews.io,HostGrid.launch] at hi
    change i.1.val<limit p q ∧ column p c q i.2.1<width p at hi
    change (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).aWindow q.tile q.batch q.chunkHead i<_
    rw [a_offset p c q i coords.2.2]
    exact tensor_bound p h _ _ _ _ coords.1 (valid_sequence p q _ hi.1) coords.2.2 hi.2
  · intro present i hi
    change p.hasResidual=Bool.true at present
    simp [HostGrid.io,RealViews.io,HostGrid.launch] at hi
    change row p c q i.1<limit p q ∧ column p c q i.2.1<width p at hi
    change (HostGrid.io Bool.false c.BM c.BN c.BCS (HostContract.parameters (dense p))).residualWindow q.tile q.batch q.chunkHead i<_
    rw [residual_offset p c q i coords.2.2 present]
    simpa [extent,present,residual_size p h present] using
      tensor_bound p h _ _ _ _ coords.1 (valid_sequence p q _ hi.1) coords.2.2 hi.2
  · intro i hi
    simp [HostGrid.io,RealViews.io,HostGrid.launch] at hi
    change row p c q i.1<limit p q ∧ column p c q i.2.1<width p at hi
    change HostGrid.address c.BM c.BN (HostContract.parameters (dense p)) q i<_
    rw [output_offset p c q i coords.2.2]
    exact tensor_bound p h _ _ _ _ coords.1 (valid_sequence p q _ hi.1) coords.2.2 hi.2

def denseLayout (p : Metadata) (h : InputContract p) (a : Allocation p)
    (nonempty : 0<(p.tensor .a).numel) (c : Tuning) : HostContract.Layout (dense p) c (flat p a) where
  disjoint := flat_disjoint p h a
  closed := extent_closed p
  outputAllocated := by simp [flat,dense,call]
  inputsSeparate := by
    change ("A":RegionName)≠"Out" ∧ ("Dout":RegionName)≠"Out" ∧ ("Residual":RegionName)≠"Out"
    decide
  outputDistinct := output_distinct p h nonempty c
  programWrites := program_writes p h c
  bounds := dense_bounds p h c

def layout (p : Metadata) (h : InputContract p) (a : Allocation p)
    (nonempty : 0<(p.tensor .a).numel) (c : Tuning) : HostContract.Layout (call p) c (flat p a) where
  disjoint := flat_disjoint p h a
  closed := extent_closed p
  outputAllocated := by simp [flat,call]
  inputsSeparate := by
    change ("A":RegionName)≠"Out" ∧ ("Dout":RegionName)≠"Out" ∧ ("Residual":RegionName)≠"Out"
    decide
  outputDistinct := by rw [parameters_eq p h nonempty]; exact output_distinct p h nonempty c
  programWrites := by rw [parameters_eq p h nonempty,grid_eq p h nonempty]; exact program_writes p h c
  bounds := by
    intro q hq s
    rw [parameters_eq p h nonempty,grid_eq p h nonempty]
    rw [grid_eq p h nonempty] at hq
    exact dense_bounds p h c q hq s

#axiomsClean dense_bounds
#axiomsClean flat_disjoint
#axiomsClean layout
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Direct
