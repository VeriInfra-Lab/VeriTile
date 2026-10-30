import bench.tritonbench_g.matmul_triton_autotune.Sources.Mathematical_WholeTask

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def paddedInner (r : Request) := r.parameters.numKBlocks*r.config.bk
def envelopeA (r : Request) := r.rows*r.strideAM+(paddedInner r+r.config.bk)*r.strideAK
def envelopeB (r : Request) := (paddedInner r+r.config.bk)*r.strideBK+r.columns*r.strideBN
def envelopeC (r : Request) := tileRows r*r.config.bm*max r.columns 1+tileColumns r*r.config.bn

structure TensorMetadata where
  (shape strides : List Nat)
  dtype : FloatDType
  device : String

def byteSpansDisjoint (a n b m : Nat) : Prop := n=0 ∨ m=0 ∨ a+n≤b ∨ b+m≤a

/-- Ordinary rank-two tensor/view and fresh torch.empty allocation contracts.
A/B may alias; C has a fresh byte span. No output values are constrained. Input
strides are the observed tensor strides, including noncontiguous B views. -/
structure TensorContract {α : Type} (r : Request) (s : State α) where
  (extentA extentB : Nat)
  (baseA baseB baseC : Nat)
  (metadataA metadataB metadataC : TensorMetadata)
  shapeA : metadataA.shape=[r.rows,r.inner]
  shapeB : metadataB.shape=[r.innerB,r.columns]
  shapeC : metadataC.shape=[r.rows,r.columns]
  stridesA : metadataA.strides=[r.strideAM,r.strideAK]
  stridesB : metadataB.strides=[r.strideBK,r.strideBN]
  stridesC : metadataC.strides=[max r.columns 1,1]
  dtypeA : metadataA.dtype=.fp32
  dtypeB : metadataB.dtype=.fp32
  dtypeC : metadataC.dtype=.fp16
  deviceB : metadataB.device=metadataA.device
  deviceC : metadataC.device=metadataA.device
  dimensions : r.inner=r.innerB
  contiguousA : r.aContiguous=Bool.true
  selected : r.config∈configurations r.target
  freshA : r.C≠r.A
  freshB : r.C≠r.B
  physicalFreshA : byteSpansDisjoint baseC (2*(r.rows*r.columns)) baseA (4*extentA)
  physicalFreshB : byteSpansDisjoint baseC (2*(r.rows*r.columns)) baseB (4*extentB)
  alignmentA : baseA%4=0
  alignmentB : baseB%4=0
  alignmentC : baseC%2=0
  spanA : baseA+4*extentA≤2^64
  spanB : baseB+4*extentB≤2^64
  spanC : baseC+2*(r.rows*r.columns)≤2^64
  sameViewOrigin : r.A=r.B → baseA=baseB
  aliasConsistency : ∀ row<r.rows, ∀ k<r.inner, ∀ k'<r.inner, ∀ col<r.columns,
    baseA+4*(row*r.strideAM+k*r.strideAK)=baseB+4*(k'*r.strideBK+col*r.strideBN) →
    s.mem r.A (row*r.strideAM+k*r.strideAK)=s.mem r.B (k'*r.strideBK+col*r.strideBN)
  inputA : ∀ row<r.rows, ∀ k<r.inner, row*r.strideAM+k*r.strideAK<extentA
  inputB : ∀ k<r.inner, ∀ col<r.columns, k*r.strideBK+col*r.strideBN<extentB
  tagA : ∀ row<r.rows, ∀ k<r.inner, ∃ x, s.mem r.A (row*r.strideAM+k*r.strideAK)=Cell.mk .fp32 x
  tagB : ∀ k<r.inner, ∀ col<r.columns, ∃ x, s.mem r.B (k*r.strideBK+col*r.strideBN)=Cell.mk .fp32 x
  byteA : baseA+4*envelopeA r<2^64
  byteB : baseB+4*envelopeB r<2^64
  byteC : baseC+2*envelopeC r<2^64

/-- Sufficient signed-index domain, including padded pointer formation and
last pointer advances. It is separate from the unrestricted Nat program proof. -/
structure IndexDomain (r : Request) : Prop where
  rows : r.rows+r.config.bm<2^31
  columns : r.columns+r.config.bn<2^31
  inner : r.inner+r.config.bk<2^31
  strideAM : r.strideAM<2^31
  strideAK : r.strideAK<2^31
  strideBK : r.strideBK<2^31
  strideBN : r.strideBN<2^31
  grid : programs r<2^31
  group : r.config.gm*tileColumns r<2^31
  addressA : envelopeA r<2^31
  addressB : envelopeB r<2^31
  addressC : envelopeC r<2^31

theorem nonempty_program (r : Request) (pid : Nat) (hp : pid<programs r) :
    0<r.rows ∧ 0<r.columns := by
  constructor
  · by_contra h
    have hz : r.rows=0 := by omega
    cases hh : r.config <;> simp [programs,tileRows,hz,hh,Config.bm] at hp
  · by_contra h
    have hz : r.columns=0 := by omega
    simp [programs,tileColumns,hz,Config.bn] at hp

theorem source_subtraction (r : Request) (b : Nat) (hb : b<r.parameters.numKBlocks) :
    b*r.config.bk<r.inner ∧ (r.inner : Int)-(b*r.config.bk : Nat)=(r.inner-b*r.config.bk : Nat) := by
  cases hh : r.config <;> simp [Request.parameters,hh,Config.bk] at hb ⊢ <;> omega

theorem output_bound (r : Request) (i j : Nat) (hi : i<r.rows) (hj : j<r.columns) :
    outputOffset r i j<r.rows*r.columns := by
  have hn : max r.columns 1=r.columns := by omega
  unfold outputOffset
  rw [hn]
  have hh := Nat.mul_le_mul_left r.columns (show i+1≤r.rows by omega)
  nlinarith

theorem a_offset {α : Type} (r : Request) (s : State α) (b : Nat) (i : FPValue.AIndex r.parameters) :
    FPValue.aOffset r.parameters s b i=
      FPValue.aRow r.parameters s i.1*r.strideAM+(b*r.config.bk+i.2.1.val)*r.strideAK := by
  simp only [FPValue.aOffset,FPValue.BK,Request.parameters]; ring

theorem b_offset {α : Type} (r : Request) (s : State α) (b : Nat) (i : FPValue.BIndex r.parameters) :
    FPValue.bOffset r.parameters s b i=
      (b*r.config.bk+i.1.val)*r.strideBK+FPValue.bCol r.parameters s i.2.1*r.strideBN := by
  simp only [FPValue.bOffset,FPValue.BK,Request.parameters]; ring

theorem loadA_bound {α : Type} (r : Request) (s : State α) (t : TensorContract r s)
    (pid b : Nat) (hp : pid<programs r) (i : FPValue.AIndex r.parameters)
    (hm : i.2.1.val<r.inner-b*r.config.bk) :
    FPValue.aOffset r.parameters (instanceState r pid s) b i<t.extentA := by
  have hr := (nonempty_program r pid hp).1
  rw [a_offset]
  exact t.inputA _ (Nat.mod_lt _ hr) _ (by omega)

theorem loadB_bound {α : Type} (r : Request) (s : State α) (t : TensorContract r s)
    (pid b : Nat) (hp : pid<programs r) (i : FPValue.BIndex r.parameters)
    (hm : i.1.val<r.inner-b*r.config.bk) :
    FPValue.bOffset r.parameters (instanceState r pid s) b i<t.extentB := by
  have hc := (nonempty_program r pid hp).2
  rw [b_offset]
  exact t.inputB _ (by omega) _ (Nat.mod_lt _ hc)

theorem store_bound {α : Type} (r : Request) (s : State α) (pid : Nat)
    (i : FPValue.CIndex r.parameters) (hi : FPValue.active r.parameters (instanceState r pid s) i) :
    FPValue.cOffset r.parameters (instanceState r pid s) i<r.rows*r.columns := by
  rw [source_offset]
  exact output_bound r _ _ hi.1 hi.2

theorem padded_a_bound {α : Type} (r : Request) (s : State α) (pid b : Nat)
    (hp : pid<programs r) (hb : b≤r.parameters.numKBlocks) (i : FPValue.AIndex r.parameters) :
    FPValue.aOffset r.parameters (instanceState r pid s) b i≤envelopeA r := by
  have hr := (nonempty_program r pid hp).1
  have ha : FPValue.aRow r.parameters (instanceState r pid s) i.1<r.rows := Nat.mod_lt _ hr
  have hi : i.2.1.val<r.config.bk := i.2.1.isLt
  have hk := Nat.mul_le_mul_right r.config.bk hb
  have hk' : b*r.config.bk+i.2.1.val≤paddedInner r+r.config.bk := by unfold paddedInner; omega
  rw [a_offset]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ (Nat.le_of_lt ha)) (Nat.mul_le_mul_right _ hk')

theorem padded_b_bound {α : Type} (r : Request) (s : State α) (pid b : Nat)
    (hp : pid<programs r) (hb : b≤r.parameters.numKBlocks) (i : FPValue.BIndex r.parameters) :
    FPValue.bOffset r.parameters (instanceState r pid s) b i≤envelopeB r := by
  have hc := (nonempty_program r pid hp).2
  have hn : FPValue.bCol r.parameters (instanceState r pid s) i.2.1<r.columns := Nat.mod_lt _ hc
  have hi : i.1.val<r.config.bk := i.1.isLt
  have hk := Nat.mul_le_mul_right r.config.bk hb
  have hk' : b*r.config.bk+i.1.val≤paddedInner r+r.config.bk := by unfold paddedInner; omega
  rw [b_offset]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ hk') (Nat.mul_le_mul_right _ (Nat.le_of_lt hn))

theorem padded_c_bound {α : Type} (r : Request) (s : State α) (pid : Nat)
    (hp : pid<programs r) (i : FPValue.CIndex r.parameters) :
    FPValue.cOffset r.parameters (instanceState r pid s) i≤envelopeC r := by
  have hg := grouped_bounds (tileRows r) (tileColumns r) r.config.gm pid r.config.positive.2.2.2 hp
  have hr : FPValue.row r.parameters (instanceState r pid s) i.1<tileRows r*r.config.bm := by
    change FPValue.pidM r.parameters (instanceState r pid s)*r.config.bm+i.1.val<_
    rw [source_pidM]
    have hi : i.1.val<r.config.bm := i.1.isLt
    have hmul := Nat.mul_le_mul_right r.config.bm (Nat.succ_le_of_lt hg.2.1)
    nlinarith
  have hc : FPValue.col r.parameters (instanceState r pid s) i.2.1<tileColumns r*r.config.bn := by
    change FPValue.pidN r.parameters (instanceState r pid s)*r.config.bn+i.2.1.val<_
    rw [source_pidN]
    have hi : i.2.1.val<r.config.bn := i.2.1.isLt
    have hmul := Nat.mul_le_mul_right r.config.bn (Nat.succ_le_of_lt hg.2.2)
    nlinarith
  rw [source_offset]
  unfold outputOffset envelopeC
  rw [Nat.mul_comm (max r.columns 1)]
  exact Nat.add_le_add (Nat.mul_le_mul_right _ (Nat.le_of_lt hr)) (Nat.le_of_lt hc)

/-- Actual source loads and stores are bounded; all padded addresses have
finite signed offsets and finite byte pointers; each output has one owner. -/
structure SourceSafety {α : Type} (r : Request) (s : State α) (t : TensorContract r s) : Prop where
  control : ∀ pid, pid<programs r → 0<r.rows ∧ 0<r.columns ∧
    0<groupSize (tileRows r) (tileColumns r) r.config.gm pid ∧
    tileRow (tileRows r) (tileColumns r) r.config.gm pid<tileRows r ∧
    tileCol (tileRows r) (tileColumns r) r.config.gm pid<tileColumns r ∧
    pid<2^31 ∧ r.config.gm*tileColumns r<2^31
  groupSubtraction : ∀ pid, pid<programs r →
    groupBase (tileRows r) (tileColumns r) r.config.gm pid<tileRows r
  subtraction : ∀ b, b<r.parameters.numKBlocks →
    b*r.config.bk<r.inner ∧ (r.inner : Int)-(b*r.config.bk : Nat)=(r.inner-b*r.config.bk : Nat)
  loadA : ∀ pid b, pid<programs r → ∀ i : FPValue.AIndex r.parameters,
    i.2.1.val<r.inner-b*r.config.bk → FPValue.aOffset r.parameters (instanceState r pid s) b i<t.extentA
  loadB : ∀ pid b, pid<programs r → ∀ i : FPValue.BIndex r.parameters,
    i.1.val<r.inner-b*r.config.bk → FPValue.bOffset r.parameters (instanceState r pid s) b i<t.extentB
  store : ∀ pid (i : FPValue.CIndex r.parameters), FPValue.active r.parameters (instanceState r pid s) i →
    FPValue.cOffset r.parameters (instanceState r pid s) i<r.rows*r.columns
  paddedA : ∀ pid b, pid<programs r → b≤r.parameters.numKBlocks → ∀ i : FPValue.AIndex r.parameters,
    FPValue.aOffset r.parameters (instanceState r pid s) b i<2^31 ∧
      t.baseA+4*FPValue.aOffset r.parameters (instanceState r pid s) b i<2^64
  paddedB : ∀ pid b, pid<programs r → b≤r.parameters.numKBlocks → ∀ i : FPValue.BIndex r.parameters,
    FPValue.bOffset r.parameters (instanceState r pid s) b i<2^31 ∧
      t.baseB+4*FPValue.bOffset r.parameters (instanceState r pid s) b i<2^64
  paddedC : ∀ pid, pid<programs r → ∀ i : FPValue.CIndex r.parameters,
    FPValue.cOffset r.parameters (instanceState r pid s) i<2^31 ∧
      t.baseC+2*FPValue.cOffset r.parameters (instanceState r pid s) i<2^64
  uniqueStore : ∀ p q, p<programs r → q<programs r → ∀ i j : FPValue.CIndex r.parameters,
    FPValue.active r.parameters (instanceState r p s) i → FPValue.active r.parameters (instanceState r q s) j →
    FPValue.cOffset r.parameters (instanceState r p s) i=FPValue.cOffset r.parameters (instanceState r q s) j → p=q ∧ i=j

theorem source_safety {α : Type} (r : Request) (s : State α) (t : TensorContract r s)
    (h : IndexDomain r) : SourceSafety r s t where
  control := by
    intro pid hp
    have hn := nonempty_program r pid hp
    have hg := grouped_bounds (tileRows r) (tileColumns r) r.config.gm pid r.config.positive.2.2.2 hp
    exact ⟨hn.1,hn.2,hg.1,hg.2.1,hg.2.2,lt_trans hp h.grid,h.group⟩
  groupSubtraction := by
    intro pid hp
    have hg := (grouped_bounds (tileRows r) (tileColumns r) r.config.gm pid r.config.positive.2.2.2 hp).2.1
    unfold tileRow at hg
    omega
  subtraction := source_subtraction r
  loadA := loadA_bound r s t
  loadB := loadB_bound r s t
  store := store_bound r s
  paddedA := by
    intro p b hp hb i
    have he := padded_a_bound r s p b hp hb i
    exact ⟨lt_of_le_of_lt he h.addressA,
      lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 he) _) t.byteA⟩
  paddedB := by
    intro p b hp hb i
    have he := padded_b_bound r s p b hp hb i
    exact ⟨lt_of_le_of_lt he h.addressB,
      lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 he) _) t.byteB⟩
  paddedC := by
    intro p hp i
    have he := padded_c_bound r s p hp i
    exact ⟨lt_of_le_of_lt he h.addressC,
      lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 2 he) _) t.byteC⟩
  uniqueStore := source_store_unique r s

/-- The library frame is the entire allocated return buffer, exactly M*N
half elements; no in-bounds returned coordinate is left unwritten. -/
theorem footprint_iff (r : Request) (region : RegionName) (offset : Nat) :
    Footprint r region offset ↔ region=r.C ∧ offset<r.rows*r.columns := by
  constructor
  · rintro ⟨he,i,hi,j,hj,ho⟩
    exact ⟨he,ho ▸ output_bound r i j hi hj⟩
  · rintro ⟨he,ho⟩
    have hn : 0<r.columns := by nlinarith
    have hi : offset/r.columns<r.rows := (Nat.div_lt_iff_lt_mul hn).mpr ho
    refine ⟨he,offset/r.columns,hi,offset%r.columns,Nat.mod_lt _ hn,?_⟩
    simp only [outputOffset,max_eq_left (by omega : 1≤r.columns)]
    exact (Nat.div_add_mod offset r.columns).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
