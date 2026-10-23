import bench.tritonbench_g.matmul_leakyrelu.Proofs.WholeTask

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def signedLimit : Nat := 2147483648
def paddedInner (r : Request) : Nat := ((r.inner+31)/32)*32

/-- Public sufficient finite-index domain. Bounds include every padded lane
and the final pointer advance after the last K block, not just active loads.
The usual allocation/byte-pointer contract is separate below. -/
structure IndexDomain (r : Request) : Prop where
  rows : r.rows+32<signedLimit
  columns : r.columns+32<signedLimit
  inner : r.inner+32<signedLimit
  strideAM : r.strideAM<signedLimit
  strideAK : r.strideAK<signedLimit
  strideBK : r.strideBK<signedLimit
  strideBN : r.strideBN<signedLimit
  grid : programs r<signedLimit
  group : 4*tileColumns r<signedLimit
  paddedA : r.rows*r.strideAM+(paddedInner r+32)*r.strideAK<signedLimit
  paddedB : (paddedInner r+32)*r.strideBK+r.columns*r.strideBN<signedLimit
  paddedC : (tileRows r*32)*max r.columns 1+tileColumns r*32<signedLimit

structure TensorMetadata where
  (shape strides : List Nat)
  dtype : FloatDType
  device : String

def byteSpansDisjoint (baseA sizeA baseB sizeB : Nat) : Prop :=
  sizeA=0 ∨ sizeB=0 ∨ baseA+sizeA≤baseB ∨ baseB+sizeB≤baseA

/-- Ordinary successful Torch tensor/allocation contract. Metadata fields are
the observed Python attributes; bases are actual view data pointers and extents
are available element spans from those pointers. C is the result of
torch.empty((M,N),device=a.device,dtype=float16), with unspecified contents.
No callback supplies a kernel result. Regions identify typed views: A/B may
physically alias and their observations must agree; C is physically fresh. -/
structure TensorContract {α : Type} (r : Request) (s : State α) where
  (extentA extentB extentC : Nat)
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
  freshA : r.C≠r.A
  freshB : r.C≠r.B
  physicalFreshA : byteSpansDisjoint baseC (2*extentC) baseA (4*extentA)
  physicalFreshB : byteSpansDisjoint baseC (2*extentC) baseB (4*extentB)
  alignmentA : baseA%4=0
  alignmentB : baseB%4=0
  alignmentC : baseC%2=0
  spanA : baseA+4*extentA≤2^64
  spanB : baseB+4*extentB≤2^64
  spanC : baseC+2*extentC≤2^64
  sameViewOrigin : r.A=r.B → baseA=baseB
  aliasConsistency : ∀ row<r.rows, ∀ k<r.inner, ∀ k'<r.inner, ∀ col<r.columns,
    baseA+4*(row*r.strideAM+k*r.strideAK)=baseB+4*(k'*r.strideBK+col*r.strideBN) →
    s.mem r.A (row*r.strideAM+k*r.strideAK)=s.mem r.B (k'*r.strideBK+col*r.strideBN)
  inputA : ∀ row<r.rows, ∀ k<r.inner,
    row*r.strideAM+k*r.strideAK<extentA
  inputB : ∀ k<r.inner, ∀ col<r.columns,
    k*r.strideBK+col*r.strideBN<extentB
  output : ∀ row<r.rows, ∀ col<r.columns, outputOffset r row col<extentC
  tagA : ∀ row<r.rows, ∀ k<r.inner,
    ∃ x, s.mem r.A (row*r.strideAM+k*r.strideAK)=Cell.mk .fp32 x
  tagB : ∀ k<r.inner, ∀ col<r.columns,
    ∃ x, s.mem r.B (k*r.strideBK+col*r.strideBN)=Cell.mk .fp32 x
  byteA : baseA+4*(r.rows*r.strideAM+(paddedInner r+32)*r.strideAK)<2^64
  byteB : baseB+4*((paddedInner r+32)*r.strideBK+r.columns*r.strideBN)<2^64
  byteC : baseC+2*((tileRows r*32)*max r.columns 1+tileColumns r*32)<2^64

theorem source_control_safe (r : Request) (h : IndexDomain r) (pid : Nat)
    (hp : pid<programs r) :
    0<r.rows ∧ 0<r.columns ∧
    0<groupSize (tileRows r) (tileColumns r) pid ∧
    groupBase (tileRows r) (tileColumns r) pid<tileRows r ∧
    tileRow (tileRows r) (tileColumns r) pid<tileRows r ∧
    tileCol (tileRows r) (tileColumns r) pid<tileColumns r ∧
    pid<signedLimit ∧ 4*tileColumns r<signedLimit ∧
    r.rows+32<signedLimit ∧ r.columns+32<signedLimit ∧ r.inner+32<signedLimit := by
  have hr : 0<r.rows := by
    by_contra hh
    have hz : r.rows=0 := by omega
    simp [programs,tileRows,hz] at hp
  have hc : 0<r.columns := by
    by_contra hh
    have hz : r.columns=0 := by omega
    simp [programs,tileColumns,hz] at hp
  have hb := grouped_bounds (tileRows r) (tileColumns r) pid hp
  have hf : groupBase (tileRows r) (tileColumns r) pid<tileRows r := by
    have hz := Nat.zero_le ((pid%(4*tileColumns r))%groupSize (tileRows r) (tileColumns r) pid)
    unfold tileRow at hb
    omega
  exact ⟨hr,hc,hb.1,hf,hb.2.1,hb.2.2,lt_trans hp h.grid,h.group,h.rows,h.columns,h.inner⟩

theorem source_subtraction_safe (r : Request) (block : Nat) (hb : block<(r.inner+31)/32) :
    block*32<r.inner ∧ (r.inner : Int)-(block*32 : Nat)=(r.inner-block*32 : Nat) := by
  constructor
  · omega
  · omega

theorem source_address_envelopes {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (pid block : Nat) (hp : pid<programs r) (hb : block≤(r.inner+31)/32)
    (i j : Fin 32) :
    FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)≤
      r.rows*r.strideAM+(paddedInner r+32)*r.strideAK ∧
    FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)≤
      (paddedInner r+32)*r.strideBK+r.columns*r.strideBN ∧
    FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)≤
      (tileRows r*32)*max r.columns 1+tileColumns r*32 := by
  have hc := source_control_safe r h pid hp
  have ha : FPValue.aRow r.parameters (instanceState r pid s) i<r.rows :=
    Nat.mod_lt _ hc.1
  have hbcol : FPValue.bCol r.parameters (instanceState r pid s) j<r.columns :=
    Nat.mod_lt _ hc.2.1
  have hki : block*32+i.val≤paddedInner r+32 := by unfold paddedInner; omega
  have hkj : block*32+j.val≤paddedInner r+32 := by unfold paddedInner; omega
  have ham := Nat.mul_le_mul_right r.strideAM (Nat.le_of_lt ha)
  have hak := Nat.mul_le_mul_right r.strideAK hkj
  have hbk := Nat.mul_le_mul_right r.strideBK hki
  have hbn := Nat.mul_le_mul_right r.strideBN (Nat.le_of_lt hbcol)
  have hrow : FPValue.row r.parameters (instanceState r pid s) i<tileRows r*32 := by
    change FPValue.pidM r.parameters (instanceState r pid s)*32+i.val<tileRows r*32
    rw [source_pidM]
    have := (grouped_bounds (tileRows r) (tileColumns r) pid hp).2.1
    omega
  have hcol : FPValue.col r.parameters (instanceState r pid s) j<tileColumns r*32 := by
    change FPValue.pidN r.parameters (instanceState r pid s)*32+j.val<tileColumns r*32
    rw [source_pidN]
    have := (grouped_bounds (tileRows r) (tileColumns r) pid hp).2.2
    omega
  have hcm := Nat.mul_le_mul_right (max r.columns 1) (Nat.le_of_lt hrow)
  refine ⟨?_,?_,?_⟩
  · have he : FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)=
        FPValue.aRow r.parameters (instanceState r pid s) i*r.strideAM+(block*32+j.val)*r.strideAK := by
      simp only [FPValue.aOffset,FPValue.BK,Request.parameters]
      ring
    rw [he]
    exact Nat.add_le_add ham hak
  · have he : FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)=
        (block*32+i.val)*r.strideBK+FPValue.bCol r.parameters (instanceState r pid s) j*r.strideBN := by
      simp only [FPValue.bOffset,FPValue.BK,Request.parameters]
      ring
    rw [he]
    exact Nat.add_le_add hbk hbn
  · rw [source_offset]
    unfold outputOffset
    have hle : max r.columns 1*FPValue.row r.parameters (instanceState r pid s) i+
        FPValue.col r.parameters (instanceState r pid s) j≤
        (tileRows r*32)*max r.columns 1+tileColumns r*32 := by
      rw [Nat.mul_comm (max r.columns 1)]
      exact Nat.add_le_add hcm (Nat.le_of_lt hcol)
    exact hle

theorem source_padded_addresses {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (pid block : Nat) (hp : pid<programs r) (hb : block≤(r.inner+31)/32)
    (i j : Fin 32) :
    FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<signedLimit ∧
    FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<signedLimit ∧
    FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)<signedLimit := by
  have he := source_address_envelopes r h s pid block hp hb i j
  exact ⟨lt_of_le_of_lt he.1 h.paddedA,lt_of_le_of_lt he.2.1 h.paddedB,
    lt_of_le_of_lt he.2.2 h.paddedC⟩

theorem source_byte_addresses {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (t : TensorContract r s) (pid block : Nat)
    (hp : pid<programs r) (hb : block≤(r.inner+31)/32) (i j : Fin 32) :
    t.baseA+4*FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<2^64 ∧
    t.baseB+4*FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<2^64 ∧
    t.baseC+2*FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)<2^64 := by
  have he := source_address_envelopes r h s pid block hp hb i j
  exact ⟨lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 he.1) _) t.byteA,
    lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 4 he.2.1) _) t.byteB,
    lt_of_le_of_lt (Nat.add_le_add_left (Nat.mul_le_mul_left 2 he.2.2) _) t.byteC⟩

theorem source_active_loadA {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (t : TensorContract r s) (pid block : Nat) (hp : pid<programs r)
    (i j : Fin 32) (hmask : j.val<r.inner-block*32) :
    FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<t.extentA := by
  have hc := source_control_safe r h pid hp
  have ha : FPValue.aRow r.parameters (instanceState r pid s) i<r.rows := Nat.mod_lt _ hc.1
  have hk : block*32+j.val<r.inner := by omega
  have he : FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)=
      FPValue.aRow r.parameters (instanceState r pid s) i*r.strideAM+(block*32+j.val)*r.strideAK := by
    simp only [FPValue.aOffset,FPValue.BK,Request.parameters]
    ring
  rw [he]
  exact t.inputA _ ha _ hk

theorem source_active_loadB {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (t : TensorContract r s) (pid block : Nat) (hp : pid<programs r)
    (i j : Fin 32) (hmask : i.val<r.inner-block*32) :
    FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<t.extentB := by
  have hc := source_control_safe r h pid hp
  have hb : FPValue.bCol r.parameters (instanceState r pid s) j<r.columns := Nat.mod_lt _ hc.2.1
  have hk : block*32+i.val<r.inner := by omega
  have he : FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)=
      (block*32+i.val)*r.strideBK+FPValue.bCol r.parameters (instanceState r pid s) j*r.strideBN := by
    simp only [FPValue.bOffset,FPValue.BK,Request.parameters]
    ring
  rw [he]
  exact t.inputB _ hk _ hb

theorem source_active_store {α : Type} (r : Request)
    (s : State α) (t : TensorContract r s) (pid : Nat) (i : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters (instanceState r pid s) i) :
    FPValue.cOffset r.parameters (instanceState r pid s) i<t.extentC := by
  rw [source_offset]
  exact t.output _ hi.1 _ hi.2

theorem source_active_byte_spans {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (t : TensorContract r s) (pid block : Nat) (hp : pid<programs r)
    (i j : Fin 32) :
    (j.val<r.inner-block*32 →
      t.baseA+4*FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)+4≤2^64) ∧
    (i.val<r.inner-block*32 →
      t.baseB+4*FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)+4≤2^64) ∧
    (FPValue.active r.parameters (instanceState r pid s) (i,j,PUnit.unit) →
      t.baseC+2*FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)+2≤2^64) := by
  refine ⟨?_,?_,?_⟩
  · intro hm
    have ho := source_active_loadA r h s t pid block hp i j hm
    have hs := t.spanA
    omega
  · intro hm
    have ho := source_active_loadB r h s t pid block hp i j hm
    have hs := t.spanB
    omega
  · intro hm
    have ho := source_active_store r s t pid (i,j,PUnit.unit) hm
    have hs := t.spanC
    omega

/-- Source-address conclusions, not a primitive law or caller-supplied outcome.
Padded pointer formation is distinct from active memory dereference. -/
structure SourceSafety {α : Type} (r : Request) (s : State α) (t : TensorContract r s) : Prop where
  control : ∀ pid, pid<programs r →
    0<r.rows ∧ 0<r.columns ∧ 0<groupSize (tileRows r) (tileColumns r) pid ∧
    groupBase (tileRows r) (tileColumns r) pid<tileRows r ∧
    tileRow (tileRows r) (tileColumns r) pid<tileRows r ∧
    tileCol (tileRows r) (tileColumns r) pid<tileColumns r ∧
    pid<signedLimit ∧ 4*tileColumns r<signedLimit ∧
    r.rows+32<signedLimit ∧ r.columns+32<signedLimit ∧ r.inner+32<signedLimit
  subtraction : ∀ block, block<(r.inner+31)/32 →
    block*32<r.inner ∧ (r.inner : Int)-(block*32 : Nat)=(r.inner-block*32 : Nat)
  padded : ∀ pid block, pid<programs r → block≤(r.inner+31)/32 → ∀ i j : Fin 32,
    FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<signedLimit ∧
    FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<signedLimit ∧
    FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)<signedLimit
  bytes : ∀ pid block, pid<programs r → block≤(r.inner+31)/32 → ∀ i j : Fin 32,
    t.baseA+4*FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<2^64 ∧
    t.baseB+4*FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<2^64 ∧
    t.baseC+2*FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)<2^64
  loadA : ∀ pid block, pid<programs r → ∀ i j : Fin 32, j.val<r.inner-block*32 →
    FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<t.extentA
  loadB : ∀ pid block, pid<programs r → ∀ i j : Fin 32, i.val<r.inner-block*32 →
    FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)<t.extentB
  storeC : ∀ pid (i : FPValue.CIndex r.parameters),
    FPValue.active r.parameters (instanceState r pid s) i →
    FPValue.cOffset r.parameters (instanceState r pid s) i<t.extentC
  activeBytes : ∀ pid block, pid<programs r → ∀ i j : Fin 32,
    (j.val<r.inner-block*32 →
      t.baseA+4*FPValue.aOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)+4≤2^64) ∧
    (i.val<r.inner-block*32 →
      t.baseB+4*FPValue.bOffset r.parameters (instanceState r pid s) block (i,j,PUnit.unit)+4≤2^64) ∧
    (FPValue.active r.parameters (instanceState r pid s) (i,j,PUnit.unit) →
      t.baseC+2*FPValue.cOffset r.parameters (instanceState r pid s) (i,j,PUnit.unit)+2≤2^64)
  uniqueStore : ∀ p q, p<programs r → q<programs r → ∀ i j : FPValue.CIndex r.parameters,
    FPValue.active r.parameters (instanceState r p s) i →
    FPValue.active r.parameters (instanceState r q s) j →
    FPValue.cOffset r.parameters (instanceState r p s) i=
      FPValue.cOffset r.parameters (instanceState r q s) j → p=q ∧ i=j

theorem source_safety {α : Type} (r : Request) (h : IndexDomain r)
    (s : State α) (t : TensorContract r s) : SourceSafety r s t where
  control := source_control_safe r h
  subtraction := source_subtraction_safe r
  padded := source_padded_addresses r h s
  bytes := source_byte_addresses r h s t
  loadA := source_active_loadA r h s t
  loadB := source_active_loadB r h s t
  storeC := source_active_store r s t
  activeBytes := source_active_byte_spans r h s t
  uniqueStore := source_store_unique r s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
