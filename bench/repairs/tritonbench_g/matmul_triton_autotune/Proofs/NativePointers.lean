import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeControl

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000

def aOff (r : Request) (pid : Nat) (k : BitVec 64) (i e : Nat) : BitVec 64 :=
  aRow r pid i*scalar r.strideAM + safe (scalar r.inner) (kcoord r k e)*scalar r.strideAK
def bOff (r : Request) (pid : Nat) (k : BitVec 64) (e j : Nat) : BitVec 64 :=
  safe (scalar r.inner) (kcoord r k e)*scalar r.strideBK + bCol r pid j*scalar r.strideBN
def cRowOff (r : Request) (pid i : Nat) : BitVec 64 :=
  scalar (max r.columns 1)*safe (scalar r.rows) (row r pid i)
def cColOff (r : Request) (pid j : Nat) : BitVec 64 :=
  scalar 1*safe (scalar r.columns) (col r pid j)
def cOff (r : Request) (pid i j : Nat) : BitVec 64 := cRowOff r pid i+cColOff r pid j
def kMask (r : Request) (k : BitVec 64) (e : Nat) : Bool :=
  (kcoord r k e).slt (scalar r.inner)
def cMask (r : Request) (pid i j : Nat) : Bool :=
  (row r pid i).slt (scalar r.rows) && (col r pid j).slt (scalar r.columns)
def aPtr (r : Request) (base pid : Nat) (k : BitVec 64) (i e : Nat) := ptr base 4 (aOff r pid k i e)
def bPtr (r : Request) (base pid : Nat) (k : BitVec 64) (e j : Nat) := ptr base 4 (bOff r pid k e j)
/-- Preserve the source's two successive output-pointer additions. -/
def cPtr (r : Request) (base pid i j : Nat) : BitVec 64 :=
  ptr base 2 (cRowOff r pid i) + BitVec.ofInt 64 (2*(cColOff r pid j).toInt)
theorem cPtr_eq (r : Request) (base pid i j : Nat) :
    cPtr r base pid i j=ptr base 2 (cOff r pid i j) := by
  simp only [cPtr,ptr,cOff,BitVec.ofInt_mul,BitVec.ofInt_toInt]
  change word base+word 2*cRowOff r pid i+word 2*cColOff r pid j=
    word base+word 2*(cRowOff r pid i+cColOff r pid j)
  rw [BitVec.mul_add,BitVec.add_assoc]

theorem aOff_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (i : Fin r.config.bm) (e : Fin r.config.bk) :
    aOff r pid (word k) i.val e.val=
      word (safeAOffset r (FPValue.aRow r.parameters (instanceState r pid s) i) (k*r.config.bk+e.val)) := by
  have hd := (dimension_machine r s t pid hp).2.2
  have he := inner_lane_machine r s t pid k e.val hp hk e.isLt
  rw [aOff,aRow_eq r s t pid hp hpid i,kcoord_eq r k e]
  simp only [scalar_eq,safe_eq r.inner _ (by omega) he,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  rfl
theorem bOff_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (e : Fin r.config.bk) (j : Fin r.config.bn) :
    bOff r pid (word k) e.val j.val=
      word (safeBOffset r (k*r.config.bk+e.val) (FPValue.bCol r.parameters (instanceState r pid s) j)) := by
  have hd := (dimension_machine r s t pid hp).2.2
  have he := inner_lane_machine r s t pid k e.val hp hk e.isLt
  rw [bOff,bCol_eq r s t pid hp hpid j,kcoord_eq r k e]
  simp only [scalar_eq,safe_eq r.inner _ (by omega) he,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  rfl
theorem cOff_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (i : Fin r.config.bm) (j : Fin r.config.bn) :
    cOff r pid i.val j.val=word (safeCOffset r
      (FPValue.row r.parameters (instanceState r pid s) i)
      (FPValue.col r.parameters (instanceState r pid s) j)) := by
  obtain ⟨hm,hn,hk⟩ := dimension_machine r s t pid hp
  obtain ⟨hi,hj⟩ := lane_machine r s t pid hp (i,j,PUnit.unit)
  simp only [cOff,cRowOff,cColOff,row_eq r s t pid hp hpid i,col_eq r s t pid hp hpid j,
    scalar_eq,safe_eq _ _ (by omega : r.rows<2^63) hi,
    safe_eq _ _ (by omega : r.columns<2^63) hj,← BitVec.ofNat_mul,← BitVec.ofNat_add,Nat.one_mul]
  rfl

theorem kMask_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hk : k<r.parameters.numKBlocks) (e : Fin r.config.bk) :
    kMask r (word k) e.val=decide (k*r.config.bk+e.val<r.inner) := by
  have hd := (dimension_machine r s t pid hp).2.2
  rw [kMask,kcoord_eq r k e,scalar_eq,
    slt_eq _ _ (inner_lane_machine r s t pid k e.val hp hk e.isLt) (by omega)]
theorem cMask_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (i : FPValue.CIndex r.parameters) :
    cMask r pid i.1.val i.2.1.val=decide (FPValue.active r.parameters (instanceState r pid s) i) := by
  obtain ⟨hm,hn,hk⟩ := dimension_machine r s t pid hp
  obtain ⟨hi,hj⟩ := lane_machine r s t pid hp i
  rw [cMask,row_eq r s t pid hp hpid i.1,col_eq r s t pid hp hpid i.2.1,
    scalar_eq,scalar_eq,slt_eq _ _ hi (by omega),slt_eq _ _ hj (by omega)]
  simp [FPValue.active,Request.parameters]
  rfl

/-- Physical interval validity for every formed pointer, masked or active. -/
def FullBytes (base bytes extent : Nat) (address : BitVec 64) : Prop :=
  ∀ byte, byte<bytes → base≤address.toNat+byte ∧
    address.toNat+byte<base+bytes*extent ∧ address.toNat+byte<2^64

theorem a_access {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (i : Fin r.config.bm) (e : Fin r.config.bk) :
    let off := safeAOffset r (FPValue.aRow r.parameters (instanceState r pid s) i) (k*r.config.bk+e.val)
    (aOff r pid (word k) i.val e.val).toInt=(off : Int) ∧
    FullBytes t.baseA 4 t.extentA (aPtr r t.baseA pid (word k) i.val e.val) ∧
    ((aPtr r t.baseA pid (word k) i.val e.val).toNat-t.baseA)/4=off := by
  have hm := (nonempty_program r pid hp).1
  have hrow : FPValue.aRow r.parameters (instanceState r pid s) i<r.rows := Nat.mod_lt _ hm
  have hK : 0<r.inner := by have := (source_subtraction r k hk).1; omega
  have hbound := safeA_bound r s t _ (k*r.config.bk+e.val) hrow hK
  have hmach := safeA_machine r s t _ (k*r.config.bk+e.val) hrow hK
  dsimp only
  rw [aOff_eq r s t pid k hp hpid hk i e]
  refine ⟨signed _ hmach.1,?_,?_⟩
  all_goals rw [aPtr,aOff_eq r s t pid k hp hpid hk i e]
  · exact (bytes_exact _ 4 _ _ (by decide) t.spanA hbound).2.1
  · exact (bytes_exact _ 4 _ _ (by decide) t.spanA hbound).2.2
theorem b_access {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid k : Nat) (hp : pid<programs r) (hpid : pid<2^31) (hk : k<r.parameters.numKBlocks)
    (e : Fin r.config.bk) (j : Fin r.config.bn) :
    let off := safeBOffset r (k*r.config.bk+e.val) (FPValue.bCol r.parameters (instanceState r pid s) j)
    (bOff r pid (word k) e.val j.val).toInt=(off : Int) ∧
    FullBytes t.baseB 4 t.extentB (bPtr r t.baseB pid (word k) e.val j.val) ∧
    ((bPtr r t.baseB pid (word k) e.val j.val).toNat-t.baseB)/4=off := by
  have hn := (nonempty_program r pid hp).2
  have hcol : FPValue.bCol r.parameters (instanceState r pid s) j<r.columns := Nat.mod_lt _ hn
  have hK : 0<r.inner := by have := (source_subtraction r k hk).1; omega
  have hbound := safeB_bound r s t (k*r.config.bk+e.val) _ hcol hK
  have hmach := safeB_machine r s t (k*r.config.bk+e.val) _ hcol hK
  dsimp only
  rw [bOff_eq r s t pid k hp hpid hk e j]
  refine ⟨signed _ hmach.1,?_,?_⟩
  all_goals rw [bPtr,bOff_eq r s t pid k hp hpid hk e j]
  · exact (bytes_exact _ 4 _ _ (by decide) t.spanB hbound).2.1
  · exact (bytes_exact _ 4 _ _ (by decide) t.spanB hbound).2.2
theorem c_access {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (i : Fin r.config.bm) (j : Fin r.config.bn) :
    let off := safeCOffset r (FPValue.row r.parameters (instanceState r pid s) i)
      (FPValue.col r.parameters (instanceState r pid s) j)
    (cOff r pid i.val j.val).toInt=(off : Int) ∧
    FullBytes t.baseC 2 (r.rows*r.columns) (cPtr r t.baseC pid i.val j.val) ∧
    ((cPtr r t.baseC pid i.val j.val).toNat-t.baseC)/2=off := by
  obtain ⟨hm,hn⟩ := nonempty_program r pid hp
  have hbound := safeC_bound r (FPValue.row r.parameters (instanceState r pid s) i)
    (FPValue.col r.parameters (instanceState r pid s) j) hm hn
  have hmach := safeC_machine r s t (FPValue.row r.parameters (instanceState r pid s) i)
    (FPValue.col r.parameters (instanceState r pid s) j) hm hn
  dsimp only
  rw [cOff_eq r s t pid hp hpid i j]
  refine ⟨signed _ hmach.1,?_,?_⟩
  all_goals rw [cPtr_eq,cOff_eq r s t pid hp hpid i j]
  · exact (bytes_exact _ 2 _ _ (by decide) t.spanC hbound).2.1
  · exact (bytes_exact _ 2 _ _ (by decide) t.spanC hbound).2.2

#axiomsClean a_access
#axiomsClean b_access
#axiomsClean c_access
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
