import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeWords

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000

structure Control where
  (numM numN perGroup group first size pm pn stop : BitVec 64)

/-- The actual source prelude after scalar ABI conversion and signed-i32 PID
extension. Intermediate additions/products/subtractions are word operations. -/
def control (r : Request) (pid : Nat) : Control :=
  let nm := cdiv (scalar r.rows) r.config.bm
  let nn := cdiv (scalar r.columns) r.config.bn
  let pg := word r.config.gm*nn
  let g := (fromI32 pid).sdiv pg
  let first := g*word r.config.gm
  let size := minimum (nm-first) (word r.config.gm)
  let within := (fromI32 pid).srem pg
  ⟨nm,nn,pg,g,first,size,first+within.srem size,within.sdiv size,
    cdiv (scalar r.inner) r.config.bk⟩

def naturalControl (r : Request) (pid : Nat) : Control :=
  ⟨word (tileRows r),word (tileColumns r),word (r.config.gm*tileColumns r),
    word (pid/(r.config.gm*tileColumns r)),
    word (groupBase (tileRows r) (tileColumns r) r.config.gm pid),
    word (Whole.groupSize (tileRows r) (tileColumns r) r.config.gm pid),
    word (tileRow (tileRows r) (tileColumns r) r.config.gm pid),
    word (tileCol (tileRows r) (tileColumns r) r.config.gm pid),word r.parameters.numKBlocks⟩

theorem config_bound (c : Config) :
    c.bm<2^31 ∧ c.bn<2^31 ∧ c.bk<2^31 ∧ c.gm<2^31 := by cases c <;> decide

def metadata (r : Request) : List Nat :=
  [r.rows,r.columns,r.inner,r.strideAM,r.strideAK,r.strideBK,r.strideBN,max r.columns 1,1]

/-- Every actually passed dimension/stride, including unused singleton or
zero strides, decodes as the intended nonnegative signed-i64 metadata. -/
theorem metadata_exact {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) :
    ∀ n∈metadata r, (scalar n).toInt=(n : Int) := by
  obtain ⟨hm,hn,hk⟩ := dimension_machine r s t pid hp
  have ham := t.strideAM
  have hak := t.strideAK
  have hbk := t.strideBK
  have hbn := t.strideBN
  intro n h
  simp only [metadata,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h
  all_goals subst n; rw [scalar_eq,signed _ (by omega)]

theorem control_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) :
    control r pid=naturalControl r pid := by
  obtain ⟨hm,hn,hk,hgrid,hpg⟩ := control_machine r s t pid hp
  obtain ⟨hr,hc,hpgpos,hspos,hfirst,hsub⟩ := candidate_scalar_control_safe r pid hp
  obtain ⟨hbm,hbn,hbk,hgm⟩ := config_bound r.config
  have htm : tileRows r<2^63 := lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
  have htn : tileColumns r<2^63 := lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
  have hf : pid/(r.config.gm*tileColumns r)*r.config.gm<tileRows r := hfirst
  have hd : pid/(r.config.gm*tileColumns r)<2^63 :=
    lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
  have hs : min (tileRows r-pid/(r.config.gm*tileColumns r)*r.config.gm) r.config.gm<2^63 :=
    lt_of_le_of_lt (min_le_right _ _) (by omega)
  have hw : pid%(r.config.gm*tileColumns r)<2^63 :=
    lt_of_le_of_lt (Nat.mod_le _ _) (by omega)
  simp only [control,scalar_eq,fromI32_eq pid hpid,
    cdiv_eq r.rows r.config.bm r.config.positive.1 hm,
    cdiv_eq r.columns r.config.bn r.config.positive.2.1 hn,
    cdiv_eq r.inner r.config.bk r.config.positive.2.2.1 hk]
  have eqM : (r.rows+r.config.bm-1)/r.config.bm=tileRows r := rfl
  have eqN : (r.columns+r.config.bn-1)/r.config.bn=tileColumns r := rfl
  simp only [eqM,eqN]
  simp only [← BitVec.ofNat_mul]
  rw [sdiv_eq pid (r.config.gm*tileColumns r) (by omega) hpg]
  simp only [← BitVec.ofNat_mul]
  rw [sub_eq _ _ (by omega),minimum_eq _ _ (by omega) (by omega),
    srem_eq pid (r.config.gm*tileColumns r) (by omega) hpg,
    srem_eq _ _ hw hs,sdiv_eq _ _ hw hs]
  simp only [← BitVec.ofNat_add]
  rfl

def row (r : Request) (pid i : Nat) : BitVec 64 :=
  (control r pid).pm*word r.config.bm+fromI32 i
def col (r : Request) (pid j : Nat) : BitVec 64 :=
  (control r pid).pn*word r.config.bn+fromI32 j
def aRow (r : Request) (pid i : Nat) : BitVec 64 := (row r pid i).srem (scalar r.rows)
def bCol (r : Request) (pid j : Nat) : BitVec 64 := (col r pid j).srem (scalar r.columns)
def kcoord (r : Request) (k : BitVec 64) (e : Nat) : BitVec 64 :=
  k*word r.config.bk+fromI32 e

theorem row_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (i : Fin r.config.bm) :
    row r pid i.val=word (FPValue.row r.parameters (instanceState r pid s) i) := by
  rw [row,control_eq r s t pid hp hpid,fromI32_eq i.val (by have := (config_bound r.config).1; omega)]
  simp only [naturalControl,← BitVec.ofNat_mul,← BitVec.ofNat_add,FPValue.row,source_pidM]
  rfl
theorem col_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (j : Fin r.config.bn) :
    col r pid j.val=word (FPValue.col r.parameters (instanceState r pid s) j) := by
  rw [col,control_eq r s t pid hp hpid,fromI32_eq j.val (by have := (config_bound r.config).2.1; omega)]
  simp only [naturalControl,← BitVec.ofNat_mul,← BitVec.ofNat_add,FPValue.col,source_pidN]
  rfl

theorem aRow_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (i : Fin r.config.bm) :
    aRow r pid i.val=word (FPValue.aRow r.parameters (instanceState r pid s) i) := by
  have hn := lane_machine r s t pid hp (i,⟨0,r.config.positive.2.1⟩,PUnit.unit)
  have hm := (dimension_machine r s t pid hp).1
  rw [aRow,row_eq r s t pid hp hpid i,scalar_eq,srem_eq _ _ hn.1 (by omega)]
  rfl
theorem bCol_eq {α : Type} (r : Request) (s : State α) (t : TensorAPI r s)
    (pid : Nat) (hp : pid<programs r) (hpid : pid<2^31) (j : Fin r.config.bn) :
    bCol r pid j.val=word (FPValue.bCol r.parameters (instanceState r pid s) j) := by
  have hn := lane_machine r s t pid hp (⟨0,r.config.positive.1⟩,j,PUnit.unit)
  have hm := (dimension_machine r s t pid hp).2.1
  rw [bCol,col_eq r s t pid hp hpid j,scalar_eq,srem_eq _ _ hn.2 (by omega)]
  rfl
theorem kcoord_eq (r : Request) (k : Nat) (e : Fin r.config.bk) :
    kcoord r (word k) e.val=word (k*r.config.bk+e.val) := by
  rw [kcoord,fromI32_eq e.val (by have := (config_bound r.config).2.2.1; omega)]
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]

#axiomsClean control_eq
#axiomsClean aRow_eq
#axiomsClean bCol_eq
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
