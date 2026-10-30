import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineBounds

namespace VeriTile.Bench.Repairs.ParallelAttention.SourceGrid
open VeriTile.Bench.Repairs.ParallelAttention
set_option maxHeartbeats 8000000
noncomputable section

/-- Source program IDs are (axis 0, axis 1, axis 2). -/
abbrev Program := Nat × Nat × Nat
abbrev Lane := Program × (Nat × Nat)

def rowBlock : OutputKind → Nat
  | .forward | .normalizer => 128
  | .query | .key | .value => 64

def featureBlock (s : Shape) : OutputKind → Nat
  | .forward | .value => s.BV
  | .normalizer => 1
  | .query | .key => s.BK

/-- The actual three-dimensional source grid, without a driver product bound. -/
def programs (s : Shape) (L : Nat) : List Program :=
  (List.range (blocks s.K s.BK * blocks s.V s.BV)).product
    ((List.range (blocks s.T L)).product (List.range (s.B*s.H)))

def lanes (s : Shape) (kind : OutputKind) : List Lane :=
  (programs s (rowBlock kind)).product
    ((List.range (rowBlock kind)).product (List.range (featureBlock s kind)))

def keyID (s : Shape) (p : Program) := p.1 / blocks s.V s.BV
def valueID (s : Shape) (p : Program) := p.1 % blocks s.V s.BV
def time (kind : OutputKind) (l : Lane) := l.1.2.1 * rowBlock kind + l.2.1

def feature (s : Shape) (kind : OutputKind) (l : Lane) :=
  match kind with
  | .forward | .value => valueID s l.1 * s.BV + l.2.2
  | .normalizer => l.2.2
  | .query | .key => keyID s l.1 * s.BK + l.2.2

def shard (s : Shape) (kind : OutputKind) (l : Lane) :=
  match kind with
  | .forward | .normalizer | .value => keyID s l.1
  | .query | .key => valueID s l.1

/-- Block-pointer boundary checks, plus the source's unique Z writer guard.
The normalizer has a singleton logical feature axis rather than BV copies. -/
def storeMask (s : Shape) (kind : OutputKind) (l : Lane) : Prop :=
  time kind l < s.T ∧ feature s kind l < kind.width s ∧
    (kind = .normalizer → valueID s l.1 = 0)

instance (s : Shape) (kind : OutputKind) (l : Lane) : Decidable (storeMask s kind l) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- Canonical source offset, including the actual scratch shard. -/
def index (s : Shape) (kind : OutputKind) (l : Lane) :=
  shard s kind l * kind.cells s +
    dense s.T (kind.width s) l.1.2.2 (time kind l) (feature s kind l)

def activeLanes (s : Shape) (kind : OutputKind) : List Lane :=
  (lanes s kind).filter (fun l => decide (storeMask s kind l))

def stores (s : Shape) (kind : OutputKind) : List Event :=
  (activeLanes s kind).map (fun l => (kind,index s kind l))

def Domain (s : Shape) (kind : OutputKind) (l : Lane) : Prop :=
  l.1.1 < blocks s.K s.BK * blocks s.V s.BV ∧
  l.1.2.1 < blocks s.T (rowBlock kind) ∧ l.1.2.2 < s.B*s.H ∧
  l.2.1 < rowBlock kind ∧ l.2.2 < featureBlock s kind

theorem lane_member (s : Shape) (kind : OutputKind) (l : Lane) :
    l ∈ activeLanes s kind ↔ Domain s kind l ∧ storeMask s kind l := by
  rcases l with ⟨⟨x,c,bh⟩,r,f⟩
  simp [activeLanes,lanes,programs,Domain,and_assoc]

theorem lanes_nodup (s : Shape) (kind : OutputKind) : (activeLanes s kind).Nodup := by
  apply List.Nodup.filter
  exact (List.nodup_range |>.product ((List.nodup_range).product (List.nodup_range))).product
    ((List.nodup_range).product (List.nodup_range))

theorem program_decode (s : Shape) (h : ShapeContract s) (kind : OutputKind) (l : Lane)
    (hd : Domain s kind l) :
    l.1.1 < blocks s.V s.BV ∧ keyID s l.1 = 0 ∧ valueID s l.1 = l.1.1 := by
  have hx := hd.1
  rw [Machine.key_shards s h,Nat.one_mul] at hx
  exact ⟨hx,Nat.div_eq_of_lt hx,Nat.mod_eq_of_lt hx⟩

theorem active_index_bound (s : Shape) (h : ShapeContract s) (kind : OutputKind) (l : Lane)
    (hl : l ∈ activeLanes s kind) : index s kind l < kind.count s := by
  obtain ⟨hd,hm⟩ := (lane_member s kind l).mp hl
  obtain ⟨hx,hk,hv⟩ := program_decode s h kind l hd
  have hs : shard s kind l < kind.shards s := by
    cases kind <;> simp only [shard,OutputKind.shards,hk,hv] <;> omega
  have hb := dense_bound s.T (kind.width s) (s.B*s.H) l.1.2.2
    (time kind l) (feature s kind l) hd.2.2.1 hm.1 hm.2.1
  change _ < kind.cells s at hb
  change _ < kind.shards s * kind.cells s
  unfold index
  nlinarith

/-- The source mask and grid are enough to prove unique store ownership. -/
theorem active_index_injective (s : Shape) (h : ShapeContract s) (kind : OutputKind)
    (l r : Lane) (hl : l ∈ activeLanes s kind) (hr : r ∈ activeLanes s kind)
    (eq : index s kind l = index s kind r) : l = r := by
  obtain ⟨ld,lm⟩ := (lane_member s kind l).mp hl
  obtain ⟨rd,rm⟩ := (lane_member s kind r).mp hr
  obtain ⟨lx,lk,lv⟩ := program_decode s h kind l ld
  obtain ⟨rx,rk,rv⟩ := program_decode s h kind r rd
  obtain ⟨es,eb,et,ef⟩ := scratch_unique (s.B*s.H) s.T (kind.width s)
    (shard s kind l) (shard s kind r) l.1.2.2 r.1.2.2
    (time kind l) (time kind r) (feature s kind l) (feature s kind r)
    ld.2.2.1 rd.2.2.1 lm.1 rm.1 lm.2.1 rm.2.1 eq
  obtain ⟨ec,er⟩ := tile_injective (rowBlock kind) l.1.2.1 r.1.2.1 l.2.1 r.2.1
    ld.2.2.2.1 rd.2.2.2.1 et
  have exf : l.1.1 = r.1.1 ∧ l.2.2 = r.2.2 := by
    cases kind with
    | forward | value =>
      simp only [feature,lv,rv] at ef
      exact tile_injective s.BV _ _ _ _ ld.2.2.2.2 rd.2.2.2.2 ef
    | normalizer =>
      have el := lm.2.2 rfl
      have er := rm.2.2 rfl
      simp only [lv] at el
      simp only [rv] at er
      exact ⟨el.trans er.symm,ef⟩
    | query | key =>
      simp only [shard,lv,rv] at es
      simp only [feature,lk,rk,Nat.zero_mul,Nat.zero_add] at ef
      exact ⟨es,ef⟩
  apply Prod.ext
  · exact Prod.ext exf.1 (Prod.ext ec eb)
  · exact Prod.ext er exf.2

theorem split_index (N D i : Nat) (hi : i < N*D) :
    i/D < N ∧ i%D < D ∧ i/D*D+i%D = i := by
  have hd : 0 < D := by
    by_contra h
    have hz : D=0 := by omega
    simp [hz] at hi
  exact ⟨(Nat.div_lt_iff_lt_mul hd).2 hi,Nat.mod_lt _ hd,
    by simpa [Nat.mul_comm] using Nat.div_add_mod i D⟩

/-- Range membership is decomposed into bounded source tensor coordinates. -/
theorem unpack_index (N M T D i : Nat) (hi : i < N*(M*T*D)) :
    ∃ sh bh t e, sh < N ∧ bh < M ∧ t < T ∧ e < D ∧
      i = sh*(M*T*D)+dense T D bh t e := by
  have hD : i < (N*M*T)*D := by simpa [Nat.mul_assoc] using hi
  obtain ⟨hqd,hed,ed⟩ := split_index (N*M*T) D i hD
  obtain ⟨hqt,het,et⟩ := split_index (N*M) T (i/D) hqd
  obtain ⟨hqm,hem,em⟩ := split_index N M (i/D/T) hqt
  refine ⟨i/D/T/M,i/D/T%M,i/D%T,i%D,hqm,hem,het,hed,?_⟩
  calc
    i = (i/D)*D+i%D := ed.symm
    _ = ((i/D/T/M*M+i/D/T%M)*T+i/D%T)*D+i%D := by rw [em,et]
    _ = _ := by unfold dense;ring

def owner (s : Shape) (kind : OutputKind) (sh bh t e : Nat) : Lane :=
  match kind with
  | .forward | .value => ((e/s.BV,t/rowBlock kind,bh),t%rowBlock kind,e%s.BV)
  | .normalizer => ((0,t/rowBlock kind,bh),t%rowBlock kind,0)
  | .query | .key => ((sh,t/rowBlock kind,bh),t%rowBlock kind,e)

theorem owner_active (s : Shape) (h : ShapeContract s) (kind : OutputKind)
    (sh bh t e : Nat) (hs : sh < kind.shards s) (hb : bh < s.B*s.H)
    (ht : t < s.T) (he : e < kind.width s) :
    owner s kind sh bh t e ∈ activeLanes s kind ∧
      index s kind (owner s kind sh bh t e) = sh*kind.cells s+dense s.T (kind.width s) bh t e := by
  have hp : 0 < rowBlock kind := by cases kind <;> decide
  obtain ⟨hc,hr,ert⟩ := tile_owner s.T (rowBlock kind) t hp ht
  have hvp := h.value_tile_positive s
  have hnv : 0 < blocks s.V s.BV := Machine.shards_positive s h
  have kfit := h.key_fits s
  have keyone := Machine.key_shards s h
  cases kind with
  | forward | value =>
    have hsh : sh=0 := by simpa only [OutputKind.shards,Nat.lt_one_iff] using hs
    obtain ⟨hev,her,ere⟩ := tile_owner s.V s.BV e hvp he
    have ek : e/s.BV/blocks s.V s.BV=0 := Nat.div_eq_of_lt hev
    have ev : e/s.BV%blocks s.V s.BV=e/s.BV := Nat.mod_eq_of_lt hev
    constructor
    · apply (lane_member _ _ _).mpr
      simp only [owner,Domain,keyone,Nat.one_mul,storeMask,time,feature,valueID,
        ev,OutputKind.width,ert,ere,featureBlock]
      exact ⟨⟨hev,hc,hb,hr,her⟩,ht,he,by intro hn;cases hn⟩
    · simp only [index,owner,shard,keyID,valueID,feature,time,
        ek,ev,ert,ere,hsh,Nat.zero_mul,Nat.zero_add]
  | normalizer =>
    have hsh : sh=0 := by simpa only [OutputKind.shards,Nat.lt_one_iff] using hs
    have hez : e=0 := by simpa only [OutputKind.width,Nat.lt_one_iff] using he
    constructor
    · apply (lane_member _ _ _).mpr
      simp only [owner,Domain,keyone,Nat.one_mul,storeMask,time,feature,valueID,
        Nat.zero_mod,OutputKind.width,ert,featureBlock]
      exact ⟨⟨hnv,hc,hb,hr,by decide⟩,ht,by decide,by intro _;trivial⟩
    · simp [index,owner,shard,keyID,feature,time,ert,hsh,hez]
  | query | key =>
    have ek : sh/blocks s.V s.BV=0 := Nat.div_eq_of_lt hs
    have ev : sh%blocks s.V s.BV=sh := Nat.mod_eq_of_lt hs
    have hef : e < s.BK := lt_of_lt_of_le he kfit
    constructor
    · apply (lane_member _ _ _).mpr
      simp only [owner,Domain,keyone,Nat.one_mul,storeMask,time,feature,keyID,valueID,
        ek,ev,OutputKind.width,ert,featureBlock,Nat.zero_mul,Nat.zero_add]
      exact ⟨⟨hs,hc,hb,hr,hef⟩,ht,he,by intro hn;cases hn⟩
    · simp only [index,owner,shard,keyID,valueID,feature,time,
        ek,ev,ert,Nat.zero_mul,Nat.zero_add]

theorem store_coverage (s : Shape) (h : ShapeContract s) (kind : OutputKind) :
    (stores s kind).Perm ((List.range (kind.count s)).map (fun i => (kind,i))) := by
  have hn : (stores s kind).Nodup := by
    apply List.Nodup.map_on _ (lanes_nodup s kind)
    intro l hl r hr eq
    exact active_index_injective s h kind l r hl hr (congrArg Prod.snd eq)
  apply (List.perm_ext_iff_of_nodup hn ((List.nodup_range).map
    (by intro i j eq;exact congrArg Prod.snd eq))).mpr
  intro event
  constructor
  · rintro he
    obtain ⟨l,hl,rfl⟩ := List.mem_map.mp he
    exact List.mem_map.mpr ⟨index s kind l,List.mem_range.mpr (active_index_bound s h kind l hl),rfl⟩
  · rintro he
    obtain ⟨i,hi,rfl⟩ := List.mem_map.mp he
    obtain ⟨sh,bh,t,e,hs,hb,ht,he,eq⟩ := unpack_index (kind.shards s) (s.B*s.H)
      s.T (kind.width s) i (List.mem_range.mp hi)
    obtain ⟨ho,hei⟩ := owner_active s h kind sh bh t e hs hb ht he
    exact List.mem_map.mpr ⟨owner s kind sh bh t e,ho,by rw [hei];exact congrArg (fun j => (kind,j)) eq.symm⟩

/-- These are the source's stores inside one launch program, after its masks. -/
def programStores (s : Shape) (kind : OutputKind) (p : Program) : List Event :=
  ((((List.range (rowBlock kind)).product (List.range (featureBlock s kind))).map
      (fun lane => (p,lane))).filter (fun l => decide (storeMask s kind l))).map
    (fun l => (kind,index s kind l))

theorem stores_by_program (s : Shape) (kind : OutputKind) :
    stores s kind = (programs s (rowBlock kind)).flatMap (programStores s kind) := by
  unfold stores activeLanes lanes
  rw [List.product,List.filter_flatMap,List.map_flatMap]
  rfl

/-- Program-major launch enumeration. The source store order is O then Z,
or DQ then DK then DV. Per-program lane order is a finite serialization. -/
def gridEvents (s : Shape) (L : Nat) (kinds : List OutputKind) : List Event :=
  (programs s L).flatMap (fun p => kinds.flatMap (fun kind => programStores s kind p))

theorem swap_flatMap (as : List α) (bs : List β) (f : α → β → List γ) :
    (as.flatMap (fun a => bs.flatMap (f a))).Perm
      (bs.flatMap (fun b => as.flatMap (fun a => f a b))) := by
  induction bs with
  | nil => simp
  | cons b bs ih =>
    simp only [List.flatMap_cons]
    exact (List.flatMap_append_perm as (fun a => f a b) (fun a => bs.flatMap (f a))).symm.trans
      ((List.Perm.refl _).append ih)

/-- Coverage is proved from the source grid and masks, with no event-completeness premise. -/
theorem grid_coverage (s : Shape) (h : ShapeContract s) (L : Nat) (kinds : List OutputKind)
    (commonRows : ∀ kind ∈ kinds,rowBlock kind = L) :
    (gridEvents s L kinds).Perm (events s true kinds) := by
  apply (swap_flatMap (programs s L) kinds (fun p kind => programStores s kind p)).trans
  apply List.Perm.flatMap_left
  intro kind hk
  rw [← commonRows kind hk,← stores_by_program]
  exact store_coverage s h kind

def forwardEvents (s : Shape) := gridEvents s 128 forwardKinds
def backwardEvents (s : Shape) := gridEvents s 64 backwardKinds

theorem forward_coverage (s : Shape) (h : ShapeContract s) :
    (forwardEvents s).Perm (events s true forwardKinds) := by
  apply grid_coverage s h
  intro kind hk
  simp [forwardKinds] at hk
  rcases hk with rfl|rfl <;> rfl

theorem backward_coverage (s : Shape) (h : ShapeContract s) :
    (backwardEvents s).Perm (events s true backwardKinds) := by
  apply grid_coverage s h
  intro kind hk
  simp [backwardKinds] at hk
  rcases hk with rfl|rfl|rfl <;> rfl

theorem normalizer_writer (s : Shape) (h : ShapeContract s) (l : Lane)
    (hl : l ∈ activeLanes s .normalizer) : l.1.1 = 0 ∧ l.2.2 = 0 := by
  obtain ⟨hd,hm⟩ := (lane_member s .normalizer l).mp hl
  have decoded := program_decode s h .normalizer l hd
  have writer := hm.2.2 rfl
  rw [decoded.2.2] at writer
  exact ⟨writer,Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hd.2.2.2.2)⟩

/-- The enumerated event index is the source pointer's canonical dense offset. -/
theorem source_address (s : Shape) (kind : OutputKind) (l : Lane) :
    (l.1.2.2+s.B*s.H*shard s kind l)*(s.T*kind.width s) +
      time kind l*kind.width s+feature s kind l = index s kind l := by
  unfold index OutputKind.cells dense
  ring

theorem programs_empty (s : Shape) (L : Nat) (hL : 0 < L) (h : Machine.Empty s) :
    programs s L = [] := by
  rcases h with h|h|h
  · simp [programs,h,List.product]
  · simp [programs,h,List.product]
  · have hz : blocks s.T L = 0 := by
      unfold blocks
      rw [h,Nat.zero_add]
      exact Nat.div_eq_of_lt (by omega)
    simp [programs,hz,List.product]

theorem grid_empty (s : Shape) (L : Nat) (hL : 0 < L) (kinds : List OutputKind)
    (h : Machine.Empty s) : gridEvents s L kinds = [] := by
  simp [gridEvents,programs_empty s L hL h]

/-- A scheduler only reorders the concrete source enumeration. Memory freshness,
nonoverlap, and host reduction syntax remain explicit independent obligations. -/
structure GridLayout (s : Shape) (b : Bases) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (L : Nat) (kinds : List OutputKind)
    (ks hs : List Event) : Prop where
  commonRows : ∀ kind ∈ kinds,rowBlock kind = L
  sourceSchedule : ks.Perm (gridEvents s L kinds)
  hostCoverage : hs.Perm (events s false kinds)
  kernelFresh : ∀ e ∈ ks,tensorAddress scratch e ∉ inputCells s b
  kernelUnique : ∀ e ∈ ks,∀ f ∈ ks,tensorAddress scratch e = tensorAddress scratch f → e = f
  hostFresh : ∀ e ∈ hs,tensorAddress output e ∉ allScratchCells s scratch kinds
  hostUnique : ∀ e ∈ hs,∀ f ∈ hs,tensorAddress output e = tensorAddress output f → e = f
  hostTrees : ∀ kind ∈ kinds,(trees kind).Covers (kind.shards s)

def GridLayout.toRunLayout (s : Shape) (h : ShapeContract s) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (L : Nat) (kinds : List OutputKind) (ks hs : List Event)
    (layout : GridLayout s b trees scratch output L kinds ks hs) :
    RunLayout s b trees scratch output kinds ks hs where
  kernelCoverage := layout.sourceSchedule.trans (grid_coverage s h L kinds layout.commonRows)
  hostCoverage := layout.hostCoverage
  kernelFresh := layout.kernelFresh
  kernelUnique := layout.kernelUnique
  hostFresh := layout.hostFresh
  hostUnique := layout.hostUnique
  hostTrees := layout.hostTrees

#axiomsClean lane_member
#axiomsClean program_decode
#axiomsClean active_index_bound
#axiomsClean active_index_injective
#axiomsClean unpack_index
#axiomsClean owner_active
#axiomsClean store_coverage
#axiomsClean stores_by_program
#axiomsClean grid_coverage
#axiomsClean forward_coverage
#axiomsClean backward_coverage
#axiomsClean normalizer_writer
#axiomsClean source_address
#axiomsClean grid_empty
#axiomsClean GridLayout.toRunLayout
#specNonCircular stores avoiding [events,combinedRun,wordReference]
#specNonCircular gridEvents avoiding [events,combinedRun,wordReference]
end
end VeriTile.Bench.Repairs.ParallelAttention.SourceGrid
