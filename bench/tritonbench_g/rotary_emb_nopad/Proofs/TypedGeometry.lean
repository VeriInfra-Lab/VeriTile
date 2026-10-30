import bench.tritonbench_g.rotary_emb_nopad.Proofs.Model
import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedDefinitions
import bench.tritonbench_g.rotary_emb_nopad.Proofs.ABI

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def uLayout (r : Request) : Kernels.UncachedLayout where
  q := "Q"
  k := "K"
  cos := "Cos"
  sin := "Sin"
  q_token_stride := r.qHeads*r.width
  q_head_stride := r.width
  k_token_stride := r.kHeads*r.width
  k_head_stride := r.width
  head_dim_stride := 1
  cos_token_stride := r.cosWidth
  cos_stride := 1
  sin_token_stride := r.sinWidth
  sin_stride := 1
  q_total_tokens := r.tokens
  Q_HEAD_NUM := r.qHeads
  KV_GROUP_NUM := r.groups
  HEAD_DIM := r.width
  BLOCK_PAIRS := r.blockPairs
  BLOCK_TOKENS := 4

def cLayout (r : Request) : Kernels.CachedLayout where
  q := "Q"
  k := "K"
  cos := "Cos"
  sin := "Sin"
  kv_cache := "Cache"
  BLOCK_TABLES := "Tables"
  context_lengths := "Lengths"
  q_token_stride := r.qHeads*r.width
  q_head_stride := r.width
  k_token_stride := r.kHeads*r.width
  k_head_stride := r.width
  head_dim_stride := 1
  cos_token_stride := r.cosWidth
  cos_stride := 1
  sin_token_stride := r.sinWidth
  sin_stride := 1
  cacheb_stride := r.kHeads*r.blockSize*r.width
  cacheh_stride := r.blockSize*r.width
  cachebs_stride := r.width
  cached_stride := 1
  bts_stride := r.tableColumns
  btb_stride := 1
  block_size := r.blockSize
  q_total_tokens := r.tokens
  Q_HEAD_NUM := r.qHeads
  K_HEAD_NUM := r.kHeads
  HEAD_DIM := r.width
  BLOCK_PAIRS := r.blockPairs

theorem half_width (r : Request) : r.width/2=r.half := by simp [Request.width]

theorem halves_injective (half a b : Nat) (ha : a<half) (hb : b<half) (oa ob : Bool)
    (same : a+(if oa then half else 0)=b+(if ob then half else 0)) : a=b ∧ oa=ob := by
  cases oa <;> cases ob <;> simp_all <;> omega

theorem u_offset (r : Request) (s : State A) (key odd : Bool) (i : ULane (uLayout r)) :
    uOffset (uLayout r) s key odd i=
      uToken (uLayout r) s i*(r.heads key*r.width)+
        uHead (uLayout r) s key*r.width+r.dim i.2.2.1.val odd := by
  cases key <;> simp [uOffset,uLayout,uDim,Request.heads,Request.dim,half_width]

theorem u_ownership_raw (r : Request) (heads : ∀ key,0<r.heads key)
    (s : State A) (key : Bool) :
    Pair.WritesApart (uActive (uLayout r) s key) (uOffset (uLayout r) s key) := by
  have positive := heads key
  have injective (a b : ULane (uLayout r)) (ha : uActive (uLayout r) s key a)
      (hb : uActive (uLayout r) s key b) (oa ob : Bool)
      (same : uOffset (uLayout r) s key oa a=uOffset (uLayout r) s key ob b) :
      a=b ∧ oa=ob := by
    have pa : a.2.2.1.val<r.half := by simpa [uActive,uLayout,half_width] using ha.2.2
    have pb : b.2.2.1.val<r.half := by simpa [uActive,uLayout,half_width] using hb.2.2
    have da : r.dim a.2.2.1.val oa<r.heads key*r.width := by
      have width : r.width≤r.heads key*r.width := by nlinarith
      cases oa <;> simp [Request.dim,Request.width] at * <;> omega
    have db : r.dim b.2.2.1.val ob<r.heads key*r.width := by
      have width : r.width≤r.heads key*r.width := by nlinarith
      cases ob <;> simp [Request.dim,Request.width] at * <;> omega
    rw [u_offset,u_offset] at same
    have columns := Checked.mixed_injective (r.heads key*r.width)
      (uToken (uLayout r) s a) (r.dim a.2.2.1.val oa)
      (uToken (uLayout r) s b) (r.dim b.2.2.1.val ob) da db (by omega)
    have halves := halves_injective r.half _ _ pa pb oa ob columns.2
    have tokens := columns.1
    simp only [uToken] at tokens
    exact ⟨Prod.ext (Fin.ext (by omega))
      (Prod.ext (Subsingleton.elim _ _) (Prod.ext (Fin.ext halves.1) (Subsingleton.elim _ _))),halves.2⟩
  exact ⟨fun odd a b ha hb eq => (injective a b ha hb odd odd eq).1,
    fun a b ha hb eq => Bool.noConfusion (injective a b ha hb Bool.false Bool.true eq).2⟩

theorem u_ownership (c : Checked.Call) (index : Indices) (accepted : Checked.check c index=Bool.true)
    (s : State A) (key : Bool) :
    Pair.WritesApart (uActive (uLayout c.request) s key) (uOffset (uLayout c.request) s key) := by
  apply u_ownership_raw c.request _ s key
  intro key
  cases key
  · exact (Checked.shapes c index accepted).1
  · exact (Checked.shapes c index accepted).2.1

def MetadataMatches (r : Request) (index : Indices) (s : State A) [Inhabited A] : Prop :=
  (∀ token,token<r.tokens → (s.mem "Lengths" token).read .int=index (r.lengthBase+2*token)) ∧
  (∀ o,(s.mem "Tables" o).read .int=index (r.tableBase+2*o))

theorem metadata_nonnegative [Inhabited A] (c : Checked.Call) (index : Indices)
    (accepted : Checked.check c index=Bool.true) (cached : c.request.cached=Bool.true)
    (s : State A) (observed : MetadataMatches c.request index s)
    (token : s.pids 1<c.request.tokens) (nonempty : 0<c.request.half) :
    0≤cPast (cLayout c.request) s ∧ 0≤cBlock (cLayout c.request) s := by
  have active : c.request.Active ⟨Bool.false,s.pids 1,0,0⟩ :=
    ⟨token,(Checked.shapes c index accepted).1,nonempty⟩
  have checked := Checked.metadata_checked c index accepted _ active
  simp only [Checked.metadataCheck,cached,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at checked
  have h := checked.1 ⟨s.pids 1,token⟩
  dsimp only at h
  have past : cPast (cLayout c.request) s=index (c.request.lengthBase+2*s.pids 1)-1 := by
    simpa only [cPast,cLayout] using congrArg (fun x : Int => x-1) (observed.1 _ token)
  have nonneg : 0≤cPast (cLayout c.request) s := by rw [past]; omega
  refine ⟨nonneg,?_⟩
  unfold cBlock
  rw [Int.tdiv_eq_ediv_of_nonneg nonneg]
  have natural : cPast (cLayout c.request) s=(c.request.past index (s.pids 1) : Int) := by
    rw [past,Request.past,Int.toNat_of_nonneg (by omega)]
  rw [natural]
  simp only [cLayout,Nat.cast_one,mul_one,←Int.natCast_ediv,Int.toNat_natCast]
  rw [observed.2]
  exact h.2.2.2.1

theorem c_offset [Inhabited A] (r : Request) (s : State A)
    (past : 0≤cPast (cLayout r) s) (block : 0≤cBlock (cLayout r) s)
    (odd : Bool) (i : CLane (cLayout r)) :
    cOffset (cLayout r) s Bool.true odd i=
      (cBlock (cLayout r) s).toNat*(r.kHeads*r.blockSize*r.width)+
        s.pids 0*(r.blockSize*r.width)+
        (cPast (cLayout r) s).toNat%r.blockSize*r.width+r.dim i.1.val odd := by
  simp only [cOffset,↓reduceIte,Int.tmod_eq_emod_of_nonneg past]
  rw [←Int.toNat_of_nonneg past,←Int.toNat_of_nonneg block]
  simp only [←Int.natCast_emod,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp [cLayout,cDim,Request.dim,half_width]

theorem c_ownership [Inhabited A] (r : Request) (s : State A)
    (past : 0≤cPast (cLayout r) s) (block : 0≤cBlock (cLayout r) s) (key : Bool) :
    Pair.WritesApart (cActive (cLayout r) s key) (cOffset (cLayout r) s key) := by
  have injective (a b : CLane (cLayout r)) (ha : cActive (cLayout r) s key a)
      (hb : cActive (cLayout r) s key b) (oa ob : Bool)
      (same : cOffset (cLayout r) s key oa a=cOffset (cLayout r) s key ob b) : a=b ∧ oa=ob := by
    have pa : a.1.val<r.half := by simpa [cActive,cLayout,half_width] using ha.2
    have pb : b.1.val<r.half := by simpa [cActive,cLayout,half_width] using hb.2
    have dimensions : r.dim a.1.val oa=r.dim b.1.val ob := by
      cases key
      · simpa [cOffset,cInput,cLayout,cDim,Request.dim,half_width] using same
      · rw [c_offset r s past block,c_offset r s past block] at same
        omega
    have halves := halves_injective r.half _ _ pa pb oa ob dimensions
    exact ⟨Prod.ext (Fin.ext halves.1) (Subsingleton.elim _ _),halves.2⟩
  exact ⟨fun odd a b ha hb eq => (injective a b ha hb odd odd eq).1,
    fun a b ha hb eq => Bool.noConfusion (injective a b ha hb Bool.false Bool.true eq).2⟩

#axiomsClean u_ownership
#axiomsClean metadata_nonnegative
#axiomsClean c_ownership
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
