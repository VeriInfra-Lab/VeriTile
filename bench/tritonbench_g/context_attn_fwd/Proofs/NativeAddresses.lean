import bench.tritonbench_g.context_attn_fwd.Proofs.NativeHost

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
set_option maxHeartbeats 4000000

def batchBits (p : Layout) (l : Formula.Launch) := (fromI32 l.batchHead).sdiv (scalar p.heads)
def headBits (p : Layout) (l : Formula.Launch) := (fromI32 l.batchHead).srem (scalar p.heads)
def groupedBits (p : Layout) (l : Formula.Launch) := (headBits p l).sdiv (scalar p.kvGroup)

theorem launch_exact (p : Layout) (l : Formula.Launch) (hb : l.batchHead<2^31)
    (hh : p.heads<2^63) (hg : p.kvGroup<2^63) :
    batchBits p l=word (Formula.batch p l) ∧ headBits p l=word (Formula.head p l) ∧
    groupedBits p l=word (Formula.head p l/p.kvGroup) := by
  have hhead : headBits p l=word (Formula.head p l) := by
    rw [headBits,fromI32_eq l.batchHead hb,scalar_eq,srem_eq _ _ (by omega) hh]
    rfl
  refine ⟨?_,hhead,?_⟩
  · rw [batchBits,fromI32_eq l.batchHead hb,scalar_eq,sdiv_eq _ _ (by omega) hh]
    rfl
  · rw [groupedBits,hhead,scalar_eq,sdiv_eq _ _ ?_ hg]
    have h := Nat.mod_le l.batchHead p.heads
    change l.batchHead%p.heads<2^63
    omega

theorem launch_abi (c : Host.Call) (abi : ABI c) (idx : GridIndex c.grid) :
    c.layout.heads<2^63 ∧ c.layout.kvGroup<2^63 := by
  have hb := Host.batch_bound c idx
  have hf := abi.fused_grid
  have hh : c.layout.heads≤c.batches*c.layout.heads := by
    simpa using Nat.mul_le_mul_right c.layout.heads (show 1≤c.batches by omega)
  have hg : c.layout.kvGroup≤c.layout.heads := Nat.div_le_self _ _
  omega

def queryBits (p : Layout) (l : Formula.Launch) (start row dim rs hs ds : Nat) :=
  (fromI32 start+rowBits l.block p.BM row)*scalar rs+
    headBits p l*scalar hs+fromI32 dim*scalar ds
def cacheBits (p : Layout) (l : Formula.Launch) (step row dim bs rs hs ds : Nat) :=
  batchBits p l*scalar bs+keyBits step p.BN row*scalar rs+
    groupedBits p l*scalar hs+fromI32 dim*scalar ds

theorem query_modular (p : Layout) (l : Formula.Launch) (start row dim rs hs ds : Nat)
    (hb : l.block<2^31) (hbh : l.batchHead<2^31) (hm : Tile p.BM)
    (hh : p.heads<2^63) (hg : p.kvGroup<2^63) (hstart : start<2^31)
    (hr : row<p.BM) (hd : dim<2^31) :
    queryBits p l start row dim rs hs ds=
      word ((start+Formula.queryRow p l row)*rs+Formula.head p l*hs+dim*ds) := by
  rw [queryBits,fromI32_eq start hstart,row_modular _ _ _ hb hm hr,
    (launch_exact p l hbh hh hg).2.1,fromI32_eq dim hd]
  simp only [scalar_eq,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  rfl

theorem cache_modular (p : Layout) (l : Formula.Launch) (step row dim bs rs hs ds : Nat)
    (hbh : l.batchHead<2^31) (hm : Tile p.BN) (hh : p.heads<2^63)
    (hg : p.kvGroup<2^63) (hr : row<p.BN) (hd : dim<2^31) :
    cacheBits p l step row dim bs rs hs ds=word
      (Formula.batch p l*bs+(step*p.BN+row)*rs+(Formula.head p l/p.kvGroup)*hs+dim*ds) := by
  rw [cacheBits,(launch_exact p l hbh hh hg).1,(launch_exact p l hbh hh hg).2.2,
    key_modular step p.BN row hm hr,fromI32_eq dim hd]
  simp only [scalar_eq,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

/-- Actual data pointers, including view storage offsets. Read-only regions may
overlap physically, even when their logical role names differ. -/
structure Bytes (extent : RegionBounds) where
  base : RegionName → Nat
  fits : ∀ rg,base rg+4*extent rg≤2^64

inductive DataRole where | query | key | value | output deriving DecidableEq
def DataRole.region (role : DataRole) (c : Host.Call) := match role with
  | .query => c.regions.query | .key => c.regions.key | .value => c.regions.value | .output => c.regions.out
def DataRole.row (role : DataRole) (c : Host.Call) := match role with
  | .query => c.layout.qRow | .key => c.layout.kRow | .value => c.layout.vRow | .output => c.layout.outRow
def DataRole.head (role : DataRole) (c : Host.Call) := match role with
  | .query => c.layout.qHead | .key => c.layout.kHead | .value => c.layout.vHead | .output => c.layout.outHead
def DataRole.dim (role : DataRole) (c : Host.Call) := match role with
  | .query => c.layout.qDim | .key => c.layout.kDim | .value => c.layout.vDim | .output => c.layout.outDim
def DataRole.batch (role : DataRole) (c : Host.Call) := match role with
  | .key => c.layout.kBatch | .value => c.layout.vBatch | _ => 0

def queryOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (row dim : Nat) :=
  (c.start (batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) row)*role.row c+
    Formula.head c.layout (c.launchAt idx)*role.head c+dim*role.dim c
def cacheOffset (c : Host.Call) (role : DataRole) (idx : GridIndex c.grid) (step row dim : Nat) :=
  batchAt c idx*role.batch c+(step*c.layout.BN+row)*role.row c+
    (Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)*role.head c+dim*role.dim c

def queryEvent (c : Host.Call) (B : Bytes extent) (role : DataRole) (idx : GridIndex c.grid) (row dim : Nat) :=
  access ((rowBits (c.launchAt idx).block c.layout.BM row).slt
    (lengthBits (seqAt c idx) (c.prompt (batchAt c idx)))) (B.base (role.region c)) 4
    (queryBits c.layout (c.launchAt idx) (c.start (batchAt c idx)) row dim (role.row c) (role.head c) (role.dim c))
def cacheEvent (c : Host.Call) (B : Bytes extent) (role : DataRole) (idx : GridIndex c.grid) (step row dim : Nat) :=
  access ((keyBits step c.layout.BN row).slt
    (endBits (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))))
    (B.base (role.region c)) 4
    (cacheBits c.layout (c.launchAt idx) step row dim (role.batch c) (role.row c) (role.head c) (role.dim c))

theorem dim_abi (c : Host.Call) (hc : Geometry.Checked c extent) (dim : Fin c.layout.D) : dim.val<2^31 := by
  have h := hc.width
  simp only [Finset.mem_insert,Finset.mem_singleton] at h
  have hd : dim.val<c.args.k.dims := dim.isLt
  omega

theorem event_exact (B : Bytes extent) (rg : RegionName) (on : Bool) (off : BitVec 64)
    (math : Nat) (active : Prop) [Decidable active]
    (hm : on=Bool.true ↔ active) (he : off=word math) (hb : active → math<extent rg) :
    access on (B.base rg) 4 off=if active then some (B.base rg+4*math) else none := by
  rw [he,access_exact on _ 4 math (extent rg) (by decide) (B.fits rg) (fun h => hb (hm.mp h))]
  by_cases h : active
  · simp only [h,if_true,hm.mpr h,Bool.true_eq]
  · have hf : on=Bool.false := by cases on <;> simp_all
    simp [h,hf]

theorem query_event (c : Host.Call) (B : Bytes extent) (hc : Geometry.Checked c extent) (abi : ABI c)
    (role : DataRole) (hr : role=.query ∨ role=.output) (idx : GridIndex c.grid)
    (i : Fin c.layout.BM) (d : Fin c.layout.D) :
    queryEvent c B role idx i d=if Formula.queryRow c.layout (c.launchAt idx) i<c.length (batchAt c idx)
      then some (B.base (role.region c)+4*queryOffset c role idx i d) else none := by
  have hba := Host.batch_bound c idx
  have hh := launch_abi c abi idx
  apply event_exact B _ _ _ _ _
  · simpa only [seqAt,Nat.add_sub_cancel_right,Formula.queryRow] using
      query_mask (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx)) i
        (block_abi c abi idx) (tile c) (abi.sequence _ hba) (by simp only [seqAt,batchAt]; omega) i.isLt
  · exact query_modular _ _ _ _ _ _ _ _ (block_abi c abi idx) (batchHead_abi c abi idx)
      (tile c) hh.1 hh.2 (abi.start _ hba) i.isLt (dim_abi c hc d)
  · intro hi
    rcases hr with rfl|rfl
    · exact Geometry.query_bound c hc _ _ _ _ hba (Host.head_bound c idx) hi d.isLt
    · exact Geometry.output_bound c hc _ _ _ _ hba (Host.head_bound c idx) hi d.isLt

theorem cache_event (c : Host.Call) (B : Bytes extent) (hc : Geometry.Checked c extent) (abi : ABI c)
    (role : DataRole) (hr : role=.key ∨ role=.value) (idx : GridIndex c.grid)
    (step : Nat) (j : Fin c.layout.BN) (d : Fin c.layout.D) (ht : step<countAt c idx) :
    cacheEvent c B role idx step j d=if step*c.layout.BN+j.val<endAt c idx
      then some (B.base (role.region c)+4*cacheOffset c role idx step j d) else none := by
  have hba := Host.batch_bound c idx
  have hh := launch_abi c abi idx
  apply event_exact B _ _ _ _ _
  · simpa only [c.blockN] using key_mask (c.launchAt idx).block c.layout.BM (seqAt c idx)
      (c.prompt (batchAt c idx)) step j (block_abi c abi idx) (tile c) (abi.sequence _ hba)
      (by simp only [seqAt,batchAt]; omega) ht (by simpa only [c.blockN] using j.isLt)
  · exact cache_modular _ _ _ _ _ _ _ _ _ (batchHead_abi c abi idx)
      (by rw [c.blockN]; exact tile c) hh.1 hh.2 j.isLt (dim_abi c hc d)
  · intro hj
    have hs : step*c.layout.BN+j.val<c.length (batchAt c idx)+c.prompt (batchAt c idx) :=
      lt_of_lt_of_le hj (end_le _ _ _ _)
    rcases hr with rfl|rfl
    · exact Geometry.key_bound c hc _ _ _ _ hba (Host.head_bound c idx) hs d.isLt
    · exact Geometry.value_bound c hc _ _ _ _ hba (Host.head_bound c idx) hs d.isLt

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
