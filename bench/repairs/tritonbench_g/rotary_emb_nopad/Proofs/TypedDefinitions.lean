import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Typed
import VeriTile.Triton.Semantics.Integer

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem int_half (n : Nat) : (n : Int)/2=((n/2 : Nat) : Int) := by omega
@[simp] theorem dimension_projection (B : Nat) (h : Fin 1) (i : Fin B) :
    (TileShape.dropInsertedIndex [1,B] (0 : Fin 3) 1 (0,h,i,PUnit.unit)).2.1.val=i.val := rfl

abbrev ULane (p : Kernels.UncachedLayout) := TileIndex [p.BLOCK_TOKENS,1,p.BLOCK_PAIRS]
def uToken {A : Type} (p : Kernels.UncachedLayout) (s : State A) (i : ULane p) :=
  s.pids 1*p.BLOCK_TOKENS+i.1.val
def uHead {A : Type} (p : Kernels.UncachedLayout) (s : State A) (key : Bool) :=
  if key then s.pids 0/p.KV_GROUP_NUM else s.pids 0
def uDim (p : Kernels.UncachedLayout) (odd : Bool) (i : ULane p) :=
  i.2.2.1.val+if odd then p.HEAD_DIM/2 else 0
def uActive {A : Type} (p : Kernels.UncachedLayout) (s : State A) (key : Bool) (i : ULane p) : Prop :=
  (key=Bool.true ∨ s.pids 0<p.Q_HEAD_NUM) ∧
    uToken p s i<p.q_total_tokens ∧ i.2.2.1.val<p.HEAD_DIM/2
instance {A : Type} (p : Kernels.UncachedLayout) (s : State A) (key : Bool) :
    DecidablePred (uActive p s key) := fun _ => inferInstanceAs (Decidable (_∧_∧_))
def uRegion (p : Kernels.UncachedLayout) (key : Bool) := if key then p.k else p.q
def uOffset {A : Type} (p : Kernels.UncachedLayout) (s : State A) (key odd : Bool) (i : ULane p) :=
  uToken p s i*(if key then p.k_token_stride else p.q_token_stride)+
    uHead p s key*(if key then p.k_head_stride else p.q_head_stride)+uDim p odd i*p.head_dim_stride
def uCoefficient {A : Type} (p : Kernels.UncachedLayout) (s : State A) (sine : Bool) (i : ULane p) :=
  uToken p s i*(if sine then p.sin_token_stride else p.cos_token_stride)+
    i.2.2.1.val*(if sine then p.sin_stride else p.cos_stride)

/-- Independent split-half rotation. Coefficients are loaded at entry;
the K input can be observed after Q stores, exactly as the source requires. -/
def uValue {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.UncachedLayout)
    (s data : State A) (key odd : Bool) (i : ULane p) : Cell A :=
  let a := (data.mem (uRegion p key) (uOffset p s key Bool.false i)).read .real
  let b := (data.mem (uRegion p key) (uOffset p s key Bool.true i)).read .real
  let c := (s.mem p.cos (uCoefficient p s Bool.false i)).read .real
  let z := (s.mem p.sin (uCoefficient p s Bool.true i)).read .real
  .mk .real (if odd then ops.add (ops.mul a z) (ops.mul b c)
    else ops.sub (ops.mul a c) (ops.mul b z))

def uBranch {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.UncachedLayout)
    (s data : State A) (key : Bool) (entry : State A) : State A :=
  Pair.written (uRegion p key) (uActive p s key) (uOffset p s key) (uValue ops p s data key) entry

def uReference {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.UncachedLayout)
    (s : State A) : State A :=
  let q := uBranch ops p s s Bool.false s
  if s.pids 0%p.KV_GROUP_NUM=0 then uBranch ops p s q Bool.true q else q

abbrev CLane (p : Kernels.CachedLayout) := TileIndex [p.BLOCK_PAIRS]
def cDim (p : Kernels.CachedLayout) (odd : Bool) (i : CLane p) :=
  i.1.val+if odd then p.HEAD_DIM/2 else 0
def cActive {A : Type} (p : Kernels.CachedLayout) (s : State A) (key : Bool) (i : CLane p) : Prop :=
  (key=Bool.false ∨ s.pids 0<p.K_HEAD_NUM) ∧ i.1.val<p.HEAD_DIM/2
instance {A : Type} (p : Kernels.CachedLayout) (s : State A) (key : Bool) :
    DecidablePred (cActive p s key) := fun _ => inferInstanceAs (Decidable (_∧_))
def cRegion (p : Kernels.CachedLayout) (key : Bool) := if key then p.kv_cache else p.q
def cInput {A : Type} (p : Kernels.CachedLayout) (s : State A) (key odd : Bool) (i : CLane p) :=
  s.pids 1*(if key then p.k_token_stride else p.q_token_stride)+
    s.pids 0*(if key then p.k_head_stride else p.q_head_stride)+cDim p odd i*p.head_dim_stride
def cCoefficient {A : Type} (p : Kernels.CachedLayout) (s : State A) (sine : Bool) (i : CLane p) :=
  s.pids 1*(if sine then p.sin_token_stride else p.cos_token_stride)+
    i.1.val*(if sine then p.sin_stride else p.cos_stride)
def cPast {A : Type} [Inhabited A] (p : Kernels.CachedLayout) (s : State A) : Int :=
  (s.mem p.context_lengths (s.pids 1)).read .int-1
def cBlock {A : Type} [Inhabited A] (p : Kernels.CachedLayout) (s : State A) : Int :=
  (s.mem p.BLOCK_TABLES (s.pids 1*p.bts_stride+
    ((cPast p s).tdiv p.block_size*p.btb_stride).toNat)).read .int
def cOffset {A : Type} [Inhabited A] (p : Kernels.CachedLayout) (s : State A)
    (key odd : Bool) (i : CLane p) :=
  if key then
    (cBlock p s*p.cacheb_stride+(s.pids 0*p.cacheh_stride : Nat)+
      (cPast p s).tmod p.block_size*p.cachebs_stride+(cDim p odd i*p.cached_stride : Nat)).toNat
  else cInput p s Bool.false odd i

/-- Independent split-half formula; the same cosine/sine pair rotates Q and K. -/
def cValue {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.CachedLayout)
    (s : State A) (key odd : Bool) (i : CLane p) : Cell A :=
  let r := if key then p.k else p.q
  let a := (s.mem r (cInput p s key Bool.false i)).read .real
  let b := (s.mem r (cInput p s key Bool.true i)).read .real
  let c := (s.mem p.cos (cCoefficient p s Bool.false i)).read .real
  let z := (s.mem p.sin (cCoefficient p s Bool.true i)).read .real
  .mk .real (if odd then ops.add (ops.mul a z) (ops.mul b c)
    else ops.sub (ops.mul a c) (ops.mul b z))

def cBranch {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.CachedLayout)
    (s : State A) (key : Bool) (entry : State A) :=
  Pair.written (cRegion p key) (cActive p s key) (cOffset p s key) (cValue ops p s key) entry

def cReference {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.CachedLayout)
    (s : State A) : State A :=
  if s.pids 0<p.Q_HEAD_NUM then
    cBranch ops p s Bool.false (cBranch ops p s Bool.true s) else s

#specNonCircular uValue avoiding [Kernels.uncached,uncachedExecution]
#specNonCircular cValue avoiding [Kernels.cached,cachedExecution]
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
