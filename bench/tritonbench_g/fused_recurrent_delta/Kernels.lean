import VeriTile.Triton
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Nat.Log

/-! Selected FP32 source interpreter. Addresses remain logical below; the
layout module maps them to the exact dense source pointer expressions.
Every load sees the current memory, including v, dk and dv reloads. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
inductive Region where
  | q | k | v | beta | h0 | o | ht | do_ | dht
  | dq | dk | dv | dbeta | dh0 | retq | retk | retv | retbeta
  deriving DecidableEq,Repr
structure Addr where
  region : Region
  plane : Nat
  row : Nat
  col : Nat
  deriving DecidableEq,Repr
abbrev Memory (α : Type) := Addr → α
structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
def Shape.BH (s : Shape) := s.B*s.H
def Shape.BK (s : Shape) := s.K.nextPowerOfTwo
def Shape.FBV (s : Shape) := min s.V.nextPowerOfTwo 8
def Shape.BBV (s : Shape) := min s.V.nextPowerOfTwo 32
def ceil (n d : Nat) := (n+d-1)/d
def Shape.FNV (s : Shape) := ceil s.V s.FBV
def Shape.BNV (s : Shape) := ceil s.V s.BBV
def Shape.Positive (s : Shape) := 0<s.B ∧ 0<s.H ∧ 0<s.T ∧ 0<s.K ∧ 0<s.V
structure Flags where
  vectorBeta : Bool
  initial : Bool
  final : Bool
  initialGrad : Bool
  terminalGrad : Bool
  defaultBeta : Bool

structure Ops (α : Type) where
  zero : α
  one : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  cut : α → α

/-- A finite scalar add tree. Leaves include every padded lane; zero nodes
are explicit, so no seeded/sequential interpretation is silently imposed. -/
inductive Tree where
  | leaf (i : Nat)
  | zero
  | add (a b : Tree)
  deriving Repr
def Tree.eval {α : Type} (p : Ops α) (x : Nat → α) : Tree → α
  | .leaf i => x i
  | .zero => p.zero
  | .add a b => p.add (a.eval p x) (b.eval p x)
def Tree.leaves : Tree → List Nat
  | .leaf i => [i]
  | .zero => []
  | .add a b => a.leaves++b.leaves
def Tree.Legal (n : Nat) (t : Tree) := t.leaves.Perm (List.range n)
structure Trees where
  key : Nat → Tree
  value : Nat → Tree
  host : Nat → Tree
  readout : Nat → Tree
  beta : Nat → Tree
  correction : Nat → Tree
  query : Nat → Tree
  hostK : Nat → Tree
  hostV : Nat → Tree
  hostBeta : Nat → Tree
/-- Distinct source reduction sites have independent trees. -/
def Trees.Legal (g : Trees) := ∀ n,(g.key n).Legal n ∧ (g.value n).Legal n ∧ (g.host n).Legal n ∧
  (g.readout n).Legal n ∧ (g.beta n).Legal n ∧ (g.correction n).Legal n ∧
  (g.query n).Legal n ∧ (g.hostK n).Legal n ∧ (g.hostV n).Legal n ∧ (g.hostBeta n).Legal n

def write {α : Type} (m : Memory α) (mask : Addr → Prop) (value : Addr → α) : Memory α := by
  classical
  exact fun a => if mask a then value a else m a
def rowStore {α : Type} (p : Ops α) (m : Memory α) (r : Region) (plane row lo hi : Nat)
    (value : Nat → α) := write m
  (fun a => a.region=r ∧ a.plane=plane ∧ a.row=row ∧ lo≤a.col ∧ a.col<hi)
  (fun a => p.cut (value a.col))
def stateStore {α : Type} (s : Shape) (p : Ops α) (m : Memory α) (r : Region)
    (bh iv BV : Nat) (h : Nat → Nat → α) := write m
  (fun a => a.region=r ∧ a.plane=bh ∧ a.row<s.K ∧ iv*BV≤a.col ∧ a.col<min ((iv+1)*BV) s.V)
  (fun a => p.cut (h a.row (a.col-iv*BV)))
def load {α : Type} (p : Ops α) (m : Memory α) (r : Region) (plane row col : Nat)
    (active : Prop) [Decidable active] := if active then m ⟨r,plane,row,col⟩ else p.zero
def keyLoad {α : Type} (s : Shape) (p : Ops α) (m : Memory α) (r : Region) (bh t i : Nat) :=
  load p m r bh t i (i<s.K)
def valLoad {α : Type} (s : Shape) (p : Ops α) (m : Memory α) (r : Region) (bh t iv BV j : Nat) :=
  load p m r bh t (iv*BV+j) (iv*BV+j<s.V)
def betaLoad {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α)
    (bh t iv BV j : Nat) :=
  if f.vectorBeta then valLoad s p m .beta bh t iv BV j else m ⟨.beta,bh,t,0⟩
def seed {α : Type} (s : Shape) (p : Ops α) (m : Memory α) (r : Region)
    (use : Bool) (bh iv BV : Nat) : Nat → Nat → α := fun i j =>
  if use then p.add p.zero (load p m r bh i (iv*BV+j) (i<s.K ∧ iv*BV+j<s.V)) else p.zero
structure Machine (α : Type) where
  h : Nat → Nat → α
  mem : Memory α

def forwardStep {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv t : Nat) (st : Machine α) : Machine α :=
  let k:=keyLoad s p st.mem .k bh t
  let q:=fun i => p.mul (keyLoad s p st.mem .q bh t i) scale
  let residual:=fun j => p.sub (valLoad s p st.mem .v bh t iv s.FBV j)
    ((g.key s.BK).eval p (fun i => p.mul (st.h i j) (k i)))
  let u:=fun j => p.mul (residual j) (betaLoad s f p st.mem bh t iv s.FBV j)
  let h:=fun i j => p.add (st.h i j) (p.mul (k i) (u j))
  let output:=fun j => (g.readout s.BK).eval p (fun i => p.mul (h i j) (q i))
  let m:=rowStore p st.mem .v bh t (iv*s.FBV) (min ((iv+1)*s.FBV) s.V)
    (fun v => residual (v-iv*s.FBV))
  let m:=rowStore p m .o bh t (iv*s.FBV) (min ((iv+1)*s.FBV) s.V)
    (fun v => output (v-iv*s.FBV))
  ⟨h,m⟩
def forwardLoop {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) : Nat → Machine α → Machine α
  | 0,st => st
  | n+1,st => forwardStep s f p g scale bh iv n (forwardLoop s f p g scale bh iv n st)
def forwardProgram {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) : Memory α :=
  let st:=forwardLoop s f p g scale bh iv s.T ⟨seed s p m .h0 f.initial bh iv s.FBV,m⟩
  if f.final then stateStore s p st.mem .ht bh iv s.FBV st.h else st.mem

def reverseStep {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (st : Machine α) : Machine α :=
  let t:=s.T-1-n
  let k:=keyLoad s p st.mem .k bh t
  let q:=fun i => p.mul (keyLoad s p st.mem .q bh t i) scale
  let v:=valLoad s p st.mem .v bh t iv s.BBV
  let do_:=valLoad s p st.mem .do_ bh t iv s.BBV
  let beta:=betaLoad s f p st.mem bh t iv s.BBV
  let dh:=fun i j => p.add (st.h i j) (p.mul (q i) (do_ j))
  let dk:=fun i => (g.value s.BBV).eval p (fun j => p.mul (dh i j) (p.mul (v j) (beta j)))
  let rawdv:=fun j => (g.key s.BK).eval p (fun i => p.mul (dh i j) (k i))
  let dbeta:=fun j => p.mul (rawdv j) (v j)
  let dv:=fun j => p.mul (rawdv j) (beta j)
  let m:=rowStore p st.mem .dk (iv*s.BH+bh) t 0 s.K dk
  let m:=rowStore p m .dv bh t (iv*s.BBV) (min ((iv+1)*s.BBV) s.V) (fun v => dv (v-iv*s.BBV))
  let m:=if f.vectorBeta then
      rowStore p m .dbeta (iv*s.BH+bh) t (iv*s.BBV) (min ((iv+1)*s.BBV) s.V) (fun v => dbeta (v-iv*s.BBV))
    else rowStore p m .dbeta (iv*s.BH+bh) t 0 1 (fun _ => (g.beta s.BBV).eval p dbeta)
  ⟨fun i j => p.sub (dh i j) (p.mul (k i) (dv j)),m⟩
def reverseLoop {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) : Nat → Machine α → Machine α
  | 0,st => st
  | n+1,st => reverseStep s f p g scale bh iv n (reverseLoop s f p g scale bh iv n st)

def recomputeStep {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv t : Nat) (st : Machine α) : Machine α :=
  let dk:=keyLoad s p st.mem .dk (iv*s.BH+bh) t
  let dv:=valLoad s p st.mem .dv bh t iv s.BBV
  let corrected:=fun i => p.sub (dk i) ((g.correction s.BBV).eval p (fun j => p.mul (dv j) (st.h i j)))
  let m:=rowStore p st.mem .dk (iv*s.BH+bh) t 0 s.K corrected
  let k:=keyLoad s p m .k bh t
  let v:=valLoad s p m .v bh t iv s.BBV
  let do_:=valLoad s p m .do_ bh t iv s.BBV
  let beta:=betaLoad s f p m bh t iv s.BBV
  let h:=fun i j => p.add (st.h i j) (p.mul (k i) (p.mul (v j) (beta j)))
  let dq:=fun i => p.mul ((g.query s.BBV).eval p (fun j => p.mul (h i j) (do_ j))) scale
  let m:=rowStore p m .dq (iv*s.BH+bh) t 0 s.K dq
  ⟨h,m⟩
def recomputeLoop {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) : Nat → Machine α → Machine α
  | 0,st => st
  | n+1,st => recomputeStep s f p g scale bh iv n (recomputeLoop s f p g scale bh iv n st)
def backwardProgram {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) : Memory α :=
  let st:=reverseLoop s f p g scale bh iv s.T ⟨seed s p m .dht f.terminalGrad bh iv s.BBV,m⟩
  let m:=if f.initialGrad then stateStore s p st.mem .dh0 bh iv s.BBV st.h else st.mem
  -- The source barrier is intra-program. Every reload below sees this m.
  (recomputeLoop s f p g scale bh iv s.T ⟨seed s p m .h0 f.initial bh iv s.BBV,m⟩).mem

def jobs (BH NV : Nat) := (List.range BH).flatMap (fun bh => (List.range NV).map (fun iv => (bh,iv)))
def forwardGrid {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := (jobs s.BH s.FNV).foldl (fun m job => forwardProgram s f p g scale job.1 job.2 m) m
def backwardGrid {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := (jobs s.BH s.BNV).foldl (fun m job => backwardProgram s f p g scale job.1 job.2 m) m
def prepareBeta {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α) :=
  if f.defaultBeta then write m (fun a => a.region=.beta ∧ a.plane<s.BH ∧ a.row<s.T ∧ a.col=0) (fun _ => p.one) else m
def prepareDbeta {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α) :=
  if f.vectorBeta then write m (fun a => a.region=.dbeta ∧ a.plane<s.BNV*s.BH ∧ a.row<s.T ∧ a.col<s.V) (fun _ => p.zero) else m
def hostReduce {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α) :=
  let m':=write m (fun a => a.plane<s.BH ∧ a.row<s.T ∧
      ((a.region=.retq ∨ a.region=.retk) ∧ a.col<s.K ∨
       a.region=.retv ∧ a.col<s.V ∨ a.region=.retbeta ∧ a.col<(if f.vectorBeta then s.V else 1)))
    (fun a => p.cut (match a.region with
    | .retq => (g.host s.BNV).eval p (fun iv => m ⟨.dq,iv*s.BH+a.plane,a.row,a.col⟩)
    | .retk => (g.hostK s.BNV).eval p (fun iv => m ⟨.dk,iv*s.BH+a.plane,a.row,a.col⟩)
    | .retv => (g.hostV 1).eval p (fun _ => m ⟨.dv,a.plane,a.row,a.col⟩)
    | .retbeta => (g.hostBeta s.BNV).eval p (fun iv => m ⟨.dbeta,iv*s.BH+a.plane,a.row,a.col⟩)
    | _ => p.zero))
  m'
def forwardCall {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α) :=
  forwardGrid s f p g scale (prepareBeta s f p m)
def backwardCall {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α) :=
  hostReduce s f p g (backwardGrid s f p g scale (prepareDbeta s f p m))
def forwardReturns (f : Flags) : Region × Option Region := (.o,if f.final then some .ht else none)
def backwardReturns (f : Flags) : List (Option Region) :=
  [some .retq,some .retk,some .retv,some .retbeta,none,if f.initialGrad then some .dh0 else none,none]

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta

/-! Complete actual source entries. Scalar/vector beta specialization is
shown as ordinary quotations. scale is an incoming FP32 ABI argument.
forwardDirectOutput removes the identically zero key-grid term from the
forward output pointer, using the existing single key-grid launch (pid1=0).
Every FP expression and memory effect is retained; no speedup is claimed. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Runtime
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
def forward
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) 
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ComputeKernel := if IS_HEADWISE_BETA then triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(V)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
} else triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(T)
  p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(1)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

def backward
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) 
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) :
    ComputeKernel := if IS_HEADWISE_BETA then triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, dht = tl.float32, dh0 = tl.float32,
    do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32, dbeta = tl.float32, h0 = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dbeta = dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    d_beta = d_v * b_v
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty), mask=mask_bv)
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    p_dbeta -= $(V)
    p_beta -= $(V)
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * scale
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    p_beta += $(V)
  }
} else triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, dht = tl.float32, dh0 = tl.float32,
    do_ = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32, dbeta = tl.float32, h0 = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  mask_bk = i_k * $(BK) + tl.arange(0, $(BK)) < $(K)
  mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(V)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_beta = beta + i_bh * $(T) + ($(T) - $(1))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - $(1)) * $(K)
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - $(1)) * $(V)
  p_dbeta = dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + ($(T) - $(1))
  d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_DHT {
    p_ht = dht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
  }
  for _i in range($(T)) {
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta).to(tl.float32)
    d_h += b_q[:, None] * b_do[None, :]
    d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
    d_v = tl.sum(d_h * b_k[:, None], axis=0)
    d_beta = tl.sum(d_v * b_v)
    d_v = d_v * b_beta
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
    tl.store(p_dbeta, (d_beta).to(p_dbeta.dtype.element_ty))
    d_h -= b_k[:, None] * d_v[None, :]
    p_do -= $(V)
    p_q -= $(K)
    p_k -= $(K)
    p_v -= $(V)
    p_dk -= $(K)
    p_dv -= $(V)
    p_dbeta -= $(1)
    p_beta -= $(1)
  }
  if USE_DH0 {
    p_dh0 = dh0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    tl.store(p_dh0, (d_h).to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])
  }
  tl.debug_barrier()
  h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(T)
  p_do = do_ + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  if USE_INITIAL_STATE {
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for i in range($(0), $(T), $(1)) {
    d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
    d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
    d_k -= tl.sum(d_v[None, :] * h, axis=1)
    tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
    b_beta = tl.load(p_beta).to(tl.float32)
    b_v *= b_beta
    h += b_k[:, None] * b_v[None, :]
    _d_q = h * b_do[None, :]
    d_q = tl.sum(_d_q, axis=1) * scale
    tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
    p_k += $(K)
    p_do += $(V)
    p_v += $(V)
    p_dk += $(K)
    p_dv += $(V)
    p_dq += $(K)
    p_beta += $(1)
  }
}

def forwardDirectOutput
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) 
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) :
    ComputeKernel := if IS_HEADWISE_BETA then triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_o = o + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(V)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
} else triton {
  tl.param scale : tl.float32
  tl.region q = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, o = tl.float32, h0 = tl.float32, ht = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
  p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  p_beta = beta + i_bh * $(T)
  p_o = o + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(K)
  mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(V)
  mask_kv = mask_bk[None, :] & mask_bv[:, None]
  h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
    _v_minus = tl.sum(h * b_k[None, :], axis=1)
    b_v -= _v_minus
    b_beta = tl.load(p_beta).to(tl.float32)
    tl.store(p_v, (b_v).to(p_v.dtype.element_ty), mask=mask_bv)
    b_v *= b_beta
    h += b_k[None, :] * b_v[:, None]
    _o = h * b_q[None, :]
    _o = tl.sum(_o, axis=1)
    tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
    p_q += $(K)
    p_k += $(K)
    p_o += $(V)
    p_v += $(V)
    p_beta += $(1)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(K) * $(V) +
      (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(V) +
      (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
    tl.store(p_ht, (h).to(p_ht.dtype.element_ty), mask=mask_kv)
  }
}

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Runtime
