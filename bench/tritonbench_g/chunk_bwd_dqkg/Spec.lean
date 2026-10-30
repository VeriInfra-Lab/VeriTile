import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeFlat

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton Whole
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000

namespace Spec
/-- Logical tensors use a flattened B,H coordinate. History's middle coordinate
is the chunk index; it is independent of the token/key dimensions. -/
structure Tensors where
  (query key value outputGrad : Nat → Nat → Nat → ℝ)
  (history historyGrad : Nat → Nat → Nat → Nat → ℝ)
  gate : Nat → Nat → ℝ

def views (c : Config) (μ : RegionName → Nat → ℝ) : Tensors where
  query h t e := μ "q" ((h*c.T+t)*c.K+e)
  key h t e := μ "k" ((h*c.T+t)*c.K+e)
  value h t j := μ "v" ((h*c.T+t)*c.V+j)
  outputGrad h t j := μ "do" ((h*c.T+t)*c.V+j)
  history h chunk e j := μ "h" (((h*c.NT+chunk)*c.K+e)*c.V+j)
  historyGrad h chunk e j := μ "dh" (((h*c.NT+chunk)*c.K+e)*c.V+j)
  gate h t := μ "g" (h*c.T+t)

def qk (c : Config) (a : Nat → Nat → Nat → ℝ) (h chunk r e : Nat) : ℝ :=
  if chunk*64+r<c.T ∧ e<c.K then a h (chunk*64+r) e else 0
def vd (c : Config) (a : Nat → Nat → Nat → ℝ) (h chunk r j : Nat) : ℝ :=
  if chunk*64+r<c.T ∧ j<c.V then a h (chunk*64+r) j else 0
def hist (c : Config) (a : Nat → Nat → Nat → Nat → ℝ) (h chunk e j : Nat) : ℝ :=
  if j<c.V ∧ e<c.K then a h chunk e j else 0
def gate (c : Config) (a : Tensors) (h chunk r : Nat) : ℝ :=
  if chunk*64+r<c.T then a.gate h (chunk*64+r) else 0
def lastGate (c : Config) (a : Tensors) (h chunk : Nat) : ℝ :=
  a.gate h (min (chunk*64+64) c.T-1)

/-- Independent finite contractions of pre-call input tensors. -/
def baseQ (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, vd c a.outputGrad h chunk r j * hist c a.history h chunk e j) *
    Real.exp (gate c a h chunk r) * c.scale
def baseK (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, vd c a.value h chunk r j * hist c a.historyGrad h chunk e j) *
    Real.exp (0-gate c a h chunk r+lastGate c a h chunk)
def score (c : Config) (a : Tensors) (h chunk r u : Nat) : ℝ :=
  if u≤r then
    (∑ j ∈ Finset.range c.V, vd c a.outputGrad h chunk r j * vd c a.value h chunk u j) *
      c.scale * Real.exp (gate c a h chunk r-gate c a h chunk u)
  else 0
def dq (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  baseQ c a h chunk r e + ∑ u : Fin 64, score c a h chunk r u.val * qk c a.key h chunk u.val e
def dk (c : Config) (a : Tensors) (h chunk r e : Nat) : ℝ :=
  baseK c a h chunk r e + ∑ u : Fin 64, score c a h chunk u.val r * qk c a.query h chunk u.val e
def correction (c : Config) (a : Tensors) (key h chunk : Nat) : ℝ :=
  (∑ j ∈ Finset.range c.V, ∑ e : Fin c.BK,
    hist c a.history h chunk (key*c.BK+e.val) j * hist c a.historyGrad h chunk (key*c.BK+e.val) j) *
    Real.exp (lastGate c a h chunk) +
  ∑ r : Fin 64, ∑ e : Fin c.BK,
    baseK c a h chunk r.val (key*c.BK+e.val) * qk c a.key h chunk r.val (key*c.BK+e.val)
def dg (c : Config) (a : Tensors) (key h chunk r : Nat) : ℝ :=
  (∑ e : Fin c.BK,
    (qk c a.query h chunk r (key*c.BK+e.val) * dq c a h chunk r (key*c.BK+e.val) -
    qk c a.key h chunk r (key*c.BK+e.val) * dk c a h chunk r (key*c.BK+e.val))) +
  if r<min 64 (c.T-chunk*64)-1 then 0 else correction c a key h chunk

def fromMemory (μ : RegionName → Nat → ℝ) : RealMath.Inputs :=
  ⟨μ "q",μ "k",μ "v",μ "h",μ "g",μ "do",μ "dh"⟩

theorem tile_bridge (c : Config) (μ : RegionName → Nat → ℝ) (a : CTA c)
    (r : Fin 64) (e : Fin c.BK) :
    RealMath.dq c.layout a.launch (fromMemory μ) r e =
      dq c (views c μ) a.head.val a.time.val r.val (a.key.val*c.BK+e.val) ∧
    RealMath.dk c.layout a.launch (fromMemory μ) r e =
      dk c (views c μ) a.head.val a.time.val r.val (a.key.val*c.BK+e.val) ∧
    RealMath.dg c.layout a.launch (fromMemory μ) r =
      dg c (views c μ) a.key.val a.head.val a.time.val r.val := by
  simp only [RealMath.dq,RealMath.dk,RealMath.dg,RealMath.gatedDq,RealMath.gatedDk,
    RealMath.score,RealMath.lastCorrection,RealMath.total,RealMath.dqTerm,RealMath.dkTerm,
    RealMath.dsTerm,RealMath.dgLastTerm,RealMath.value,RealMath.history,RealMath.queryKey,
    RealMath.gate,RealMath.lastGate,Config.layout,CTA.launch,fromMemory,
    dq,dk,dg,baseQ,baseK,score,correction,qk,vd,hist,gate,lastGate,views,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  exact ⟨rfl,rfl,rfl⟩

theorem logical_q (c : Config) (s : BlockState) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    expectedQ c s h t e = dq c (views c s.readMem) h.val (t.val/64) (t.val%64) e.val := by
  have hq := (tile_bridge c s.readMem (qOwner c h t e) (rowLane t.val) (keyLane c e.val)).1
  simpa only [expectedQ,qOwner,rowLane,keyLane,Nat.div_add_mod',fromMemory,inputs,RealPost.inputs] using hq
theorem logical_k (c : Config) (s : BlockState) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    expectedK c s h t e = dk c (views c s.readMem) h.val (t.val/64) (t.val%64) e.val := by
  have hk := (tile_bridge c s.readMem (qOwner c h t e) (rowLane t.val) (keyLane c e.val)).2.1
  simpa only [expectedK,qOwner,rowLane,keyLane,Nat.div_add_mod',fromMemory,inputs,RealPost.inputs] using hk
theorem logical_g (c : Config) (s : BlockState) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) :
    expectedG c s k h t = dg c (views c s.readMem) k.val h.val (t.val/64) (t.val%64) := by
  have hg := (tile_bridge c s.readMem (gOwner c k h t) (rowLane t.val) (keyLane c 0)).2.2
  simpa only [expectedG,gOwner,rowLane,keyLane,fromMemory,inputs,RealPost.inputs] using hg

#specNonCircular dq avoiding [exec,runGrid,runFlat]
#specNonCircular dk avoiding [exec,runGrid,runFlat]
#specNonCircular dg avoiding [exec,runGrid,runFlat]
#guard_msgs (drop info) in
#auditModuleAxioms
end Spec
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg
