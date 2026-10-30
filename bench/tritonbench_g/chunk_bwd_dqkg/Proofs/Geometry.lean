import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealMath

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton

/-- These are logical dimensions; no source result or execution is an input. -/
structure Config where
  (B H T K V : Nat)
  scale : ℝ

def tileWidth (n : Nat) : Nat := if n ≤ 16 then 16 else if n ≤ 32 then 32 else 64
def Config.heads (c : Config) : Nat := c.B * c.H
def Config.BK (c : Config) : Nat := tileWidth c.K
def Config.BV (c : Config) : Nat := tileWidth c.V
def Config.NT (c : Config) : Nat := (c.T + 63) / 64
def Config.NK (c : Config) : Nat := (c.K + c.BK - 1) / c.BK

theorem width_pos (n : Nat) : 0 < tileWidth n := by
  unfold tileWidth
  split_ifs <;> decide

theorem width_cases (n : Nat) : tileWidth n = 16 ∨ tileWidth n = 32 ∨ tileWidth n = 64 := by
  unfold tileWidth
  split_ifs <;> simp

theorem ceil_cover (n w : Nat) (hw : 0 < w) : n ≤ ((n + w - 1) / w) * w := by
  have h := Nat.lt_mul_div_succ (n + w - 1) hw
  rw [Nat.mul_add, Nat.mul_one, Nat.mul_comm w] at h
  omega

def Config.layout (c : Config) : RealMath.Layout where
  T := c.T
  K := c.K
  V := c.V
  BT := 64
  BK := c.BK
  BV := c.BV
  NT := c.NT
  s_k_h := c.T * c.K
  s_k_t := c.K
  s_v_h := c.T * c.V
  s_v_t := c.V
  s_h_h := c.NT * c.K * c.V
  s_h_t := c.V
  scale := c.scale

structure CTA (c : Config) where
  key : Fin c.NK
  time : Fin c.NT
  head : Fin c.heads
  deriving DecidableEq

def CTA.launch {c : Config} (a : CTA c) : RealMath.Launch :=
  ⟨a.key.val, a.time.val, a.head.val, c.heads⟩

def activeQK (c : Config) (a : CTA c) (r : Fin 64) (e : Fin c.BK) : Prop :=
  a.time.val * 64 + r.val < c.T ∧ a.key.val * c.BK + e.val < c.K

def activeG (c : Config) (a : CTA c) (r : Fin 64) : Prop :=
  a.time.val * 64 + r.val < c.T

def qAddress (c : Config) (a : CTA c) (r : Fin 64) (e : Fin c.BK) : Nat :=
  (a.head.val * c.T + (a.time.val * 64 + r.val)) * c.K + (a.key.val * c.BK + e.val)

def gAddress (c : Config) (a : CTA c) (r : Fin 64) : Nat :=
  (a.key.val * c.heads + a.head.val) * c.T + a.time.val * 64 + r.val

theorem time_start_lt (c : Config) (a : CTA c) : a.time.val * 64 < c.T := by
  have ht := a.time.isLt
  unfold Config.NT at ht
  omega

theorem T_pos (c : Config) (a : CTA c) : 0 < c.T := by
  have := time_start_lt c a
  omega

theorem heads_pos (c : Config) (a : CTA c) : 0 < c.heads :=
  Nat.zero_lt_of_lt a.head.isLt

theorem key_start_lt (c : Config) (a : CTA c) : a.key.val * c.BK < c.K := by
  have hk := a.key.isLt
  rcases width_cases c.K with h | h | h
  all_goals
    change c.BK = _ at h
    unfold Config.NK at hk
    simp only [h] at hk ⊢
    omega

theorem K_pos (c : Config) (a : CTA c) : 0 < c.K := by
  have := key_start_lt c a
  omega

/-- Uniqueness of a canonical dense row/column pair. -/
theorem pack_injective (w a b i j : Nat) (hw : 0 < w) (hi : i < w) (hj : j < w)
    (he : a * w + i = b * w + j) : a = b ∧ i = j := by
  have qa : (a * w + i) / w = a := by
    rw [Nat.mul_comm a, Nat.mul_add_div hw, Nat.div_eq_of_lt hi, Nat.add_zero]
  have qb : (b * w + j) / w = b := by
    rw [Nat.mul_comm b, Nat.mul_add_div hw, Nat.div_eq_of_lt hj, Nat.add_zero]
  have hab := congrArg (fun z => z / w) he
  dsimp only at hab
  rw [qa, qb] at hab
  refine ⟨hab, ?_⟩
  simpa only [hab, Nat.add_left_cancel_iff] using he

/-- Distinct CTAs cannot write the same logical dQ/dK element. -/
theorem q_unique (c : Config) (a b : CTA c) (r s : Fin 64) (e f : Fin c.BK)
    (ha : activeQK c a r e) (hb : activeQK c b s f)
    (he : qAddress c a r e = qAddress c b s f) : a = b ∧ r = s ∧ e = f := by
  obtain ⟨hrow,hkey⟩ := pack_injective c.K _ _ _ _ (K_pos c a) ha.2 hb.2 he
  obtain ⟨hhead,htime⟩ := pack_injective c.T _ _ _ _ (T_pos c a) ha.1 hb.1 hrow
  obtain ⟨ht,hr⟩ := pack_injective 64 _ _ _ _ (by decide) r.isLt s.isLt htime
  obtain ⟨hk,hc⟩ := pack_injective c.BK _ _ _ _ (width_pos c.K) e.isLt f.isLt hkey
  have hcta : a = b := by
    cases a; cases b
    congr 1 <;> apply Fin.ext <;> assumption
  exact ⟨hcta, Fin.ext hr, Fin.ext hc⟩

/-- The dG leading key-tile axis is part of the address, not a reduction. -/
theorem g_unique (c : Config) (a b : CTA c) (r s : Fin 64)
    (ha : activeG c a r) (hb : activeG c b s)
    (he : gAddress c a r = gAddress c b s) : a = b ∧ r = s := by
  have he' : (a.key.val * c.heads + a.head.val) * c.T + (a.time.val * 64 + r.val) =
      (b.key.val * c.heads + b.head.val) * c.T + (b.time.val * 64 + s.val) := by
    simpa only [gAddress, Nat.add_assoc] using he
  obtain ⟨hkh,htime⟩ := pack_injective c.T _ _ _ _ (T_pos c a) ha hb he'
  obtain ⟨hk,hh⟩ := pack_injective c.heads _ _ _ _ (heads_pos c a) a.head.isLt b.head.isLt hkh
  obtain ⟨ht,hr⟩ := pack_injective 64 _ _ _ _ (by decide) r.isLt s.isLt htime
  have hcta : a = b := by
    cases a; cases b
    congr 1 <;> apply Fin.ext <;> assumption
  exact ⟨hcta, Fin.ext hr⟩

theorem key_owner_lt (c : Config) (e : Fin c.K) : e.val / c.BK < c.NK := by
  apply (Nat.div_lt_iff_lt_mul (width_pos c.K)).2
  exact lt_of_lt_of_le e.isLt (ceil_cover c.K c.BK (width_pos c.K))

theorem time_owner_lt (c : Config) (t : Fin c.T) : t.val / 64 < c.NT := by
  have ht := t.isLt
  unfold Config.NT
  omega

def qOwner (c : Config) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : CTA c :=
  ⟨⟨e.val / c.BK,key_owner_lt c e⟩,⟨t.val / 64,time_owner_lt c t⟩,h⟩
def gOwner (c : Config) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : CTA c :=
  ⟨k,⟨t.val / 64,time_owner_lt c t⟩,h⟩
def rowLane (t : Nat) : Fin 64 := ⟨t % 64,Nat.mod_lt _ (by decide)⟩
def keyLane (c : Config) (e : Nat) : Fin c.BK := ⟨e % c.BK,Nat.mod_lt _ (width_pos c.K)⟩

theorem q_owner_active (c : Config) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    activeQK c (qOwner c h t e) (rowLane t.val) (keyLane c e.val) := by
  simp only [activeQK,qOwner,rowLane,keyLane]
  constructor
  · simpa only [Nat.div_add_mod'] using t.isLt
  · simpa only [Nat.div_add_mod'] using e.isLt

theorem g_owner_active (c : Config) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) :
    activeG c (gOwner c k h t) (rowLane t.val) := by
  simpa only [activeG,gOwner,rowLane,Nat.div_add_mod'] using t.isLt

theorem q_owner_address (c : Config) (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) :
    qAddress c (qOwner c h t e) (rowLane t.val) (keyLane c e.val) =
      (h.val * c.T + t.val) * c.K + e.val := by
  simp only [qAddress,qOwner,rowLane,keyLane,Nat.div_add_mod']

theorem g_owner_address (c : Config) (k : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) :
    gAddress c (gOwner c k h t) (rowLane t.val) = (k.val * c.heads + h.val) * c.T + t.val := by
  simp only [gAddress,gOwner,rowLane,Nat.add_assoc,Nat.div_add_mod']

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
