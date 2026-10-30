import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Base

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Input (α : Type) := RegionName → Nat → Nat → Nat → α

def seed {α : Type} (p : Ops α) (U : Bool) (input : Input α) (bh d : Nat) : α :=
  if U then p.add p.zero (input "h0" bh 0 d) else p.zero

/-- Mathematical/native recurrence over entry inputs only. No interpreter or output buffer. -/
def state {α : Type} (p : Ops α) (U : Bool) (input : Input α) (bh d : Nat) : Nat → α
  | 0 => seed p U input bh d
  | n+1 => p.add (p.mul (input "g" bh n d) (state p U input bh d n)) (input "x" bh n d)

def terminal {α : Type} (p : Ops α) (V : Bool) (input : Input α) (bh d : Nat) : α :=
  if V then input "dht" bh 0 d else p.zero

/-- The reverse recurrence is seeded by the returned final-state cotangent. -/
def adjoint {α : Type} (p : Ops α) (V : Bool) (input : Input α)
    (T bh d : Nat) : Nat → α
  | 0 => terminal p V input bh d
  | n+1 => p.mul (p.add (adjoint p V input T bh d n) (input "do" bh (T-1-n) d))
      (input "g" bh (T-1-n) d)

def dx {α : Type} (p : Ops α) (V : Bool) (input : Input α) (T bh d t : Nat) : α :=
  p.add (adjoint p V input T bh d (T-1-t)) (input "do" bh t d)

def previous {α : Type} (p : Ops α) (U : Bool) (input : Input α) (bh d t : Nat) : α :=
  if 0<t then state p U input bh d t
  else if U then input "h0" bh 0 d else p.zero

def dg {α : Type} (p : Ops α) (U V : Bool) (input : Input α) (T bh d t : Nat) : α :=
  p.mul (dx p V input T bh d t) (previous p U input bh d t)

def realOps : Ops ℝ := ⟨0,fun a b => a+b,fun a b => a*b⟩

def rowAddr (bh T D t d : Nat) := (bh*T+t)*D+d
def stateAddr (bh D d : Nat) := bh*D+d

theorem row_bound (BH T D bh t d : Nat) (hb : bh<BH) (ht : t<T) (hd : d<D) :
    rowAddr bh T D t d < BH*T*D := by
  dsimp [rowAddr]
  have a : bh*T+t < BH*T := by nlinarith
  nlinarith

theorem state_bound (BH D bh d : Nat) (hb : bh<BH) (hd : d<D) :
    stateAddr bh D d < BH*D := by dsimp [stateAddr];nlinarith

theorem row_injective (T D bh bh' t t' d d' : Nat)
    (ht : t<T) (ht' : t'<T) (hd : d<D) (hd' : d'<D)
    (same : rowAddr bh T D t d=rowAddr bh' T D t' d') :
    bh=bh' ∧ t=t' ∧ d=d' := by
  have hD : 0<D := by omega
  have hT : 0<T := by omega
  have rows := congrArg (fun n => n/D) same
  simp [rowAddr,Nat.add_div,Nat.div_eq_of_lt hd,Nat.div_eq_of_lt hd',Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd',Nat.not_le.mpr hd,Nat.not_le.mpr hd',hD] at rows
  have heads := congrArg (fun n => n/T) rows
  simp [Nat.add_div,Nat.div_eq_of_lt ht,Nat.div_eq_of_lt ht',Nat.mod_eq_of_lt ht,Nat.mod_eq_of_lt ht',Nat.not_le.mpr ht,Nat.not_le.mpr ht',hT] at heads
  subst bh'
  have times : t=t' := by omega
  subst t'
  exact ⟨rfl,rfl,by dsimp [rowAddr] at same;omega⟩

theorem grid_owner (D BD d : Nat) (pos : 0<BD) (hd : d<D) :
    d/BD < (D+BD-1)/BD ∧ d%BD<BD ∧ d/BD*BD+d%BD=d := by
  refine ⟨?_,Nat.mod_lt _ pos,by simpa [Nat.mul_comm] using Nat.div_add_mod d BD⟩
  rw [Nat.div_lt_iff_lt_mul pos]
  have hm := Nat.mod_add_div (D+BD-1) BD
  have hr := Nat.mod_lt (D+BD-1) pos
  have hD : 0<D := by omega
  have hs : D+BD-1+1=D+BD := by omega
  nlinarith

#specNonCircular state avoiding [FP.Structural.exec]
#specNonCircular adjoint avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
