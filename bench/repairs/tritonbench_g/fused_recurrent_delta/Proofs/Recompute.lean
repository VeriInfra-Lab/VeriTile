import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Recompute
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
def h {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α)
    (bh iv : Nat) : Nat → Nat → Nat → α
  | 0 => seed s p m .h0 f.initial bh iv s.BBV
  | t+1 => fun i j => p.add (h s f p m bh iv t i j)
    (p.mul (keyLoad s p m .k bh t i)
      (p.mul (valLoad s p m .v bh t iv s.BBV j) (betaLoad s f p m bh t iv s.BBV j)))
def corrected {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (bh iv t i : Nat) := p.sub (keyLoad s p m .dk (iv*s.BH+bh) t i)
    ((g.correction s.BBV).eval p (fun j => p.mul (valLoad s p m .dv bh t iv s.BBV j) (h s f p m bh iv t i j)))
def output {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t i : Nat) := p.mul
    ((g.query s.BBV).eval p (fun j => p.mul (h s f p m bh iv (t+1) i j) (valLoad s p m .do_ bh t iv s.BBV j))) scale
def memory {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) : Memory α := by
  classical
  exact fun a => if a.plane=iv*s.BH+bh ∧ a.row<n ∧ a.col<s.K then
    match a.region with
    | .dk => p.cut (corrected s f p g m bh iv a.row a.col)
    | .dq => p.cut (output s f p g scale m bh iv a.row a.col)
    | _ => m a
    else m a
theorem memory_current {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane col : Nat) :
    memory s f p g scale m bh iv n ⟨r,plane,n,col⟩=m ⟨r,plane,n,col⟩ := by
  simp [memory]
theorem key_current {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane col : Nat) :
    keyLoad s p (memory s f p g scale m bh iv n) r plane n col=keyLoad s p m r plane n col := by
  simp [keyLoad,load,memory_current]
theorem val_current {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane tile BV col : Nat) :
    valLoad s p (memory s f p g scale m bh iv n) r plane n tile BV col=valLoad s p m r plane n tile BV col := by
  simp [valLoad,load,memory_current]
theorem beta_current {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (plane tile BV col : Nat) :
    betaLoad s f p (memory s f p g scale m bh iv n) plane n tile BV col=betaLoad s f p m plane n tile BV col := by
  simp [betaLoad,val_current,memory_current]

theorem memory_succ {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) :
    rowStore p
      (rowStore p (memory s f p g scale m bh iv n) .dk (iv*s.BH+bh) n 0 s.K
        (corrected s f p g m bh iv n))
      .dq (iv*s.BH+bh) n 0 s.K (output s f p g scale m bh iv n)=
      memory s f p g scale m bh iv (n+1) := by
  funext a
  rcases a with ⟨r,plane,t,i⟩
  by_cases hb : plane=iv*s.BH+bh
  · subst plane
    by_cases hi : i<s.K
    · by_cases ht : t=n
      · subst t
        cases r <;> simp [rowStore,write,memory,hi]
      · have hc : t<n+1 ↔ t<n := by omega
        cases r <;> simp [rowStore,write,memory,hi,ht,hc]
    · cases r <;> simp [rowStore,write,memory,hi]
  · cases r <;> simp [rowStore,write,memory,hb]

@[simp] theorem key_after_store {α : Type} (s : Shape) (p : Ops α) (m : Memory α)
    (r r' : Region) (plane row lo hi bh t i : Nat) (value : Nat → α) (hne : r'≠r) :
    keyLoad s p (rowStore p m r plane row lo hi value) r' bh t i=keyLoad s p m r' bh t i := by
  simp [keyLoad,load,rowStore,write,hne]
@[simp] theorem val_after_store {α : Type} (s : Shape) (p : Ops α) (m : Memory α)
    (r r' : Region) (plane row lo hi bh t iv BV j : Nat) (value : Nat → α) (hne : r'≠r) :
    valLoad s p (rowStore p m r plane row lo hi value) r' bh t iv BV j=valLoad s p m r' bh t iv BV j := by
  simp [valLoad,load,rowStore,write,hne]
@[simp] theorem beta_after_store {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m : Memory α)
    (plane row lo hi bh t iv BV j : Nat) (value : Nat → α) :
    betaLoad s f p (rowStore p m .dk plane row lo hi value) bh t iv BV j=betaLoad s f p m bh t iv BV j := by
  simp [betaLoad,valLoad,load,rowStore,write]

/-- The correction reloads the actual stored dk/dv at recomputation entry;
all preceding updates in this loop affect strictly earlier time rows. -/
theorem loop_full {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) :
    recomputeLoop s f p g scale bh iv n ⟨seed s p m .h0 f.initial bh iv s.BBV,m⟩=
      ⟨h s f p m bh iv n,memory s f p g scale m bh iv n⟩ := by
  induction n with
  | zero => simp only [recomputeLoop,h];congr 1;funext a;simp [memory]
  | succ n ih =>
    rw [recomputeLoop,ih]
    simp only [recomputeStep,key_current,val_current,beta_current]
    simp only [key_after_store,val_after_store,beta_after_store,ne_eq,reduceCtorEq,not_false_eq_true]
    simp only [key_current,val_current,beta_current]
    change (Machine.mk (h s f p m bh iv (n+1)) _)=_
    congr 1
    exact memory_succ s f p g scale m bh iv n

#axiomsClean loop_full
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Recompute
