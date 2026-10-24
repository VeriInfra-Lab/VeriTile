import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Forward
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
def h {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (bh iv : Nat) : Nat → Nat → Nat → α
  | 0 => seed s p m .h0 f.initial bh iv s.FBV
  | t+1 => fun i j =>
    let prev:=h s f p g m bh iv t
    let k:=keyLoad s p m .k bh t
    let r:=p.sub (valLoad s p m .v bh t iv s.FBV j)
      ((g.key s.BK).eval p (fun k' => p.mul (prev k' j) (k k')))
    p.add (prev i j) (p.mul (k i) (p.mul r (betaLoad s f p m bh t iv s.FBV j)))
def residual {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (bh iv t j : Nat) := p.sub (valLoad s p m .v bh t iv s.FBV j)
      ((g.key s.BK).eval p (fun i => p.mul (h s f p g m bh iv t i j) (keyLoad s p m .k bh t i)))
def output {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t j : Nat) :=
  (g.readout s.BK).eval p (fun i => p.mul (h s f p g m bh iv (t+1) i j) (p.mul (keyLoad s p m .q bh t i) scale))
def memory {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) : Memory α := by
  classical
  exact fun a => if a.plane=bh ∧ a.row<n ∧ iv*s.FBV≤a.col ∧ a.col<min ((iv+1)*s.FBV) s.V then
    match a.region with
    | .v => p.cut (residual s f p g m bh iv a.row (a.col-iv*s.FBV))
    | .o => p.cut (output s f p g scale m bh iv a.row (a.col-iv*s.FBV))
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
      (rowStore p (memory s f p g scale m bh iv n) .v bh n (iv*s.FBV) (min ((iv+1)*s.FBV) s.V)
        (fun v => residual s f p g m bh iv n (v-iv*s.FBV)))
      .o bh n (iv*s.FBV) (min ((iv+1)*s.FBV) s.V)
        (fun v => output s f p g scale m bh iv n (v-iv*s.FBV))=
      memory s f p g scale m bh iv (n+1) := by
  funext a
  rcases a with ⟨r,plane,t,v⟩
  by_cases hb : plane=bh
  · subst plane
    by_cases hv : iv*s.FBV≤v ∧ v<min ((iv+1)*s.FBV) s.V
    · by_cases ht : t=n
      · subst t
        cases r <;> simp [rowStore,write,memory,hv]
      · have hc : t<n+1 ↔ t<n := by omega
        cases r <;> simp [rowStore,write,memory,hv,ht,hc]
    · have hv' : ¬(iv*s.FBV≤v ∧ v<(iv+1)*s.FBV ∧ v<s.V) := by simpa using hv
      cases r <;> simp [rowStore,write,memory,hv',and_assoc]
  · cases r <;> simp [rowStore,write,memory,hb]

/-- No materialized carry or successful-loop premise: both the register
state and the complete current memory are derived for every prefix. -/
theorem loop_full {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) :
    forwardLoop s f p g scale bh iv n ⟨seed s p m .h0 f.initial bh iv s.FBV,m⟩=
      ⟨h s f p g m bh iv n,memory s f p g scale m bh iv n⟩ := by
  induction n with
  | zero => simp only [forwardLoop,h];congr 1;funext a;simp [memory]
  | succ n ih =>
    rw [forwardLoop,ih]
    simp only [forwardStep,key_current,val_current,beta_current]
    change (Machine.mk (h s f p g m bh iv (n+1)) _)=_
    congr 1
    exact memory_succ s f p g scale m bh iv n
theorem program_full {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv : Nat) :
    forwardProgram s f p g scale bh iv m=
      if f.final then stateStore s p (memory s f p g scale m bh iv s.T) .ht bh iv s.FBV (h s f p g m bh iv s.T)
      else memory s f p g scale m bh iv s.T := by
  simp only [forwardProgram,loop_full]

#axiomsClean loop_full
#axiomsClean program_full
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Forward
