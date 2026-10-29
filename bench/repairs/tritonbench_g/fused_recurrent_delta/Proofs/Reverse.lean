import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Reverse
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
def h {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv : Nat) : Nat → Nat → Nat → α
  | 0 => seed s p m .dht f.terminalGrad bh iv s.BBV
  | n+1 => fun i j =>
    let t:=s.T-1-n
    let k:=keyLoad s p m .k bh t
    let q:=fun i => p.mul (keyLoad s p m .q bh t i) scale
    let do_:=valLoad s p m .do_ bh t iv s.BBV
    let dh:=fun i j => p.add (h s f p g scale m bh iv n i j) (p.mul (q i) (do_ j))
    let rawdv:=fun j => (g.key s.BK).eval p (fun i => p.mul (dh i j) (k i))
    p.sub (dh i j) (p.mul (k i) (p.mul (rawdv j) (betaLoad s f p m bh t iv s.BBV j)))
def enhanced {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t i j : Nat) := p.add (h s f p g scale m bh iv (s.T-1-t) i j)
      (p.mul (p.mul (keyLoad s p m .q bh t i) scale) (valLoad s p m .do_ bh t iv s.BBV j))
def dk {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t i : Nat) := (g.value s.BBV).eval p (fun j => p.mul (enhanced s f p g scale m bh iv t i j)
      (p.mul (valLoad s p m .v bh t iv s.BBV j) (betaLoad s f p m bh t iv s.BBV j)))
def rawDv {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t j : Nat) := (g.key s.BK).eval p (fun i => p.mul (enhanced s f p g scale m bh iv t i j) (keyLoad s p m .k bh t i))
def dv {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t j : Nat) := p.mul (rawDv s f p g scale m bh iv t j) (betaLoad s f p m bh t iv s.BBV j)
def dbeta {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m : Memory α)
    (bh iv t j : Nat) := p.mul (rawDv s f p g scale m bh iv t j) (valLoad s p m .v bh t iv s.BBV j)
def memory {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) : Memory α := by
  classical
  exact fun a => if a.row<s.T ∧ s.T-n≤a.row then
    match a.region with
    | .dk => if a.plane=iv*s.BH+bh ∧ a.col<s.K then p.cut (dk s f p g scale m bh iv a.row a.col) else m a
    | .dv => if a.plane=bh ∧ iv*s.BBV≤a.col ∧ a.col<min ((iv+1)*s.BBV) s.V then
        p.cut (dv s f p g scale m bh iv a.row (a.col-iv*s.BBV)) else m a
    | .dbeta => if a.plane=iv*s.BH+bh then
        if f.vectorBeta then
          if iv*s.BBV≤a.col ∧ a.col<min ((iv+1)*s.BBV) s.V then
            p.cut (dbeta s f p g scale m bh iv a.row (a.col-iv*s.BBV)) else m a
        else if a.col=0 then p.cut ((g.beta s.BBV).eval p (dbeta s f p g scale m bh iv a.row)) else m a
      else m a
    | _ => m a
    else m a

@[simp] theorem readonly {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane t col : Nat)
    (hr : r≠.dk ∧ r≠.dv ∧ r≠.dbeta) :
    memory s f p g scale m bh iv n ⟨r,plane,t,col⟩=m ⟨r,plane,t,col⟩ := by
  cases r <;> simp_all [memory]
@[simp] theorem key_readonly {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane t col : Nat)
    (hr : r≠.dk ∧ r≠.dv ∧ r≠.dbeta) :
    keyLoad s p (memory s f p g scale m bh iv n) r plane t col=keyLoad s p m r plane t col := by
  simp [keyLoad,load,hr]
@[simp] theorem val_readonly {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (r : Region) (plane t tile BV col : Nat)
    (hr : r≠.dk ∧ r≠.dv ∧ r≠.dbeta) :
    valLoad s p (memory s f p g scale m bh iv n) r plane t tile BV col=valLoad s p m r plane t tile BV col := by
  simp [valLoad,load,hr]
@[simp] theorem beta_readonly {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n plane t tile BV col : Nat) :
    betaLoad s f p (memory s f p g scale m bh iv n) plane t tile BV col=betaLoad s f p m plane t tile BV col := by
  simp [betaLoad]

theorem memory_succ {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (hn : n<s.T) :
    let t:=s.T-1-n
    let mid:=rowStore p
      (rowStore p (memory s f p g scale m bh iv n) .dk (iv*s.BH+bh) t 0 s.K (fun i => dk s f p g scale m bh iv t i))
      .dv bh t (iv*s.BBV) (min ((iv+1)*s.BBV) s.V) (fun v => dv s f p g scale m bh iv t (v-iv*s.BBV))
    (if f.vectorBeta then rowStore p mid .dbeta (iv*s.BH+bh) t (iv*s.BBV) (min ((iv+1)*s.BBV) s.V)
        (fun v => dbeta s f p g scale m bh iv t (v-iv*s.BBV))
      else rowStore p mid .dbeta (iv*s.BH+bh) t 0 1 (fun _ => (g.beta s.BBV).eval p (fun j => dbeta s f p g scale m bh iv t j)))=
      memory s f p g scale m bh iv (n+1) := by
  dsimp only
  funext a
  rcases a with ⟨r,plane,t,j⟩
  by_cases ht : t=s.T-1-n
  · subst t
    have hlt : s.T-1-n<s.T := by omega
    have hprev : ¬s.T-n≤s.T-1-n := by omega
    have hnext : s.T-(n+1)≤s.T-1-n := by omega
    cases hv : f.vectorBeta <;> cases r <;>
      simp [rowStore,write,memory,hlt,hprev,hnext,hv,and_assoc,Nat.lt_one_iff]
      <;> split_ifs <;> simp_all
  · have hc : (t<s.T ∧ s.T≤t+n) ↔ (t<s.T ∧ s.T≤t+(n+1)) := by omega
    cases hv : f.vectorBeta <;> cases r <;>
      simp [rowStore,write,memory,ht,hc,hv]

/-- Reverse state and every byte-addressed scratch cell are derived from the
reverse-loop entry memory. No assumed dk/dv/dbeta values or carry chain. -/
theorem loop_full {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv n : Nat) (hn : n≤s.T) :
    reverseLoop s f p g scale bh iv n ⟨seed s p m .dht f.terminalGrad bh iv s.BBV,m⟩=
      ⟨h s f p g scale m bh iv n,memory s f p g scale m bh iv n⟩ := by
  induction n with
  | zero => simp only [reverseLoop,h];congr 1;funext a;simp [memory]
  | succ n ih =>
    rw [reverseLoop,ih (by omega)]
    have ht : s.T-1-(s.T-1-n)=n := by omega
    simp only [reverseStep]
    simp only [key_readonly,val_readonly,beta_readonly,ne_eq,reduceCtorEq,not_false_eq_true,and_self]
    change (Machine.mk (h s f p g scale m bh iv (n+1)) _)=_
    congr 1
    have hm:=memory_succ s f p g scale m bh iv n (by omega)
    dsimp only [dk,dv,dbeta,rawDv,enhanced] at hm
    simpa only [ht] using hm

#axiomsClean loop_full
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Reverse
