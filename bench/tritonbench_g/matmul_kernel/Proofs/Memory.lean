import bench.tritonbench_g.matmul_kernel.Proofs.Geometry
namespace VeriTile.Bench.Repairs.MatmulKernel
variable {α : Type}
theorem step_readback (c : Config) (p : Ops α) (w : Work) (m : Memory α) :
    step c p w m .c (c.C.addr w.1 w.2)=value c p m w := by simp [step]
theorem step_frame (c : Config) (p : Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (h : ¬Writes c w r a) : step c p w m r a=m r a := by
  simpa only [step,Writes] using if_neg h

theorem value_inputs (c : Config) (p : Ops α) (m n : Memory α)
    (ha : m .a=n .a) (hb : m .b=n .b) (w : Work) : value c p m w=value c p n w := by
  simp only [value,block,atom,readA,readB,ha,hb]
theorem step_inputs (c : Config) (p : Ops α) (w : Work) (m : Memory α) :
    step c p w m .a=m .a ∧ step c p w m .b=m .b := by
  constructor <;> funext a <;> simp [step]
theorem step_value (c : Config) (p : Ops α) (w v : Work) (m : Memory α) :
    value c p (step c p w m) v=value c p m v :=
  value_inputs c p _ m (step_inputs c p w m).1 (step_inputs c p w m).2 v

theorem execute_frame (c : Config) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (h : ∀ w∈ws,¬Writes c w r a) : execute c p ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change execute c p ws (step c p w m) r a=_
    rw [ih _ (fun v hv => h v (by simp [hv]))]
    exact step_frame c p w m r a (h w (by simp))
theorem execute_inputs (c : Config) (p : Ops α) (ws : List Work) (m : Memory α) :
    execute c p ws m .a=m .a ∧ execute c p ws m .b=m .b := by
  constructor <;> funext a <;> apply execute_frame <;> intro w _ <;> simp [Writes]
theorem execute_readback (c : Config) (hc : c.Valid) (p : Ops α) (ws : List Work) (m : Memory α)
    (active : ∀ w∈ws,c.Active w) (v : Work) (hv : v∈ws) :
    execute c p ws m .c (c.C.addr v.1 v.2)=value c p m v := by
  induction ws generalizing m with
  | nil => simp at hv
  | cons w ws ih =>
    change execute c p ws (step c p w m) .c (c.C.addr v.1 v.2)=_
    by_cases tail : v∈ws
    · rw [ih _ (fun v hv => active v (by simp [hv])) tail,step_value]
    · have same : v=w := (List.mem_cons.mp hv).resolve_right tail
      subst v
      rw [execute_frame c p ws (step c p w m) .c (c.C.addr w.1 w.2) (by
        intro z hz h
        have eq := c_injective c hc w z (active w (by simp)) (active z (by simp [hz])) h.2
        exact tail (eq.symm ▸ hz))]
      exact step_readback c p w m

theorem execute_append (c : Config) (p : Ops α) (xs ys : List Work) (m : Memory α) :
    execute c p (xs++ys) m=execute c p ys (execute c p xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih (step c p w m)
theorem run_readback (c : Config) (hc : c.Valid) (p : Ops α) (m : Memory α) (w : Work) (hw : c.Active w) :
    run c p m .c (c.C.addr w.1 w.2)=value c p m w :=
  execute_readback c hc p _ m (schedule_active c) w (schedule_covers c hc w hw)
theorem run_frame (c : Config) (p : Ops α) (m : Memory α) (r : Region) (a : Nat)
    (h : ∀ w,c.Active w → ¬Writes c w r a) : run c p m r a=m r a :=
  execute_frame c p _ m r a (fun w hw => h w (schedule_active c w hw))
theorem run_readonly (c : Config) (p : Ops α) (m : Memory α) (r : Region) (a : Nat) (h : r≠.c) :
    run c p m r a=m r a := by
  apply run_frame
  intro w _
  simp [Writes,h]
#axiomsClean execute_readback
#axiomsClean run_frame
#axiomsClean run_readback
end VeriTile.Bench.Repairs.MatmulKernel
