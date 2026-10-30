import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedExecution

/-! Pure typed source fragments preserve memory and unassigned registers.
This is proved syntactically, independently of the floating backend. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedPreservation
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Keep {α : Type} (names : List RegName) (s t : State α) : Prop :=
  t.mem=s.mem ∧ ∀ d sh n, n∈names → t.regs d sh n=s.regs d sh n

theorem Keep.refl {α : Type} (names : List RegName) (s : State α) : Keep names s s :=
  ⟨rfl,fun _ _ _ _ => rfl⟩

theorem Keep.trans {α : Type} {names : List RegName} {s t u : State α}
    (h : Keep names s t) (k : Keep names t u) : Keep names s u :=
  ⟨k.1.trans h.1,fun d sh n hn => (k.2 d sh n hn).trans (h.2 d sh n hn)⟩

theorem keep_setReg {α : Type} (names : List RegName) (s : State α)
    (name : RegName) (d : TileDType) (sh : TileShape) (v : Values α d sh) (hn : name∉names) :
    Keep names s (s.setReg name d sh v) := by
  refine ⟨rfl,?_⟩
  intro d' sh' n hm
  exact State.setReg_ne_name _ _ _ _ _ _ _ _ (fun he => hn (he ▸ hm))

inductive Pure (names : List RegName) : List ComputeStmt → Prop
  | nil : Pure names []
  | assign (d sh n e) (hn : n∉names) (rest) (hr : Pure names rest) :
      Pure names (.assign d sh n e::rest)
  | branch (cond yes no rest) (hy : Pure names yes) (hn : Pure names no) (hr : Pure names rest) :
      Pure names (.ifThenElse cond yes no::rest)
  | when (cond body rest) (hb : Pure names body) (hr : Pure names rest) :
      Pure names (.ifThen cond body::rest)
  | range (idx lo hi stride body rest) (hn : idx∉names) (hb : Pure names body) (hr : Pure names rest) :
      Pure names (.forRangeDyn idx lo hi stride body::rest)

theorem range_keep {α : Type} [Inhabited α] (A : Algebra α) (names : List RegName)
    (idx : RegName) (body : List ComputeStmt) (hn : idx∉names)
    (hb : ∀ s t, run A body s=some t → Keep names s t)
    (cur stop stride : Nat) (s t : State α) (ht : range A idx cur stop stride body s=some t) :
    Keep names s t := by
  by_cases hz : stride=0
  · subst stride
    rw [range_zero] at ht
    cases ht
    exact Keep.refl names s
  · have main : ∀ fuel cur, stop-cur=fuel → ∀ s t,
        range A idx cur stop stride body s=some t → Keep names s t := by
      intro fuel
      induction fuel using Nat.strong_induction_on with
      | h fuel ih =>
        intro cur hcur s t ht
        by_cases hd : stop≤cur
        · rw [range_done A idx cur stop stride body s hd] at ht
          cases ht
          exact Keep.refl names s
        · have hc : cur<stop := by omega
          rw [range_step A idx cur stop stride body s hz hc] at ht
          cases hx : run A body (s.setReg idx .nat [] (fun _ => cur)) with
          | none => simp [hx] at ht
          | some u =>
            rw [hx,Option.bind_some] at ht
            exact (keep_setReg names s idx .nat [] (fun _ => cur) hn).trans
              ((hb _ _ hx).trans (ih (stop-(cur+stride)) (by omega) (cur+stride) rfl u t ht))
    exact main (stop-cur) cur rfl s t ht

theorem pure_keep {α : Type} [Inhabited α] (A : Algebra α) (names : List RegName)
    (code : List ComputeStmt) (hp : Pure names code) (s t : State α)
    (ht : run A code s=some t) : Keep names s t := by
  induction hp generalizing s t with
  | nil =>
    simp only [run, Option.some.injEq] at ht
    subst t
    exact Keep.refl names s
  | assign d sh n e hn rest hr ih =>
    simp only [run,step] at ht
    cases he : FP.Structural.evalExpr A e s with
    | none => simp [he] at ht
    | some v =>
      simp only [he,Option.bind_some] at ht
      exact (keep_setReg names s n d sh v hn).trans (ih _ _ ht)
  | branch cond yes no rest hy hn hr iy ino ir =>
    simp only [run,step] at ht
    cases he : evalOp A none cond s with
    | none => simp [he] at ht
    | some v =>
      simp only [he,Option.bind_some] at ht
      by_cases hc : v PUnit.unit = Bool.true
      · cases hx : run A yes s with
        | none => simp [hc,hx] at ht
        | some u => exact (iy s u hx).trans (ir u t (by simpa [hc,hx] using ht))
      · cases hx : run A no s with
        | none => simp [hc,hx] at ht
        | some u => exact (ino s u hx).trans (ir u t (by simpa [hc,hx] using ht))
  | when cond body rest hb hr ib ir =>
    simp only [run,step] at ht
    cases he : evalOp A none cond s with
    | none => simp [he] at ht
    | some v =>
      simp only [he,Option.bind_some] at ht
      by_cases hc : v PUnit.unit = Bool.true
      · cases hx : run A body s with
        | none => simp [hc,hx] at ht
        | some u => exact (ib s u hx).trans (ir u t (by simpa [hc,hx] using ht))
      · exact ir s t (by simpa [hc] using ht)
  | range idx lo hi stride body rest hn hb hr ib ir =>
    simp only [run,step] at ht
    cases hl : evalOp A none lo s with
    | none => simp [hl] at ht
    | some lower =>
      cases hh : evalOp A none hi s with
      | none => simp [hl,hh] at ht
      | some upper =>
        cases hs : evalOp A none stride s with
        | none => simp [hl,hh,hs] at ht
        | some delta =>
          simp only [hl,hh,hs,Option.bind_some] at ht
          cases hx : range A idx (lower PUnit.unit) (upper PUnit.unit) (delta PUnit.unit) body s with
          | none => simp [hx] at ht
          | some u =>
            exact (range_keep A names idx body hn ib _ _ _ s u hx).trans
              (ir u t (by simpa [hx] using ht))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedPreservation

