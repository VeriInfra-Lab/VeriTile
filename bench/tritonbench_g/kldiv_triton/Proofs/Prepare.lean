import bench.tritonbench_g.kldiv_triton.Proofs.Runtime

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

def Stable {M α β : Type} (ops : Ops M α β) (before after : World M) : Prop :=
  ∀ kept,(∀ old∈kept,before.live old) → Frame ops kept before after

theorem Stable.refl {M α β : Type} (ops : Ops M α β) (s : World M) : Stable ops s s :=
  fun kept _ => L2NormBwd.Host.Frame.refl ops kept s.toWorld

theorem Stable.live {M α β : Type} {ops : Ops M α β} {a b : World M}
    (h : Stable ops a b) (old : Allocation) (live : a.live old) : b.live old :=
  (h [] (by simp)).live old live

theorem Stable.trans {M α β : Type} {ops : Ops M α β} {a b c : World M}
    (ab : Stable ops a b) (bc : Stable ops b c) : Stable ops a c := by
  intro kept live
  exact (ab kept live).trans (bc kept (fun old ho => ab.live old (live old ho)))

theorem Stable.logical {M α β : Type} {ops : Ops M α β} {a b : World M}
    (h : Stable ops a b) (neg : Negation α) (v : View) (live : a.live v.storage) (backed : v.backed)
    (i : Nat) (active : i<v.numel) : logical ops neg b.memory v i=logical ops neg a.memory v i := by
  exact (h [v.storage] (by simpa using live)).logical neg v (by simp) backed i active

theorem external_stable {M α β A : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt) (op : Op A)
    (ordinary : match op with | .copyInto _ _ => False | _ => True) (before : World M) :
    Stable ops before (rt.perform op before).state := by
  intro kept live
  have h := lib.frame op before kept live
  cases op <;> first | exact h | exact False.elim ordinary

structure Prepared {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (input output : View) (before after : World M) : Prop where
  shape : output.shape=input.shape
  positive : output.negative=Bool.false
  backed : output.backed
  live : after.live output.storage
  owner : output.storage.owner=input.storage.owner
  values : ∀ i,i < input.numel → logical ops neg after.memory output i=logical ops neg before.memory input i
  storage : output.storage=input.storage ∨ ∀ old,before.live old → output.storage.separate old

structure PreparedMatrix {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (input output : Matrix) (before after : World M) : Prop extends Prepared ops neg input.view output.view before after where
  rows : output.rows=input.rows
  cols : output.cols=input.cols
  unit : Index.UnitColumns output

/-- Coordinate-preserving ordinary resolve/contiguous/clone composition. Both
failure prefixes preserve caller allocations; actual returned metadata is used. -/
theorem prepareMatrix_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (input : Matrix) (snapshot : Bool) (s : World M) (live : s.live input.view.storage) (backed : input.view.backed) :
    Ensures (fun out after => PreparedMatrix ops neg input out s after ∧ Stable ops s after ∧
      (snapshot=Bool.true → ∀ old,s.live old → out.view.storage.separate old))
      (fun _ after => Stable ops s after) (prepareMatrix rt input snapshot s) := by
  unfold prepareMatrix
  have first := external_stable ops neg arith rt lib (.resolveMatrix input) (by trivial) s
  cases e1 : rt.perform (.resolveMatrix input) s with
  | error why failed => simpa only [e1,Result.state] using first
  | ok resolved s1 =>
    simp only [e1,Result.state] at first
    simp only [Result.bind]
    obtain ⟨rows1,cols1,pos1,copy1⟩ := lib.resolveMatrix input s resolved s1 e1
    have finish (out : Matrix) (after : World M)
        (rows2 : out.rows=resolved.rows) (cols2 : out.cols=resolved.cols) (unit : Index.UnitColumns out)
        (positive : out.view.negative=Bool.false) (copy2 : Copy ops neg resolved.view out.view s1 after)
        (second : Stable ops s1 after)
        (fresh : snapshot=Bool.true → ∀ old,s1.live old → out.view.storage.separate old) :
        PreparedMatrix ops neg input out s after ∧ Stable ops s after ∧
          (snapshot=Bool.true → ∀ old,s.live old → out.view.storage.separate old) := by
      have shape : out.view.shape=input.view.shape := by rw [out.shape,input.shape,rows2,cols2,rows1,cols1]
      refine ⟨⟨⟨shape,positive,copy2.backed,copy2.live,copy2.owner.trans copy1.owner,?_,?_⟩,
        rows2.trans rows1,cols2.trans cols1,unit⟩,first.trans second,?_⟩
      · intro i hi
        rw [copy2.values i (by rw [copy1.size];exact hi),copy1.values i hi]
      · rcases copy2.storage with same | new
        · rw [same];exact copy1.storage
        · exact Or.inr (fun old ho => new old (first.live old ho))
      · intro snap old ho
        exact fresh snap old (first.live old ho)
    cases snapshot
    · simp only [Bool.false_eq_true,if_false]
      have second := external_stable ops neg arith rt lib (.contiguous resolved) (by trivial) s1
      cases e2 : rt.perform (.contiguous resolved) s1 with
      | error why failed =>
        simp only [e2,Result.state] at second
        exact first.trans second
      | ok out after =>
        simp only [e2,Result.state] at second
        obtain ⟨r,c,u,p,copy⟩ := lib.contiguous resolved s1 out after e2
        simpa [Ensures] using finish out after r c u (p.trans pos1) copy second (by intro h; cases h)
    · simp only [if_true]
      have second := external_stable ops neg arith rt lib (.cloneMatrix resolved) (by trivial) s1
      cases e2 : rt.perform (.cloneMatrix resolved) s1 with
      | error why failed =>
        simp only [e2,Result.state] at second
        exact first.trans second
      | ok out after =>
        simp only [e2,Result.state] at second
        obtain ⟨r,c,u,p,copy,fresh⟩ := lib.cloneMatrix resolved s1 out after e2
        simpa [Ensures] using finish out after r c (Index.UnitColumns.of_stride u) p copy second (fun _ => fresh)

theorem prepareGradient_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) (lib : Library ops neg arith rt)
    (input : View) (s : World M) :
    Ensures (fun out after => Prepared ops neg input out s after ∧ Stable ops s after ∧
      (∀ old,s.live old → out.storage.separate old))
      (fun _ after => Stable ops s after) (prepareGradient rt input s) := by
  unfold prepareGradient
  have first := external_stable ops neg arith rt lib (.resolve input) (by trivial) s
  cases e1 : rt.perform (.resolve input) s with
  | error why failed => simpa only [e1,Result.state] using first
  | ok resolved s1 =>
    simp only [e1,Result.state] at first
    simp only [Result.bind]
    obtain ⟨shape1,pos1,copy1⟩ := lib.resolve input s resolved s1 e1
    have second := external_stable ops neg arith rt lib (.clone resolved) (by trivial) s1
    cases e2 : rt.perform (.clone resolved) s1 with
    | error why failed =>
      simp only [e2,Result.state] at second
      exact first.trans second
    | ok out after =>
      simp only [e2,Result.state] at second
      obtain ⟨shape2,pos2,copy2,fresh⟩ := lib.clone resolved s1 out after e2
      refine ⟨⟨shape2.trans shape1,pos2,copy2.backed,copy2.live,copy2.owner.trans copy1.owner,?_,
        Or.inr (fun old ho => fresh old (first.live old ho))⟩,first.trans second,
        fun old ho => fresh old (first.live old ho)⟩
      intro i hi
      rw [copy2.values i (by rw [copy1.size];exact hi),copy1.values i hi]

#axiomsClean prepareMatrix_spec
#axiomsClean prepareGradient_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
