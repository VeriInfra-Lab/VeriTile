import bench.tritonbench_g.matmul_triton1.IO

namespace VeriTile.Bench.Repairs.MatmulTriton1.IOResults
open VeriTile.Triton FP.Structural Revision001 MatmulIO
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
variable {α : Type} [Inhabited α]

theorem source_eq (p : Core.Parameters) : Kernel.source p=Core.source p := rfl

theorem launch_initial (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (pid : Nat) (s : State α) :
    (launch rest P p pid).initial s=GridExecution.atProgram s pid := rfl

theorem schedule_eq (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (order : List Nat) (s : State α) :
    runSchedule (order.map (launch rest P p)) s=
      GridExecution.sequence (Core.one rest P p) order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid order ih =>
    simp only [List.map_cons,runSchedule,GridExecution.sequence]
    have one : (launch rest P p pid).run s=Core.one rest P p pid s := rfl
    rw [one]
    cases Core.one rest P p pid s <;> simp [ih]

theorem read_left (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (s : State α)
    (hin : (io rest P p).InputBinding xs s) (r : Fin p.rows) (k : Fin p.innerSize) :
    (s.mem p.input (r.val*p.inputRow+k.val*p.inputInner)).read .real=
      xs (0 : Fin 2) (r,k,()) := hin (0 : Fin 2) (r,k,()) trivial

theorem read_right (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (s : State α)
    (hin : (io rest P p).InputBinding xs s) (k : Fin p.innerSize) (c : Fin p.columns) :
    (s.mem p.weight (k.val*p.weightInner+c.val*p.weightColumn)).read .real=
      xs (1 : Fin 2) (k,c,()) := hin (1 : Fin 2) (k,c,()) trivial

theorem precision_target (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (s : State α)
    (hin : (io rest P p).InputBinding xs s) (r : Fin p.rows) (c : Fin p.columns) (n : Nat) :
    Core.logicalAccum P p s r.val c.val n=precisionAccum P p xs r c n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Core.logicalAccum,precisionAccum,ih]
    congr 1
    · funext k
      have mask : k.val<Core.inner p-n*p.blockInner ↔
          n*p.blockInner+k.val<p.innerSize := by unfold Core.inner; omega
      simp only [Core.logicalA,left,mask]
      split <;> rename_i h
      · exact read_left rest P p xs s hin r ⟨_,h⟩
      · rfl
    · funext k
      have mask : k.val<Core.inner p-n*p.blockInner ↔
          n*p.blockInner+k.val<p.innerSize := by unfold Core.inner; omega
      simp only [Core.logicalB,right,mask]
      split <;> rename_i h
      · exact read_right rest P p xs s hin ⟨_,h⟩ c
      · rfl

theorem real_target (rest : Algebra ℝ) (p : Core.Parameters)
    (xs : ChannelValues ℝ (inputs p)) (s : State ℝ)
    (hin : (io rest Core.realPrimitives p).InputBinding xs s)
    (r : Fin p.rows) (c : Fin p.columns) :
    Core.realMatmul p s r.val c.val=(realTarget p xs) (0 : Fin 1) (r,c,()) := by
  apply RealSum.sum_congr
  intro k hk
  change k<p.innerSize at hk
  change (s.mem p.input (r.val*p.inputRow+k*p.inputInner)).read .real *
    (s.mem p.weight (k*p.weightInner+c.val*p.weightColumn)).read .real=
    left p xs 0 r k * right p xs 0 c k
  rw [read_left rest Core.realPrimitives p xs s hin r ⟨k,hk⟩,
    read_right rest Core.realPrimitives p xs s hin ⟨k,hk⟩ c]
  simp [left,right,hk]

theorem frame (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (before final : State α) (value : Core.LogicalIndex p → α)
    (post : RawKernel.Post p before final value) : (io rest P p).Frame xs before final := by
  intro r offset hout
  apply post.2
  by_cases he : r=p.output
  · right
    intro i _
    have h := (hout.1 (0 : Fin 1) (i.1,i.2,()) trivial)
    exact h.resolve_left (fun hne => hne he)
  · exact Or.inl he

theorem complete (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (xs : ChannelValues α (inputs p)) (initial : State α)
    (hin : (io rest P p).InputBinding xs initial) :
    ∃ final, runSchedule ((io rest P p).schedule xs) initial=some final ∧
      (io rest P p).OutputBinding xs (precisionTarget P p xs) final ∧
      (io rest P p).Frame xs initial final := by
  obtain ⟨finite,typed,hf,ht,hm,hp,hq⟩ :=
    RawKernel.execute rest P p canonical ready storage geometry ha hb initial
  refine ⟨typed,?_,?_,frame rest P p xs initial typed _ hq⟩
  · exact (schedule_eq rest P p _ initial).trans ht
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change mapCell id (typed.mem p.output (j.1.val*p.outputRow+j.2.1.val*p.outputColumn))=
      Cell.mk .real (precisionAccum P p xs j.1 j.2.1 (Core.steps p))
    rw [show typed.mem p.output (j.1.val*p.outputRow+j.2.1.val*p.outputColumn)=
      Cell.mk .real (Core.logical P p initial j.1.val j.2.1.val) from hq.1 (j.1,j.2.1) trivial]
    change Cell.mk .real (Core.logicalAccum P p initial j.1.val j.2.1.val (Core.steps p))=_
    rw [precision_target rest P p xs initial hin]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOResults
