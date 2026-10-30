import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.IOBinding
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.RealReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.PublicIO
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

/-- Internal source-order recurrence. Public mathematical/native formulas are
independent and are connected to this helper below. -/
def target (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) :
    (IO.program M kind c s seed p).Inputs → (IO.program M kind c s seed p).Outputs :=
  match kind with
  | .forward => fun a o => Fin.cases
      (fun j => M.cast (some .fp32) .real .real
        (KernelReference.forward M (Reference.forwardInputs a) seed j.1.val j.2))
      (fun i => Fin.elim0 i) o
  | .backward => fun a o => Fin.cases
      (fun j => M.cast (some .fp32) .real .real
        (KernelReference.reverse M (Reference.backwardInputs a) c.N (c.N-1-j.1.val) j.2))
      (Fin.cases (fun j => M.cast (some .fp32) .real .real
        (KernelReference.contraction M
          (KernelReference.reverse M (Reference.backwardInputs a) c.N (c.N-1-j.1.val))
          ((Reference.backwardInputs a).saved j.1.val)))
        (Fin.cases (fun j => M.cast (some .fp32) .real .fp32
          (KernelReference.reverse M (Reference.backwardInputs a) c.N c.N j))
          (fun i => Fin.elim0 i))) o

theorem implements (M : Algebra α) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (hn : 0<c.N) :
    (IO.program M kind c s seed p).Implements (target M kind c s seed p) := by
  apply ProgramIO.Implements.intro
  · intro a bounds hb u _
    exact IOBinding.safe M kind c s seed p hn a bounds hb u
  · intro a u hin
    let initial := (IO.launch M kind c s seed p).initial u
    have hin' : (IO.program M kind c s seed p).InputBinding a initial := by
      intro i j active
      simpa only [initial,Invocation.initial_mem] using hin i j active
    obtain ⟨v,facts⟩ := KernelExecution.complete M kind c s seed p initial hn rfl rfl rfl
    refine ⟨v,?_,?_,IOBinding.frame M kind c s seed p a u v facts.frame⟩
    · cases kind <;> exact (runSchedule_singleton _ _).trans facts.success
    · cases kind with
      | forward =>
        intro o j _
        fin_cases o
        obtain ⟨t,e,j,⟨⟩⟩ := j
        change mapCell id (v.mem "O" (address c p t.val e.val j.val))=_
        rw [facts.outputs t e j]
        change Cell.mk .real (M.cast (some .fp32) .real .real
          (KernelReference.forward M (KernelReference.read c s p initial) seed t.val (e,j,PUnit.unit)))=_
        rw [IOBinding.forward_eq M c s seed p a initial hin' t.val t.isLt]
        rfl
      | backward =>
        intro o j _
        fin_cases o
        · obtain ⟨t,e,j,⟨⟩⟩ := j
          change mapCell id (v.mem "DI" (address c p t.val e.val j.val))=_
          rw [facts.outputs.1 t e j]
          change Cell.mk .real (M.cast (some .fp32) .real .real
            (KernelReference.reverse M (KernelReference.read c s p initial) c.N
              (c.N-1-t.val) (e,j,PUnit.unit)))=_
          rw [IOBinding.reverse_eq M c s seed p a initial hin' _ (by omega)]
          rfl
        · obtain ⟨t,⟨⟩⟩ := j
          change mapCell id (v.mem "DG" (gradientAddress c p t.val))=_
          rw [facts.outputs.2.1 t]
          change Cell.mk .real (M.cast (some .fp32) .real .real
            (KernelReference.contraction M
              (KernelReference.reverse M (KernelReference.read c s p initial) c.N (c.N-1-t.val))
              ((KernelReference.read c s p initial).saved t.val)))=_
          rw [IOBinding.reverse_eq M c s seed p a initial hin' _ (by omega),
            IOBinding.saved_eq M c s seed p a initial hin' t.val t.isLt]
          rfl
        · obtain ⟨e,j,⟨⟩⟩ := j
          change mapCell id (v.mem "DL" (dseedAddress c p e.val j.val))=_
          rw [facts.outputs.2.2 e j]
          change Cell.mk .fp32 (M.cast (some .fp32) .real .fp32
            (KernelReference.reverse M (KernelReference.read c s p initial) c.N c.N (e,j,PUnit.unit)))=_
          rw [IOBinding.reverse_eq M c s seed p a initial hin' _ (Nat.le_refl _)]
          rfl

theorem real_target (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) :
    target KernelRealReference.algebra kind c s seed p=Reference.real kind c s seed p := by
  funext a o
  cases kind with
  | forward =>
    fin_cases o
    funext ⟨t,e,j,⟨⟩⟩
    exact KernelRealReference.forward_formula _ seed t.val e j
  | backward =>
    fin_cases o
    · funext ⟨t,e,j,⟨⟩⟩
      change KernelReference.reverse KernelRealReference.algebra _ c.N (c.N-1-t.val) _=_
      exact congrFun (KernelReal.reverse_value c (Reference.backwardInputs a) t) (e,j,PUnit.unit)
    · funext ⟨t,⟨⟩⟩
      change KernelReference.contraction KernelRealReference.algebra _ _=_
      rw [KernelReal.reverse_value c (Reference.backwardInputs a) t,
        KernelRealReference.contraction_formula]
      rfl
    · funext ⟨e,j,⟨⟩⟩
      change KernelReference.reverse KernelRealReference.algebra _ c.N c.N _=_
      simpa using KernelRealReference.reverse_formula (Reference.backwardInputs a) c.N c.N (Nat.le_refl _) e j

theorem real (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (hn : 0<c.N) :
    (IO.program KernelRealReference.algebra kind c s seed p).Implements (Reference.real kind c s seed p) := by
  rw [←real_target]
  exact implements _ kind c s seed p hn

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.PublicIO
