import bench.tritonbench_g.matmul_leakyrelu.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.IOFacts
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
variable {α : Type} [Inhabited α]

theorem entered (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (pid : Nat) (s : State α) :
    (job M D p pid).initial s=Whole.instanceState p.request pid s := rfl

theorem schedule_run (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : List Nat) (s : State α) :
    runSchedule (order.map (job M D p)) s=run M D p.request order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    change (exec (FPFused.backend M D) (Whole.contractedKernel p.request)
      (Whole.instanceState p.request pid s)).bind (runSchedule (rest.map (job M D p))) =
      (exec (FPFused.backend M D) (Whole.contractedKernel p.request)
        (Whole.instanceState p.request pid s)).bind (run M D p.request rest)
    cases h : exec (FPFused.backend M D) (Whole.contractedKernel p.request)
      (Whole.instanceState p.request pid s) <;> simp [ih]

theorem input_a (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (vals : Inputs α p) (s : State α) (hin : (io M D p order).InputBinding vals s)
    (i : Fin p.rows) (k : Fin p.inner) :
    Whole.inputA p.request s i.val k.val=matrixA vals i k :=
  hin (0 : Fin 2) (i,k,()) trivial

theorem input_b (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (vals : Inputs α p) (s : State α) (hin : (io M D p order).InputBinding vals s)
    (k : Fin p.inner) (j : Fin p.columns) :
    Whole.inputB p.request s k.val j.val=matrixB vals k j :=
  hin (1 : Fin 2) (k,j,()) trivial

theorem outside (M : Algebra α) (D : FPFused.DotAcc α)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (vals : Inputs α p) (r : RegionName) (o : Nat) :
    (io M D p order).Outside vals r o ↔ ¬Whole.Footprint p.request r o := by
  constructor
  · intro h ⟨hr,i,hi,j,hj,ho⟩
    exact (h.1 (0 : Fin 1) (⟨i,hi⟩,⟨j,hj⟩,()) trivial).elim
      (fun ne => ne hr) (fun ne => ne ho)
  · intro h
    constructor
    · intro port j _
      fin_cases port
      by_cases hr : r=p.C
      · exact Or.inr (fun he => h ⟨hr,j.1.val,j.1.isLt,j.2.1.val,j.2.1.isLt,he⟩)
      · exact Or.inl hr
    · intro k
      exact k.elim0

end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.IOFacts
