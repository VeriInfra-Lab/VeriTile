import bench.tritonbench_g.matmul_leakyrelu.Proofs.OriginalReal
import bench.tritonbench_g.matmul_leakyrelu.Proofs.IOSafety
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Pair
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- One local typed FP32 rewrite, before activation or storage. The complete
32-lane operands and incoming accumulator remain arbitrary, including padding. -/
def DotAddContract (M : Algebra α) (D : FPValue.Dot α) (F : FPFused.DotAcc α) : Prop :=
  ∀ (a b acc : TileIndex [32,32] → α) (i : TileIndex [32,32]),
    F.apply (some .fp32) .fp32 .fp32 .real {} (batch := []) a b acc i =
      M.binary (some .fp32) .real .add (acc i)
        (M.cast (some .fp32) .fp32 .real
          (D.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch := []) a b i))

def contractedNumerics (Q : Whole.Numerics α) : Whole.Contracted.Numerics α where
  zero := Q.zero
  dotAcc := fun a b acc => Q.add acc (Q.dot a b)
  nonnegative := Q.nonnegative
  rightSlope := Q.slopeMul
  castHalf := Q.castHalf

theorem contracted_laws (M : Algebra α) (D : FPValue.Dot α) (F : FPFused.DotAcc α)
    (Q : Whole.Numerics α) (ports : Whole.PrimitiveLaws M D Q)
    (contract : DotAddContract M D F)
    (comparison : F.le = D.le)
    (commute : ∀ x y, M.binary (some .fp32) .real .mul x y = M.binary (some .fp32) .real .mul y x) :
    Whole.Contracted.PrimitiveLaws M F (contractedNumerics Q) := by
  refine ⟨ports.initialZero,ports.loadZero,?_,?_,?_,ports.halfCast⟩
  · intro a b acc i
    rw [contract,ports.add,ports.dot]
    rfl
  · intro x
    rw [comparison]
    exact ports.comparison x
  · intro x
    rw [commute]
    exact ports.slope x

theorem reference_equal (Q : Whole.Numerics α) (r : Whole.Request)
    (initial : State α) (i j : Nat) :
    Whole.reference Q r initial i j = Whole.Contracted.reference (contractedNumerics Q) r initial i j := rfl

/-- Local typed dot contraction and strict slope commutation lifted to the
complete actual source schedules: safe success, every output cell and both frames. -/
theorem equivalent (M : Algebra α) (D : FPValue.Dot α) (F : FPFused.DotAcc α)
    (Q : Whole.Numerics α) (ports : Whole.PrimitiveLaws M D Q)
    (contract : DotAddContract M D F) (comparison : F.le = D.le)
    (commute : ∀ x y, M.binary (some .fp32) .real .mul x y = M.binary (some .fp32) .real .mul y x)
    (p : KernelInterface.Parameters) (order : Schedule p.request)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) :
    (Original.io M D p order).Equivalent (variant M D F p order) := by
  intro vals _ bounds bounded initial _
  obtain ⟨left,lrun,lvalues,lframe⟩ := OriginalReal.scheduled_grid M D Q ports
    p.request order initial separateA separateB
  obtain ⟨right,rrun,rvalues,rframe⟩ := scheduled_grid M F (contractedNumerics Q)
    (contracted_laws M D F Q ports contract comparison commute)
    p.request order initial separateA separateB
  refine ⟨OriginalIOSafety.whole M D p order vals bounds bounded initial,
    IOSafety.whole M F p order vals bounds bounded initial,
    left,right,(OriginalFacts.schedule_run M D p order.order initial).trans lrun,
    (IOFacts.schedule_run M F p order.order initial).trans rrun,?_,?_,?_⟩
  · intro port j _
    fin_cases port
    change mapCell id (left.mem p.C (Whole.outputOffset p.request j.1.val j.2.1.val)) =
      mapCell id (right.mem p.C (Whole.outputOffset p.request j.1.val j.2.1.val))
    erw [lvalues _ j.1.isLt _ j.2.1.isLt,rvalues _ j.1.isLt _ j.2.1.isLt,reference_equal]
  · intro r o outside
    exact lframe r o ((OriginalFacts.outside M D p order vals r o).mp outside)
  · intro r o outside
    exact rframe r o ((OriginalFacts.outside M D p order vals r o).mp outside)

abbrev Word := FP.Value .fp32

def fp32Algebra (cfg : FP.Config) (rest : Algebra Word) : Algebra Word :=
  {rest with binary := fun precision dtype op x y =>
    if precision = some .fp32 ∧ dtype = .real ∧ op = .mul then FP.mul cfg x y
    else rest.binary precision dtype op x y}

theorem fp32_mul_commutes (cfg : FP.Config) (rest : Algebra Word) :
    ∀ x y, (fp32Algebra cfg rest).binary (some .fp32) .real .mul x y =
      (fp32Algebra cfg rest).binary (some .fp32) .real .mul y x := by
  intro x y
  simpa [fp32Algebra] using FP.ScalarOpsLaws.mul_comm cfg x y

#axiomsClean equivalent
#axiomsClean fp32_mul_commutes
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.Pair
