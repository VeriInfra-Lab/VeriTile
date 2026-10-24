import bench.repairs.tritonbench_g.mul_exponent_compensator.PairIO
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.IOComplete
import VeriTile.Triton.Float.ScalarOpsLaws
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Commuted
open VeriTile.Triton FP.Structural Interface
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Proof interpretation used only to reuse the complete source execution.
All source operations other than multiplication keep their original meaning. -/
def flipMul {α : Type} (M : Algebra α) : Algebra α :=
  { M with binary := fun p d op a b => match op with
      | .mul => M.binary p d .mul b a
      | _ => M.binary p d op a b }

theorem execution {α : Type} [Inhabited α] (M : Algebra α)
    (src dst : RegionName) (n B : Nat) (s : State α) :
    exec M (scalarLeftKernel src dst n B) s =
      exec (flipMul M) (candidateKernel src dst n B) s := by
  simp [scalarLeftKernel,scalarLeftBody,candidateKernel,sourceBody,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numericLt,natLt,bop,store,Region.cast,ComputeDType.eraseDType,flipMul]
  rfl

theorem run_schedule {α : Type} [Inhabited α] (M : Algebra α)
    (src dst : RegionName) (n B : Nat) (order : List Nat) (s : State α) :
    runSchedule (order.map (invocation M src dst n B)) s =
      runSchedule (order.map (Interface.invocation (flipMul M) src dst n B)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    change (exec M (scalarLeftKernel src dst n B)
      ((invocation M src dst n B pid).initial s)).bind _ = _
    rw [execution]
    change (exec (flipMul M) (candidateKernel src dst n B)
      ((Interface.invocation (flipMul M) src dst n B pid).initial s)).bind _ = _
    congr 1
    funext next
    exact ih next

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat)
    (input : ∀ i, i<p.n → i<bounds p.x)
    (output : ∀ i, i<p.n → i<bounds p.output) (s : State α) :
    TraceSafe M bounds (scalarLeftKernel p.x p.output p.n p.block) s := by
  simp [TraceSafe,scalarLeftKernel,scalarLeftBody,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    State.setReg,numericLt,natLt,bop,Region.cast,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,
    input,output,numeric,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  refine ⟨fun i h=>input _ h,fun i h=>output _ h,?_⟩
  split <;> trivial

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α)
    (src dst : RegionName) (n B : Nat) (bounds : RegionName → Nat)
    (input : ∀ i, i<n → i<bounds src)
    (output : ∀ i, i<n → i<bounds dst) (order : List Nat) (s : State α) :
    ScheduleSafe bounds (order.map (invocation M src dst n B)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨source_safe M (parameters src dst n B) bounds input output _,fun t _ => ih t⟩

theorem scalar_native (profile : FP.Config) (x : KernelArithmetic.FPWord) :
    scalar (flipMul (KernelArithmetic.nativeAlgebra profile)) x =
      KernelArithmetic.native profile x := by
  simp only [scalar,flipMul,KernelArithmetic.nativeAlgebra,KernelArithmetic.native]
  simp only [KernelArithmetic.literalWord,if_pos rfl]
  exact FP.ScalarOpsLaws.mul_comm profile _ _

theorem implements (profile : FP.Config) (src dst : RegionName) (n B : Nat)
    (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) :
    (variant (KernelArithmetic.nativeAlgebra profile) src dst n B order).toProgramIO.Implements
      (formula (KernelArithmetic.native profile) src dst n) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial hin
    apply schedule_safe _ src dst n B bounds
    · intro i hi
      exact hb.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
  · intro vals initial hin
    obtain ⟨final,he,hm⟩ := ScheduledSource.complete
      (flipMul (KernelArithmetic.nativeAlgebra profile)) src dst n B positive order schedule initial
    refine ⟨final,?_,?_,?_⟩
    · change runSchedule (order.map (invocation _ src dst n B)) initial = some final
      rw [run_schedule]
      exact he
    · intro o j _
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      have input := hin (0 : Fin 1) j trivial
      change (initial.mem src j.1.val).read .real=vals (0 : Fin 1) j at input
      change mapCell id (final.mem dst j.1.val)=Cell.mk .real
        (KernelArithmetic.native profile (vals (0 : Fin 1) j))
      rw [hm]
      change mapCell id (if dst=dst ∧ j.1.val<n then
        Cell.mk .real (scalar (flipMul (KernelArithmetic.nativeAlgebra profile))
          ((initial.mem src j.1.val).read .real)) else initial.mem dst j.1.val)=_
      rw [if_pos ⟨rfl,j.1.isLt⟩]
      change Cell.mk .real (scalar (flipMul (KernelArithmetic.nativeAlgebra profile))
        ((initial.mem src j.1.val).read .real))=_
      rw [scalar_native,input]
    · intro r i outside
      have untouched : ¬(r=dst ∧ i<n) := by
        rintro ⟨rfl,hi⟩
        have h := outside.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
        exact h.elim (fun h=>h rfl) (fun h=>h rfl)
      rw [hm]
      exact if_neg untouched

#axiomsClean execution
#axiomsClean scalar_native
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Commuted
