import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SafetySteps

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
open VeriTile Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem perTile_commands (c : Parameters) (candidate : Bool) : perTile c candidate =
    [command c candidate 0,command c candidate 1,command c candidate 2,command c candidate 3,
     command c candidate 4,command c candidate 5,command c candidate 6] := by cases candidate <;> rfl

theorem tile_safe (M : Algebra α) (c : Parameters) (candidate : Bool) (s : State α) (T : Nat)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) (bounds : RegionName → Nat)
    (hi : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.input.base+((T*c.B+j.val:Nat):Int)*c.input.stride).toNat<bounds c.input.region)
    (ho : ∀ j : Fin c.B,T*c.B+j.val<c.n →
      (c.output.base+((T*c.B+j.val:Nat):Int)*c.output.stride).toNat<bounds c.output.region) :
    traceSafeList M bounds (perTile c candidate) s := by
  rw [perTile_commands]
  refine traceSafeList_cons_of_step ?_ (step0 M c candidate s T ht)
    (traceSafeList_cons_of_step ?_ (step1 M c candidate s T ht)
      (traceSafeList_cons_of_step ?_ (step2 M c candidate s T ht)
        (traceSafeList_cons_of_step ?_ (step3 M c candidate s T ht)
          (traceSafeList_cons_of_step ?_ (step4 M c candidate s T ht)
            (traceSafeList_cons_of_step ?_ (step5 M c candidate s T ht)
              (traceSafeList_cons_intro ?_ (by intro t _;exact traceSafeList_nil _ _ _)))))))
  all_goals cases candidate
  all_goals simp only [command,perTile,oneTileKernel,oneTileCommuted,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
    safeMask_unfold,addressSafe_unfold,maskActive_unfold,precisionContext,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    b1,b2,b3,b4,b5,b6,State.setReg,ht,numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals first | exact fun j active => hi j.1 active | exact fun j active => ho j.1 active
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourceSafety
