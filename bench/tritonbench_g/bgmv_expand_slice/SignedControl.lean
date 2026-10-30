import bench.tritonbench_g.bgmv_expand_slice.SignedProgram

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Control
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem single (M : Algebra α) (st : ComputeStmt) (s t : State α) (exec : step M st s=some t) :
    run M [st] s=some t := by simp only [run,exec,bind,Option.bind]

theorem pass (M : Algebra α) (cond : Op .bool []) (body : List ComputeStmt) (s t : State α)
    (guard : FP.Structural.evalOp M none cond s=some (fun _ => Bool.true))
    (exec : run M body s=some t) : step M (.ifThen cond body) s=some t := by
  simp only [step,guard,bind,Option.bind,ite_true,exec]

theorem skip (M : Algebra α) (cond : Op .bool []) (body : List ComputeStmt) (s : State α)
    (guard : FP.Structural.evalOp M none cond s=some (fun _ => Bool.false)) :
    step M (.ifThen cond body) s=some s := by
  simp only [step,guard,bind,Option.bind,Bool.false_eq_true,ite_false]

theorem integer_guard (M : Algebra α) (s : State α) (value : Int)
    (register : s.regs .int [] "lora_index"=some (fun _ => value)) :
    FP.Structural.evalOp M none Program.guard s=some (fun _ => decide (value≠-1)) := by
  simp [Program.guard,evalOp_unfold,register,numericEq,bind,Option.bind]
  rfl


theorem loop_one (M : Algebra α) (body : List ComputeStmt) (s t : State α)
    (length : s.regs .int [] "split_n_length"=some (fun _ => 1))
    (exec : run M body (s.setReg "n" .int [] (fun _ => 0))=some t) :
    step M (.forRangeInt "n" (.constInt 0) (.ref .int [] "split_n_length") (.constNat 256) body) s=some t := by
  simp only [step,evalOp_unfold,length,bind,Option.bind]
  rw [FP.Structural.rangeInt]
  simp only [show (256:Nat)≠0 from by decide,if_false,show (0:Int)<1 from by decide,if_true,exec,bind,Option.bind]
  rw [FP.Structural.rangeInt]
  simp

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Control
