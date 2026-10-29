import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Contract

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
set_option maxHeartbeats 800000

theorem forwardCheck_iff (x : TensorArgument) (y : Option TensorArgument)
    (block column : IntegerArgument) :
    forwardCheck x y block column=.ok () ↔ AdmittedForward x y block column := by
  cases y with
  | none =>
    cases x <;> cases block <;> cases column <;>
      simp only [forwardCheck,AdmittedForward,ForwardContract]
    all_goals try split_ifs
    all_goals simp_all <;> omega
  | some y =>
    cases y <;> cases x <;> cases block <;> cases column <;>
      simp only [forwardCheck,AdmittedForward,ForwardContract]
    all_goals try split_ifs
    all_goals simp_all <;> omega

theorem backwardCheck_iff (x : TensorView) (u : TensorArgument) (width : Nat) :
    backwardCheck x u width=.ok () ↔ AdmittedBackward x u width := by
  cases u <;> simp [backwardCheck,AdmittedBackward]

theorem rejected_forward_iff (x : TensorArgument) (y : Option TensorArgument)
    (block column : IntegerArgument) :
    (∃ e,forwardCheck x y block column=.error e) ↔ ¬AdmittedForward x y block column := by
  rw [←forwardCheck_iff]
  cases forwardCheck x y block column <;> simp

theorem rejected_backward_iff (x : TensorView) (u : TensorArgument) (width : Nat) :
    (∃ e,backwardCheck x u width=.error e) ↔ ¬AdmittedBackward x u width := by
  rw [←backwardCheck_iff]
  cases backwardCheck x u width <;> simp

#axiomsClean forwardCheck_iff
#axiomsClean backwardCheck_iff
#axiomsClean rejected_forward_iff
#axiomsClean rejected_backward_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
