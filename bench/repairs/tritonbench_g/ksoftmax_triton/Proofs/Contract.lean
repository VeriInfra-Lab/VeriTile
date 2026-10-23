import bench.repairs.tritonbench_g.ksoftmax_triton.Contract
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Native
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.Ksoftmax

theorem checkForward_iff (y x : TensorMetadata) (m : ArgumentMetadata) (mode : Option String) (flags : Bool) :
    checkForward y x m mode flags = .ok () ↔ flags=Bool.true ∧ ForwardContract y x m mode := by
  by_cases h : x.numel=0
  · cases flags <;> simp [checkForward,ForwardContract,h,ite_eq_iff,and_assoc,and_comm,and_left_comm]
  · have hn : 0<x.numel := Nat.pos_of_ne_zero h
    cases flags <;> simp [checkForward,ForwardContract,h,hn,ite_eq_iff,and_assoc,and_comm,and_left_comm]

theorem checkBackward_iff (d g o : TensorMetadata) (flags : Bool) :
    checkBackward d g o flags = .ok () ↔ flags=Bool.true ∧ BackwardContract d g o := by
  by_cases h : o.numel=0
  · cases flags <;> simp [checkBackward,BackwardContract,h,ite_eq_iff,and_assoc,and_comm,and_left_comm]
  · have hn : 0<o.numel := Nat.pos_of_ne_zero h
    cases flags <;> simp [checkBackward,BackwardContract,h,hn,ite_eq_iff,and_assoc,and_comm,and_left_comm]

def ForwardArguments (y x mask : ArgumentMetadata) (mode : Option String) (flags : Bool) : Prop :=
  flags=Bool.true ∧ match y,x with
  | .tensor y,.tensor x => ForwardContract y x mask mode
  | _,_ => False

def BackwardArguments (d g o : ArgumentMetadata) (flags : Bool) : Prop :=
  flags=Bool.true ∧ match d,g,o with
  | .tensor d,.tensor g,.tensor o => BackwardContract d g o
  | _,_,_ => False

theorem checkForwardArguments_iff (y x mask : ArgumentMetadata) (mode : Option String) (flags : Bool) :
    checkForward y x mask mode flags=.ok () ↔ ForwardArguments y x mask mode flags := by
  cases y <;> cases x
  all_goals first
    | exact checkForward_iff _ _ _ _ _
    | cases flags <;> simp [checkForward,ForwardArguments,ite_eq_iff]

theorem checkBackwardArguments_iff (d g o : ArgumentMetadata) (flags : Bool) :
    checkBackward d g o flags=.ok () ↔ BackwardArguments d g o flags := by
  cases d <;> cases g <;> cases o
  all_goals first
    | exact checkBackward_iff _ _ _ _
    | cases flags <;> simp [checkBackward,BackwardArguments]

theorem shape_three (x : TensorMetadata) (h : x.shape.length=3) :
    x.shape=[batches x,rows x,columns x] := by
  rcases x with ⟨a,b,c,s,d,e,f⟩
  cases s with
  | nil => simp at h
  | cons v s => cases s with
    | nil => simp at h
    | cons w s => cases s with
      | nil => simp at h
      | cons z s =>
        have hs : s=[] := by simpa using h
        subst s
        rfl

theorem count_three (x : TensorMetadata) (h : x.shape.length=3) :
    x.numel=batches x*rows x*columns x := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [shape_three x h]
  simp [Nat.mul_assoc]

#axiomsClean checkForwardArguments_iff
#axiomsClean checkBackwardArguments_iff
#axiomsClean checkForward_iff
#axiomsClean checkBackward_iff
#axiomsClean shape_three
#axiomsClean count_three
end VeriTile.Bench.Repairs.Ksoftmax
