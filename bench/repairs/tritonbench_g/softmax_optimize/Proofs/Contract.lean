import bench.repairs.tritonbench_g.softmax_optimize.Contract
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Config
namespace VeriTile.Bench.Repairs.SoftmaxOptimize

theorem check_iff (x : TensorMetadata) : check x=.ok () ↔ InputContract x := by
  by_cases empty : x.numel=0
  · simp [check,InputContract,empty,ite_eq_iff,and_assoc]
  · have pos : 0<x.numel := Nat.pos_of_ne_zero empty
    simp [check,InputContract,empty,pos,ite_eq_iff,and_assoc]

theorem dynamic_iff (arg : ArgumentMetadata) :
    check arg=.ok () ↔ ∃ x, arg=.tensor x ∧ InputContract x := by
  cases arg with
  | nonTensor => simp [check]
  | tensor x => simpa using check_iff x

theorem count_two (x : TensorMetadata) (rank : x.shape.length=2) : x.numel=rows x*columns x := by
  rcases x with ⟨dt,la,de,shape,co,ne,ptr⟩
  rcases shape with _|⟨m,_|⟨n,rest⟩⟩ <;> simp_all [rows,columns,VeriTile.Torch.TensorMetadata.numel]

theorem dims_positive (x : TensorMetadata) (h : InputContract x) (live : x.numel≠0) :
    0<rows x ∧ 0<columns x ∧ rows x<2^31 ∧ columns x<2^31 := by
  have count := count_two x h.2.2.2.1
  have size := h.2.2.2.2.2.2.1
  have hm : 0<rows x := by
    by_contra bad
    have hz : rows x=0 := by omega
    simp [hz] at count
    omega
  have hn : 0<columns x := by
    by_contra bad
    have hz : columns x=0 := by omega
    simp [hz] at count
    omega
  refine ⟨hm,hn,?_,?_⟩ <;> nlinarith

#axiomsClean check_iff
#axiomsClean dynamic_iff
#axiomsClean count_two
#axiomsClean dims_positive

theorem allocated_contract (x y : TensorMetadata) (h : InputContract x) (a : Allocated x y) :
    InputContract y := by
  have size : y.numel=x.numel := congrArg List.prod a.1
  exact ⟨a.2.2.2.1,a.2.2.1,by rw [a.2.1];exact h.2.2.1,
    by rw [a.1];exact h.2.2.2.1,a.2.2.2.2.1,a.2.2.2.2.2.1,
    by rw [size];exact h.2.2.2.2.2.2.1,fun live => (a.2.2.2.2.2.2.1 live).1⟩

#axiomsClean allocated_contract
end VeriTile.Bench.Repairs.SoftmaxOptimize
