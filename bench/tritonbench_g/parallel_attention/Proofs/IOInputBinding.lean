import bench.tritonbench_g.parallel_attention.Proofs.IOSchedule
import bench.tritonbench_g.parallel_attention.Proofs.IOInputCongruence

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Equalities only at declared finite input coordinates. -/
structure InputsAgree (direction : Direction) (shape : Shape)
    (a b : IOLogical.InputValues α) : Prop where
  query : ∀ off,off<heads shape*shape.T*shape.K → a "q" off=b "q" off
  key : ∀ off,off<heads shape*shape.T*shape.K → a "k" off=b "k" off
  value : ∀ off,off<heads shape*shape.T*shape.V → a "v" off=b "v" off
  outGrad : direction=.backward → ∀ off,off<heads shape*shape.T*shape.V → a "do" off=b "do" off
  normGrad : direction=.backward → ∀ off,off<heads shape*shape.T → a "dz" off=b "dz" off

theorem inputs_agree (direction : Direction) (shape : Shape) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (s : State α)
    (h : (dataIO α direction shape jobs).InputBinding xs s) :
    InputsAgree direction shape (fun r off=>(s.mem r off).read .real) (inputValues shape xs) := by
  constructor
  · intro off ho
    have hi:=h (⟨0,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
    simpa [dataIO,memory,inputPorts,inputValues,ho,mapValue] using hi
  · intro off ho
    have hi:=h (⟨1,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
    simpa [dataIO,memory,inputPorts,inputValues,ho,mapValue] using hi
  · intro off ho
    have hi:=h (⟨2,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
    simpa [dataIO,memory,inputPorts,inputValues,ho,mapValue] using hi
  · intro hd off ho
    have hi:=h (⟨3,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inr hd)
    simpa [dataIO,memory,inputPorts,inputValues,ho,mapValue] using hi
  · intro hd off ho
    have hi:=h (⟨4,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inr hd)
    simpa [dataIO,memory,inputPorts,inputValues,ho,mapValue] using hi

private theorem rectangle_dense {a b : IOLogical.InputValues α} {R : RegionName}
    (M T D head : Nat) (hh : head<M)
    (h : ∀ off,off<M*T*D → a R off=b R off) :
    IOLogical.RectangleEq a b R (head*(T*D)) T D D 1 := by
  intro row col hr hc
  apply h
  simpa only [dense,Nat.add_mul,Nat.mul_assoc,Nat.mul_one] using dense_bound T D M head row col hh hr hc

private theorem rectangle_transpose {a b : IOLogical.InputValues α} {R : RegionName}
    (M T D head : Nat) (hh : head<M)
    (h : ∀ off,off<M*T*D → a R off=b R off) :
    IOLogical.RectangleEq a b R (head*(T*D)) D T 1 D := by
  intro row col hr hc
  apply h
  have hbound:=dense_bound T D M head col row hh hc hr
  simpa only [dense,Nat.add_mul,Nat.mul_assoc,Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hbound

theorem forward_inputs (direction : Direction) (shape : Shape) (scale : ℝ)
    (p : Program shape 128) (a b : IOLogical.InputValues α) (h : InputsAgree direction shape a b) :
    IOLogical.ForwardEq (forwardConfig shape scale) (pid p) a b := by
  constructor
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt h.query
  · exact rectangle_transpose (heads shape) shape.T shape.K p.head.val p.head.isLt h.key
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt h.value

theorem backward_inputs (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (a b : IOLogical.InputValues α) (h : InputsAgree .backward shape a b) :
    IOLogical.BackwardEq (backwardConfig shape scale p.tile) a b := by
  constructor
  · exact rectangle_transpose (heads shape) shape.T shape.K p.head.val p.head.isLt h.query
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt h.key
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt h.value
  · exact rectangle_transpose (heads shape) shape.T shape.V p.head.val p.head.isLt (h.outGrad rfl)
  · intro row hr
    apply h.normGrad rfl
    change p.head.val*shape.T+row<heads shape*shape.T
    have hh:=p.head.isLt
    change p.head.val<heads shape at hh
    change row<shape.T at hr
    nlinarith

theorem query_inputs (shape : Shape) (scale : ℝ)
    (p : Program shape 64) (a b : IOLogical.InputValues α) (h : InputsAgree .backward shape a b) :
    IOLogical.QueryEq (dqConfig shape scale p.tile) a b := by
  constructor
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt h.query
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt h.key
  · exact rectangle_transpose (heads shape) shape.T shape.V p.head.val p.head.isLt h.value
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt (h.outGrad rfl)
  · intro row hr
    apply h.normGrad rfl
    change p.head.val*shape.T+row<heads shape*shape.T
    have hh:=p.head.isLt
    change p.head.val<heads shape at hh
    change row<shape.T at hr
    nlinarith

#axiomsClean inputs_agree
#axiomsClean forward_inputs
#axiomsClean backward_inputs
#axiomsClean query_inputs
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
