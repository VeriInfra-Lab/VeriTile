import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked
open VeriTile.Bench.Repairs.Ksoftmax

/-- Exactly the original admission domain plus the missing live grid-y cap. -/
def ForwardContract (y x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) : Prop :=
  Ksoftmax.ForwardContract y x mask mode ∧ (0<x.numel → rows x≤65535)

def BackwardContract (dst grad saved : TensorMetadata) : Prop :=
  Ksoftmax.BackwardContract dst grad saved ∧ (0<saved.numel → rows saved≤65535)

inductive InputError where | tensor | flags | metadata | shape | device | mode | mask | extent | gridY | memory
  deriving DecidableEq, Repr

/-- `flags` records exact Python bool classification for LOG and CAUSAL. -/
def checkForward (y x mask : ArgumentMetadata) (mode : Option String) (flags : Bool) : Except InputError Unit :=
  if !flags then .error .flags else
  if ¬KnownMode mode then .error .mode else
  match y,x with
  | .tensor y,.tensor x =>
    if ¬(TensorContract y ∧ TensorContract x) then .error .metadata
    else if ¬(x.shape.length=3 ∧ y.shape=x.shape) then .error .shape
    else if y.device≠x.device then .error .device
    else if ¬MaskContract x mask mode then .error .mask
    else if x.numel=0 then .ok ()
    else if ¬ExtentContract x then .error .extent
    else if rows x > 65535 then .error .gridY
    else if ¬(y.dataPtr.toNat%4=0 ∧ PairContract y x ∧ MaskMemory y mask mode) then .error .memory
    else .ok ()
  | _,_ => .error .tensor

def checkBackward (dst grad saved : ArgumentMetadata) (flags : Bool) : Except InputError Unit :=
  if !flags then .error .flags else match dst,grad,saved with
  | .tensor dst,.tensor grad,.tensor saved =>
    if ¬(TensorContract dst ∧ TensorContract grad ∧ TensorContract saved) then .error .metadata
    else if ¬(saved.shape.length=3 ∧ dst.shape=saved.shape ∧ grad.shape=saved.shape) then .error .shape
    else if ¬(dst.device=saved.device ∧ grad.device=saved.device) then .error .device
    else if saved.numel=0 then .ok ()
    else if ¬ExtentContract saved then .error .extent
    else if rows saved > 65535 then .error .gridY
    else if ¬(dst.dataPtr.toNat%4=0 ∧ PairContract dst grad ∧ PairContract dst saved) then .error .memory
    else .ok ()
  | _,_,_ => .error .tensor


theorem checkForward_iff (y x : TensorMetadata) (m : ArgumentMetadata) (mode : Option String) (flags : Bool) :
    checkForward y x m mode flags = .ok () ↔ flags=Bool.true ∧ ForwardContract y x m mode := by
  by_cases h : x.numel=0
  · cases flags <;> simp [checkForward,ForwardContract,Ksoftmax.ForwardContract,h,ite_eq_iff,and_assoc,and_comm,and_left_comm]
  · have hn : 0<x.numel := Nat.pos_of_ne_zero h
    cases flags <;> simp [checkForward,ForwardContract,Ksoftmax.ForwardContract,h,hn,ite_eq_iff,and_assoc,and_comm,and_left_comm]

theorem checkBackward_iff (d g o : TensorMetadata) (flags : Bool) :
    checkBackward d g o flags = .ok () ↔ flags=Bool.true ∧ BackwardContract d g o := by
  by_cases h : o.numel=0
  · cases flags <;> simp [checkBackward,BackwardContract,Ksoftmax.BackwardContract,h,ite_eq_iff,and_assoc,and_comm,and_left_comm]
  · have hn : 0<o.numel := Nat.pos_of_ne_zero h
    cases flags <;> simp [checkBackward,BackwardContract,Ksoftmax.BackwardContract,h,hn,ite_eq_iff,and_assoc,and_comm,and_left_comm]

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


/-- Architectural axes of the actual `(B,Q)` launch; empty calls do not launch. -/
def SupportedLaunch (x : TensorMetadata) : Prop :=
  0<x.numel → 0<batches x ∧ batches x≤2^31-1 ∧ 0<rows x ∧ rows x≤65535

theorem axes (x : TensorMetadata) (rank : x.shape.length=3)
    (extent : 0<x.numel → ExtentContract x) (query : 0<x.numel → rows x≤65535) :
    SupportedLaunch x := by
  intro live
  have product : 0<batches x*rows x*columns x := by
    rw [←count_three x rank]
    exact live
  have pair := Nat.pos_of_mul_pos_right product
  exact ⟨Nat.pos_of_mul_pos_right pair,by have := (extent live).1; omega,
    Nat.pos_of_mul_pos_left pair,query live⟩

theorem forward_launch {y x : TensorMetadata} {mask : ArgumentMetadata} {mode : Option String}
    (admitted : ForwardContract y x mask mode) : SupportedLaunch x := by
  exact axes x admitted.1.2.2.1 (fun live => (admitted.1.2.2.2.2.2.2.2 live).1) admitted.2

theorem backward_launch {dst grad saved : TensorMetadata}
    (admitted : BackwardContract dst grad saved) : SupportedLaunch saved := by
  exact axes saved admitted.1.2.2.2.1 (fun live => (admitted.1.2.2.2.2.2.2.2.2 live).1) admitted.2

theorem forward_domain_exact (y x : TensorMetadata) (mask : ArgumentMetadata) (mode : Option String) :
    ForwardContract y x mask mode ↔ Ksoftmax.ForwardContract y x mask mode ∧
      (x.numel=0 ∨ rows x≤65535) := by
  unfold ForwardContract
  by_cases empty : x.numel=0
  · simp [empty]
  · have live : 0<x.numel := Nat.pos_of_ne_zero empty
    simp [empty,live]

theorem backward_domain_exact (dst grad saved : TensorMetadata) :
    BackwardContract dst grad saved ↔ Ksoftmax.BackwardContract dst grad saved ∧
      (saved.numel=0 ∨ rows saved≤65535) := by
  unfold BackwardContract
  by_cases empty : saved.numel=0
  · simp [empty]
  · have live : 0<saved.numel := Nat.pos_of_ne_zero empty
    simp [empty,live]

theorem forward_empty_unchanged (y x : TensorMetadata) (mask : ArgumentMetadata)
    (mode : Option String) (flags : Bool) (empty : x.numel=0) :
    checkForward y x mask mode flags=.ok () ↔ Ksoftmax.checkForward y x mask mode flags=.ok () := by
  rw [checkForward_iff,Ksoftmax.checkForward_iff,forward_domain_exact]
  simp [empty]

theorem backward_empty_unchanged (dst grad saved : TensorMetadata) (flags : Bool) (empty : saved.numel=0) :
    checkBackward dst grad saved flags=.ok () ↔ Ksoftmax.checkBackward dst grad saved flags=.ok () := by
  rw [checkBackward_iff,Ksoftmax.checkBackward_iff,backward_domain_exact]
  simp [empty]

#axiomsClean checkForwardArguments_iff
#axiomsClean checkBackwardArguments_iff
#axiomsClean forward_domain_exact
#axiomsClean backward_domain_exact
#axiomsClean forward_empty_unchanged
#axiomsClean backward_empty_unchanged
#axiomsClean forward_launch
#axiomsClean backward_launch
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked
