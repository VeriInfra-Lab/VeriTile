import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Bytes
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Launch

/-! The ordinary host API contract: shape indexing, fresh allocation, ordered
assertions, zero-count return, then launch and return the same allocation.
The result below is the stream-ordered observation of that returned handle.
The CUDA runtime/compiler implementations are outside this source semantics.
Launch resource rejection is an error outcome, never a claimed correct value. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.Connection
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostCorrect HostSchedule
open Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure TensorResult where
  identity : Nat
  shape : List Nat
  dtype : TileDType
  device : Nat
  dataPointer : Nat

def returned (h : Input) (s : Finite.Storage h.shape.count) (identity device : Nat) : TensorResult :=
  ⟨identity, h.shape.output, .bf16, device, s.outputBase⟩

/-- Every premise concerns initial storage, ordinary freshness and the grid.
No successful run, final value, or numerical atom is a premise. -/
theorem completion (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (identity device : Nat) (jobs : List Nat)
    (fresh : h.input≠h.output) (rep : InputRep h s heap initial)
    (order : jobs.Perm (List.range (grid h))) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      (returned h s identity device).identity=identity ∧
      (returned h s identity device).shape=h.shape.output ∧
      (returned h s identity device).dtype=.bf16 ∧
      (returned h s identity device).device=device ∧
      h.shape.output.prod=2*h.shape.count ∧
      (∀ j, j<2*h.shape.count →
        read (physical h s heap final.mem) (s.outputBase+2*j) =
          Bits.expectedBits (targetNibble s heap j)) ∧
      (∀ a, (a<s.outputBase ∨ s.outputBase+4*h.shape.count≤a) →
        physical h s heap final.mem a=heap a) ∧
      (∀ i, i<h.shape.count → physical h s heap final.mem (s.inputPointer+i)=heap (s.inputPointer+i)) ∧
      (∀ r o, (r≠h.output ∨ h.shape.count*2≤o) → final.mem r o=initial.mem r o) := by
  obtain ⟨final,run,memory,frame⟩ :=
    ScheduleCorrect.fp4_to_bf16_arbitrary_program_order_correctness rest h jobs initial
      fresh (packed_of_rep h s heap initial rep) order
  refine ⟨final,run,rfl,rfl,rfl,rfl,?_,?_,physical_frame h s heap final.mem,
    physical_input h s heap final.mem,frame⟩
  · simpa [Nat.mul_comm] using output_numel h.shape
  · intro j hj
    have hb : read (physical h s heap final.mem) (s.outputBase+2*j) =
        Bits.expectedBits (targetNibble s heap j) := by
      rw [physical_read h s heap final.mem j hj, memory]
      have bound : j<h.shape.count*2 := by omega
      simp only [expectedMemory, bound, and_self, if_true]
      simp only [cell, value]
      rw [read_embed, nibble_from_bytes h s heap initial rep j hj]
    exact hb

/-- Hardware grid representability is a launch API admission condition, not
an input-domain restriction. The wrapper may report a launch setup error if
this fails. Allocation metadata already proves all i64 bounds independently. -/
def GridAdmitted (h : Input) : Prop := grid h ≤ 2147483647

inductive Failure where
  | rank | allocation | contiguous | cuda | grid | setup
  deriving DecidableEq

/-- First host stage outcome. `allocationReady` and `setupReady` denote ordinary
API outcomes. In particular, setup failure cannot classify a wrong returned
value as success. The empty branch performs no grid/setup operation. -/
def protocol (shape : Option Shape) (allocationReady contiguous inputCuda outputCuda setupReady : Bool) :
    Except Failure Shape :=
  match shape with
  | none => .error .rank
  | some shape =>
    if !allocationReady then .error .allocation
    else if !contiguous then .error .contiguous
    else if !(inputCuda && outputCuda) then .error .cuda
    else if shape.count=0 then .ok shape
    else if 2147483647 < (shape.count+511)/512 then .error .grid
    else if !setupReady then .error .setup
    else .ok shape

theorem protocol_success (shape : Shape) (gridSafe : (shape.count+511)/512≤2147483647) :
    protocol (some shape) Bool.true Bool.true Bool.true Bool.true Bool.true = .ok shape := by
  simp [protocol, show ¬2147483647 < (shape.count+511)/512 by omega]

theorem empty_without_launch (shape : Shape) (empty : shape.count=0) (setupReady : Bool) :
    protocol (some shape) Bool.true Bool.true Bool.true Bool.true setupReady = .ok shape := by
  simp [protocol, empty]

theorem rank_first (a c x y q : Bool) : protocol none a c x y q = .error .rank := by rfl

theorem allocation_before_asserts (shape : Shape) (c x y q : Bool) :
    protocol (some shape) Bool.false c x y q = .error .allocation := by rfl

theorem contiguous_before_cuda (shape : Shape) (x y q : Bool) :
    protocol (some shape) Bool.true Bool.false x y q = .error .contiguous := by rfl

theorem protocol_grid (h : Input) (allocationReady contiguous inputCuda outputCuda setupReady : Bool)
    (result : Shape)
    (success : protocol (some h.shape) allocationReady contiguous inputCuda outputCuda setupReady = .ok result) :
    grid h≤2147483647 := by
  by_contra rejected
  have tooLarge : 2147483647 < (h.shape.count+511)/512 := by
    unfold grid at rejected
    omega
  have nonempty : h.shape.count≠0 := by
    intro zero
    simp only [zero] at tooLarge
    omega
  simp only [protocol, nonempty, tooLarge, if_false, if_true] at success
  split_ifs at success <;> cases success

structure Observation where
  tensor : TensorResult
  state : State Word
  bytes : Heap

/-- Stream-ordered observation of the handle returned by the wrapper. Generic
launch semantics serializes disjoint program stores in any complete order;
finite arithmetic and byte-store refinement are proved separately, not assumed
as a final-output law. The empty source returns immediately and jobs is []. -/
def observe (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (identity device : Nat) (jobs : List Nat)
    (allocationReady contiguous inputCuda outputCuda setupReady : Bool) :
    Option (Except Failure Observation) :=
  match protocol (some h.shape) allocationReady contiguous inputCuda outputCuda setupReady with
  | .error e => some (.error e)
  | .ok _ => (Launch.run rest h jobs initial).map fun final =>
      .ok ⟨returned h s identity device, final, physical h s heap final.mem⟩

/-- The independent complete observable contract. No source execution occurs
in its definition. BF16 payload equality retains both signed zeros. -/
def ResultSpec (h : Input) (s : Finite.Storage h.shape.count) (heap : Heap)
    (initial : State Word) (identity device : Nat) (observed : Observation) : Prop :=
  observed.tensor.identity=identity ∧ observed.tensor.shape=h.shape.output ∧
  observed.tensor.dtype=.bf16 ∧ observed.tensor.device=device ∧
  observed.tensor.dataPointer=s.outputBase ∧ h.shape.output.prod=2*h.shape.count ∧
  (∀ j, j<2*h.shape.count → read observed.bytes (s.outputBase+2*j)=Bits.expectedBits (targetNibble s heap j)) ∧
  (∀ a, (a<s.outputBase ∨ s.outputBase+4*h.shape.count≤a) → observed.bytes a=heap a) ∧
  (∀ i, i<h.shape.count → observed.bytes (s.inputPointer+i)=heap (s.inputPointer+i)) ∧
  (∀ r o, (r≠h.output ∨ h.shape.count*2≤o) → observed.state.mem r o=initial.mem r o)

theorem observe_correct (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (identity device : Nat) (jobs : List Nat)
    (fresh : h.input≠h.output) (rep : InputRep h s heap initial)
    (order : jobs.Perm (List.range (grid h)))
    (allocationReady contiguous inputCuda outputCuda setupReady : Bool) :
    ∃ outcome, observe rest h s heap initial identity device jobs
        allocationReady contiguous inputCuda outputCuda setupReady = some outcome ∧
      match outcome with
      | .error e => protocol (some h.shape) allocationReady contiguous inputCuda outputCuda setupReady = .error e
      | .ok observed => ResultSpec h s heap initial identity device observed := by
  unfold observe
  cases hp : protocol (some h.shape) allocationReady contiguous inputCuda outputCuda setupReady with
  | error e => exact ⟨.error e, rfl, rfl⟩
  | ok shape =>
    obtain ⟨final,hr,hi,hs,ht,hd,hn,hv,hf,hx,hl⟩ :=
      completion rest h s heap initial identity device jobs fresh rep order
    obtain ⟨actual,ran,memory⟩ := Launch.complete_transfer rest h s jobs initial final
      (protocol_grid h allocationReady contiguous inputCuda outputCuda setupReady shape hp) order hr
    refine ⟨.ok ⟨returned h s identity device, actual, physical h s heap actual.mem⟩, ?_, ?_⟩
    · simp only [ran,Option.map_some]
    · dsimp only [ResultSpec]
      rw [memory]
      exact ⟨hi,hs,ht,hd,rfl,hn,hv,hf,hx,hl⟩

#specNonCircular ResultSpec avoiding [observe, runSchedule, Launch.run, FP.Structural.exec]
#axiomsClean observe_correct
#axiomsClean completion
#axiomsClean protocol_success
#axiomsClean empty_without_launch
#axiomsClean allocation_before_asserts
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Connection
