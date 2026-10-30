import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Footprint
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev Backing := SwigluTriton.Memory.Allocation

def wordByte (word : Word) (lane : Nat) : Byte := BitVec.ofNat 8 (word.toNat/256^lane)

def Covers (backing : Nat → Backing) (point : Address) (owner address : Nat) : Prop :=
  owner = (backing point.1).owner ∧ (backing point.1).baseByte+2*point.2 ≤ address ∧
    address < (backing point.1).baseByte+2*point.2+2

/-- The raw observer is independent of typed storage words. Its only value
connection is one scalar codec law; its frame law covers one event only. -/
structure ByteABI (abi : LocalABI Word σ ops) (backing : Nat → Backing) (unit : Nat → Nat) where
  observe : σ → Nat → Nat → Byte
  codec : ∀ state point, unit point.1*(point.2+1) ≤ (backing point.1).bytes → ∀ lane, lane < unit point.1 →
    observe state (backing point.1).owner ((backing point.1).baseByte+unit point.1*point.2+lane) =
      wordByte (abi.observe state point) lane
  frame : ∀ (β : Type) (event : Event Word β) state owner address,
    (match event with | .storeHalf point _ => ¬Covers backing point owner address | _ => True) →
      observe (abi.step event state).2 owner address = observe state owner address

/-- Actual Tensor/storage observations, including nonzero input offsets,
broadcast strides and read-only aliasing. Caller A/B are FP32; C is half. -/
structure TensorBinding (backing : Nat → Backing) (unit : Nat → Nat) (tensor : Tensor) : Prop where
  owner : (backing tensor.storage).owner = tensor.device
  base : (backing tensor.storage).baseByte = tensor.allocationBase.toNat
  capacity : (backing tensor.storage).bytes = tensor.allocationBytes
  width : unit tensor.storage = tensor.dtype.bytes

/-- Ordinary fresh allocation interval and capacity, with no numeric result. -/
structure OutputBacking (backing : Nat → Backing) (unit : Nat → Nat)
    (live : Nat → Prop) (output : Tensor) : Prop where
  tensor : TensorBinding backing unit output
  dtype : output.dtype = .fp16
  offset : output.offset = 0
  rows : output.rowStride = max output.columns 1
  columns : output.columnStride = 1
  bounded : output.Bounded
  fresh : ∀ old, live old → (backing old).separate (backing output.storage)

theorem output_capacity (layout : OutputBacking backing unit live output) :
    2*(output.rows*output.columns) ≤ (backing output.storage).bytes := by
  rw [layout.tensor.capacity]
  by_cases empty : output.rows = 0 ∨ output.columns = 0
  · rcases empty with h | h <;> simp [h]
  · have span := layout.bounded.span
    simp only [Tensor.span,empty,if_false,layout.dtype,Storage.bytes,layout.offset,layout.rows,layout.columns,
      Nat.mul_one,Nat.zero_add,max_eq_left (show 1 ≤ output.columns by omega)] at span
    have rows : output.rows-1+1 = output.rows := by omega
    have columns : output.columns-1+1 = output.columns := by omega
    have product := congrArg (fun r => r*output.columns) rows
    simp only [Nat.add_mul,Nat.one_mul] at product
    omega

theorem output_outside (layout : OutputBacking backing unit live output) (old owner address : Nat)
    (present : live old) (covered : (backing old).covers owner address) :
    ∀ point, Allowed output point → ¬Covers backing point owner address := by
  intro point allowed hit
  have capacity := output_capacity layout
  have sep := layout.fresh old present
  rcases point with ⟨storage,index⟩
  obtain ⟨rfl,_,indexBound⟩ := allowed
  simp only [layout.offset,Nat.sub_zero] at indexBound
  simp only [Covers] at hit
  simp only [SwigluTriton.Memory.Allocation.covers] at covered
  rcases sep with device | left | right
  · exact device (covered.1.symm.trans hit.1)
  · omega
  · omega

theorem native_frame (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (code : Code Word β) (safe : Code.Safe allowed code) (state : σ) (owner address : Nat)
    (outside : ∀ point, allowed point → ¬Covers backing point owner address) :
    raw.observe (code.native abi state).2 owner address = raw.observe state owner address := by
  induction code generalizing state with
  | pure x => rfl
  | event e rest ih =>
    simp only [Code.native]
    rw [ih _ (safe.2 _) _]
    apply raw.frame
    cases e <;> try trivial
    exact outside _ safe.1

def runPrefix (abi : LocalABI α σ ops) : Nat → Code α β → σ → σ
  | 0,_,state => state
  | _+1,.pure _,state => state
  | fuel+1,.event e rest,state =>
    let step := abi.step e state
    runPrefix abi fuel (rest step.1) step.2

theorem prefix_frame (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (code : Code Word β) (safe : Code.Safe allowed code) (state : σ) (fuel owner address : Nat)
    (outside : ∀ point, allowed point → ¬Covers backing point owner address) :
    raw.observe (runPrefix abi fuel code state) owner address = raw.observe state owner address := by
  induction fuel generalizing code state with
  | zero => rfl
  | succ fuel ih =>
    cases code with
    | pure x => rfl
    | event e rest =>
      simp only [runPrefix]
      rw [ih _ (safe.2 _) _]
      apply raw.frame
      cases e <;> try trivial
      exact outside _ safe.1

#axiomsClean output_capacity
#axiomsClean native_frame
#axiomsClean prefix_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
