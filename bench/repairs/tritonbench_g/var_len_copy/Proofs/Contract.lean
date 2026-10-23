import bench.repairs.tritonbench_g.var_len_copy.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section
open Classical
set_option maxHeartbeats 3000000

theorem checkInputs_iff (m : Inputs) (s : Nat → Segment) :
    checkInputs m s = .ok () ↔ InputContract m s := by
  have fields : InputContract m s ↔
      (∀ r, (m r).layout = .strided) ∧ (∀ r, (m r).shape.length = 1) ∧ MetadataTypes m ∧
      (∀ r, (m r).device.kind = "cuda") ∧ (∀ r, (m r).device = (m .source).device) ∧
      (∀ r, (m r).isContiguous = Bool.true) ∧ (∀ r, (m r).isNeg = Bool.false) ∧
      ((m .oldStarts).shape = (m .lengths).shape ∧ (m .oldStarts).shape = (m .newStarts).shape) ∧
      (m .oldStarts).numel < 2^31 ∧ (∀ r, 0 < (m r).numel → base m r % 4 = 0) ∧
      DisjointOutput m ∧
      (∀ p, p < (m .oldStarts).numel → 0 < (s p).len →
        0 ≤ (s p).oldStart.toInt ∧ (s p).old+(s p).len ≤ (m .source).numel) ∧
      (∀ p, p < (m .oldStarts).numel → 0 < (s p).len →
        0 ≤ (s p).newStart.toInt ∧ (s p).dest+(s p).len ≤ (m .output).numel) ∧
      SegmentsDisjoint (request m s) := by
    constructor
    · intro h
      exact ⟨h.layout,h.rank,h.dtype,h.cuda,h.device,h.contiguous,h.noNeg,h.count,
        h.extent,h.alignment,h.separation,h.sourceBound,h.outputBound,h.distinct⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o⟩
      exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,n,o⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (s : Nat → Segment) :
    checkArguments m s = .ok () ↔ AdmittedArguments m s := by
  cases m <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

theorem contract_valid (m : Inputs) (s : Nat → Segment) (h : InputContract m s) :
    Valid (request m s) := by
  refine ⟨h.sourceBound,h.outputBound,?_⟩
  intro p q i j hp hq hi hj same
  by_cases equal : p=q
  · subst q
    exact ⟨rfl,by omega⟩
  · have disjoint := h.distinct p q hp hq equal (by omega) (by omega)
    simp only [request] at disjoint same hi hj
    rcases disjoint with left | right <;> omega

/-- Ordinary actual-pointer/backing bounds and readback of the three int32
metadata arrays. No payload output or kernel execution property occurs here. -/
structure TensorBinding (m : Inputs) (s : Nat → Segment) (before : Memory) : Prop where
  capacity : ∀ role, base m role + 4*(m role).numel ≤ 2^64
  oldCells : ∀ p, p < (m .oldStarts).numel → before (addr (base m .oldStarts) p) = (s p).oldStart
  lengthCells : ∀ p, p < (m .oldStarts).numel → before (addr (base m .lengths) p) = (s p).length
  newCells : ∀ p, p < (m .oldStarts).numel → before (addr (base m .newStarts) p) = (s p).newStart

theorem prepared_direct (m : Inputs) (s : Nat → Segment) (h : InputContract m s)
    (before : Memory) (binding : TensorBinding m s before) :
    Prepared (directLayout m) (request m s) before := by
  have lengths : (m .oldStarts).numel = (m .lengths).numel := congrArg List.prod h.count.1
  have starts : (m .oldStarts).numel = (m .newStarts).numel := congrArg List.prod h.count.2
  refine ⟨rfl,rfl,rfl,h.extent,binding.capacity .oldStarts,?_,?_,
    binding.capacity .source,binding.capacity .output,binding.oldCells,binding.lengthCells,
    binding.newCells,?_⟩
  · simpa only [directLayout,lengths] using binding.capacity .lengths
  · simpa only [directLayout,starts] using binding.capacity .newStarts
  · intro i hi
    have separate (role : Role) (other : role ≠ .output) (j : Nat) (hj : j < (m role).numel) :
        addr (base m .output) i ≠ addr (base m role) j := by
      have disjoint := h.separation role other
      change i < (m .output).numel at hi
      unfold addr
      rcases disjoint with left | right <;> omega
    refine ⟨fun j hj => separate .source (by decide) j hj,?_⟩
    intro p hp
    exact ⟨separate .oldStarts (by decide) p hp,
      separate .lengths (by decide) p (by change p < (m .oldStarts).numel at hp; omega),
      separate .newStarts (by decide) p (by change p < (m .oldStarts).numel at hp; omega)⟩

/-- Interpretation of the caller's contiguous physical buffers in logical
kernel regions. Metadata is decoded as signed int32; arbitrary Real payloads
are read at the same actual pointer plus four-byte element offset. This is an
initial-memory interpretation, not wrapper execution or an output assumption. -/
def logicalState (m : Inputs) (metadata : Memory) (payload : Nat → ℝ)
    (scratch : VeriTile.Triton.BlockState) : VeriTile.Triton.BlockState :=
  { scratch with mem := fun r i =>
      if r = "old" then VeriTile.Triton.MemCell.of .int (metadata (addr (base m .oldStarts) i)).toInt
      else if r = "length" then VeriTile.Triton.MemCell.of .int (metadata (addr (base m .lengths) i)).toInt
      else if r = "new" then VeriTile.Triton.MemCell.of .int (metadata (addr (base m .newStarts) i)).toInt
      else if r = "source" then VeriTile.Triton.MemCell.real (payload (addr (base m .source) i))
      else if r = "output" then VeriTile.Triton.MemCell.real (payload (addr (base m .output) i))
      else scratch.mem r i }

theorem logicalState_binding (m : Inputs) (segments : Nat → Segment) (metadata : Memory)
    (payload : Nat → ℝ) (scratch : VeriTile.Triton.BlockState)
    (binding : TensorBinding m segments metadata) :
    KernelMetadataBinding m segments (logicalState m metadata payload scratch) := by
  constructor
  · intro p hp
    simp [logicalState, VeriTile.Triton.BlockState.readMemValue,
      VeriTile.Triton.BlockState.readMemTyped, binding.oldCells p hp]
  · intro p hp
    simp [logicalState, VeriTile.Triton.BlockState.readMemValue,
      VeriTile.Triton.BlockState.readMemTyped, binding.lengthCells p hp]
  · intro p hp
    simp [logicalState, VeriTile.Triton.BlockState.readMemValue,
      VeriTile.Triton.BlockState.readMemTyped, binding.newCells p hp]

theorem logicalState_payload (m : Inputs) (metadata : Memory) (payload : Nat → ℝ)
    (scratch : VeriTile.Triton.BlockState) (i : Nat) :
    (logicalState m metadata payload scratch).readMem "source" i = payload (addr (base m .source) i) ∧
    (logicalState m metadata payload scratch).readMem "output" i = payload (addr (base m .output) i) := by
  simp [logicalState, VeriTile.Triton.BlockState.readMem, VeriTile.Triton.BlockState.readMemValue,
    VeriTile.Triton.BlockState.readMemTyped]

#axiomsClean checkInputs_iff
#axiomsClean checkArguments_iff
#axiomsClean contract_valid
#axiomsClean prepared_direct
#axiomsClean logicalState_binding
#axiomsClean logicalState_payload
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy
