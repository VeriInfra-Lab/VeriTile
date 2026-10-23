import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelReal
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelNative
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelNativeTensor
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelGeometry

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelEntries
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelTensor
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def scalarCell (mode : Mode) (value : α) : Cell α := .mk (dtype mode).toTileDType (ofFloat (dtype mode) value)

def logicalIndex (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (i : TileIndex [(configs c).bm,(configs c).bn]) : Nat :=
  KernelValues.row (parameters r c regions) s i.1*r.n+KernelValues.col (parameters r c regions) s i.2.1

/-- Actual Tensor destinations for all active lanes, including optional saved
storage, together with every untouched tagged cell. The source masks determine
the exact footprint; arbitrary suffix and other-region cells remain framed. -/
structure TensorResult (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (before after : State α)
    (out saved : Nat → Nat → α) : Prop where
  output : ∀ i : TileIndex [(configs c).bm,(configs c).bn],
    active (parameters r c regions) before i=Bool.true →
    after.mem (regions 0) (physicalIndex r.mode (prepared.tensors 0) (logicalIndex r c regions before i))=
      scalarCell r.mode (out (KernelValues.row (parameters r c regions) before i.1)
        (KernelValues.col (parameters r c regions) before i.2.1))
  saved : r.hasSaved=Bool.true → ∀ i : TileIndex [(configs c).bm,(configs c).bn],
    active (parameters r c regions) before i=Bool.true →
    after.mem (regions 1) (physicalIndex r.mode (prepared.tensors 1) (logicalIndex r c regions before i))=
      scalarCell r.mode (saved (KernelValues.row (parameters r c regions) before i.1)
        (KernelValues.col (parameters r c regions) before i.2.1))
  frame : ∀ region offset,
    (region≠regions 0 ∨ ∀ i, active (parameters r c regions) before i=Bool.true →
      offset≠KernelValues.outputOffset (parameters r c regions) before i) →
    (r.hasSaved=Bool.false ∨ region≠regions 1 ∨ ∀ i, active (parameters r c regions) before i=Bool.true →
      offset≠savedOffset (parameters r c regions) before i) → after.mem region offset=before.mem region offset

theorem destination_offset (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (s : State α) (i : TileIndex [(configs c).bm,(configs c).bn])
    (enabled : active (parameters r c regions) s i=Bool.true) (slot : Fin 5)
    (destination : slot=0 ∨ slot=1) (live : used r slot) :
    physicalIndex r.mode (prepared.tensors slot) (logicalIndex r c regions s i)=
      KernelValues.outputOffset (parameters r c regions) s i := by
  have bounds : KernelValues.row (parameters r c regions) s i.1<r.batch.rows ∧
      KernelValues.col (parameters r c regions) s i.2.1<r.n := by simpa [active,parameters] using enabled
  have index := (destination_index r regions prepared slot destination live _ _ bounds.1 bounds.2).1
  rw [logicalIndex,index]
  simp only [KernelValues.outputOffset,parameters,max_eq_left (show 1≤r.n by omega),Nat.mul_one]

def Formed (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (s : State α) : Prop :=
  ∀ i : TileIndex [(configs c).bm,(configs c).bn],
    KernelValues.row (parameters r c regions) s i.1<2^31 ∧
    KernelValues.col (parameters r c regions) s i.2.1<2^31 ∧
    KernelValues.outputOffset (parameters r c regions) s i<2^31

/-- Kernel-only Real theorem on actual prepared Tensor pointers, for every
original autotune configuration. Caller admission and prepared Tensor storage
facts are ordinary premises; execution, all stored values, and frame are derived. -/
specification correct (r : Request) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (raw : Nat → ℝ) (s : State ℝ)
    (binding : StorageBound r regions prepared raw s) (pid : s.pids 0<r.shape.programs c) :
    ∃ final, exec (engine KernelRealArithmetic.algebra KernelRealArithmetic.dot)
      (source r.mode (parameters r c regions)) s=some final ∧
      TensorResult r c regions prepared s final (denseOutput r regions prepared raw) (densePre r regions prepared raw) ∧
      Formed r c regions s := by
  have separate : (parameters r c regions).saved≠(parameters r c regions).output :=
    prepared.regionDistinct.ne (by decide)
  obtain ⟨final,execution,out,saved,frame⟩ := KernelReal.outputs r.mode (parameters r c regions) s
    (configured_positive r c regions) (configured_strides r c regions) separate
  simp only [show (parameters r c regions).output=regions 0 from rfl] at out
  simp only [show (parameters r c regions).saved=regions 1 from rfl,
    show (parameters r c regions).saveInputs=r.hasSaved from rfl] at saved
  refine ⟨final,execution,⟨?_,?_,frame⟩,?_⟩
  · intro i enabled
    have bounds : KernelValues.row (parameters r c regions) s i.1<r.batch.rows ∧
        KernelValues.col (parameters r c regions) s i.2.1<r.n := by simpa [active,parameters] using enabled
    rw [destination_offset r c regions prepared s i enabled 0 (Or.inl rfl) (by simp [used]),out i enabled]
    have reference := dense_bridge r c regions prepared raw s binding _ _ bounds.1 bounds.2
    change outputCell KernelRealArithmetic.algebra r.mode
      (realActivation r.activation (KernelRealArithmetic.densePre r.mode (parameters r c regions) s _ _))=_
    rw [reference]
    rfl
  · intro flag i enabled
    have bounds : KernelValues.row (parameters r c regions) s i.1<r.batch.rows ∧
        KernelValues.col (parameters r c regions) s i.2.1<r.n := by simpa [active,parameters] using enabled
    rw [destination_offset r c regions prepared s i enabled 1 (Or.inr rfl) (by simp [used,flag])]
    have address : KernelValues.outputOffset (parameters r c regions) s i=savedOffset (parameters r c regions) s i := by
      simp only [KernelValues.outputOffset,savedOffset,parameters,flag,if_true]
    rw [address,saved flag i enabled,dense_bridge r c regions prepared raw s binding _ _ bounds.1 bounds.2]
    rfl
  · exact KernelGeometry.formed_output_bound r c regions s ts flag admit pid

/-- Native word theorem with an independent actual-Tensor expression target.
Compiler policy is explicit, and fixed-lane ISA/libdevice ports remain the
stated machine-semantics boundary. There is no host-engine or copyback premise. -/
specification fp_equiv (profile : LocalProfile) (policy : KernelNativeModel.CompilerPolicy)
    (r : Request) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (raw : Nat → Word) (s : State Word)
    (binding : StorageBound r regions prepared raw s) (pid : s.pids 0<r.shape.programs c) :
    ∃ final, KernelNativeModel.execute profile policy c r.mode (parameters r c regions) s=some final ∧
      TensorResult r c regions prepared s final
        (KernelNativeTensor.resultWord profile policy r c regions prepared raw)
        (KernelNativeTensor.savedWord profile policy r c regions prepared raw) ∧
      Formed r c regions s := by
  have separate : (parameters r c regions).saved≠(parameters r c regions).output :=
    prepared.regionDistinct.ne (by decide)
  obtain ⟨final,execution,out,saved,frame⟩ := KernelNative.outputs profile policy c r.mode (parameters r c regions) s
    (configured_positive r c regions) (configured_strides r c regions) separate
  simp only [show (parameters r c regions).output=regions 0 from rfl] at out
  simp only [show (parameters r c regions).saved=regions 1 from rfl,
    show (parameters r c regions).saveInputs=r.hasSaved from rfl] at saved
  refine ⟨final,execution,⟨?_,?_,frame⟩,?_⟩
  · intro i enabled
    have bounds : KernelValues.row (parameters r c regions) s i.1<r.batch.rows ∧
        KernelValues.col (parameters r c regions) s i.2.1<r.n := by simpa [active,parameters] using enabled
    rw [destination_offset r c regions prepared s i enabled 0 (Or.inl rfl) (by simp [used]),out i enabled]
    unfold KernelNativeModel.resultWord
    rw [KernelNativeTensor.pre_bridge r c regions prepared raw s binding i bounds.1 bounds.2]
    rfl
  · intro flag i enabled
    have bounds : KernelValues.row (parameters r c regions) s i.1<r.batch.rows ∧
        KernelValues.col (parameters r c regions) s i.2.1<r.n := by simpa [active,parameters] using enabled
    rw [destination_offset r c regions prepared s i enabled 1 (Or.inr rfl) (by simp [used,flag])]
    have address : KernelValues.outputOffset (parameters r c regions) s i=savedOffset (parameters r c regions) s i := by
      simp only [KernelValues.outputOffset,savedOffset,parameters,flag,if_true]
    rw [address,saved flag i enabled]
    unfold KernelNativeModel.savedWord KernelNativeModel.preWord
    rw [KernelNativeTensor.pre_bridge r c regions prepared raw s binding i bounds.1 bounds.2]
    rfl
  · exact KernelGeometry.formed_output_bound r c regions s ts flag admit pid

#axiomsClean correct
#axiomsClean fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelEntries
