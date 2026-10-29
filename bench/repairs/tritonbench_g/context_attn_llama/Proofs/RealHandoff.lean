import bench.repairs.tritonbench_g.context_attn_llama.Proofs.Wrapper
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.Views

namespace VeriTile.Bench.Repairs.ContextAttnLlama.RealHandoff
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000

/-- An independent interpretation of initial logical words as mathematical
inputs. This never evaluates a floating operation or asserts FP=Real. A usual
finite-word interpretation may be chosen with any explicit exceptional extension. -/
def inputs (lift : BitVec 32 → ℝ) (input : Initial c) : Formula.Inputs where
  q := fun n => lift (input.query.word input.before n)
  k := fun n => lift (input.key.word input.before n)
  v := fun n => lift (input.value.word input.before n)
  start := input.start.integer input.before
  sequence := input.sequence.integer input.before
  prompt := input.prompt.integer input.before
  request := input.request.integer input.before
  pages := input.pages.integer input.before

def leaf (lift : BitVec 32 → ℝ) : NativeGraph.Expr → ℝ
  | .leaf word => lift word
  | _ => 0

/-- Only the prepared initial cells are handed to the existing Real execution.
This is not a second floating interpreter or a byte serialization layer. -/
def cell (lift : BitVec 32 → ℝ) : Cell NativeGraph.Expr → MemCell
  | .mk .int value => MemCell.of .int value
  | .mk .real value => MemCell.real (leaf lift value)
  | _ => MemCell.real 0

def state (lift : BitVec 32 → ℝ) (prepared : Runtime) : BlockState where
  mem := fun rg n => cell lift (prepared.kernel.mem rg n)
  regs := fun _ _ _ => none
  pids := prepared.kernel.pids
  numPids := prepared.kernel.numPids

lemma read_word (lift : BitVec 32 → ℝ) (prepared : Runtime) (rg : RegionName) (n : Nat)
    (word : BitVec 32) (h : (prepared.kernel.mem rg n).read .real = .leaf word) :
    (state lift prepared).readMem rg n = lift word := by
  change (match (cell lift (prepared.kernel.mem rg n)).readAs .real with
    | some (some value) => value | _ => 0) = lift word
  generalize hx : prepared.kernel.mem rg n = x at *
  cases x with
  | mk dtype value =>
    cases dtype <;> simp [Cell.read,defaultValue] at h
    all_goals subst value
    all_goals rfl

lemma read_integer (lift : BitVec 32 → ℝ) (prepared : Runtime) (rg : RegionName) (n : Nat) :
    (state lift prepared).readMemValue .int rg n = (prepared.kernel.mem rg n).read .int := by
  change (match (cell lift (prepared.kernel.mem rg n)).readAs .int with
    | some value => value | none => 0) = (prepared.kernel.mem rg n).read .int
  cases hx : prepared.kernel.mem rg n with
  | mk dtype value =>
    cases dtype <;> rfl

/-- Reads is derived from the actual successful ordered preparation and the
original caller words, for every logical offset and all eight read roles. -/
theorem prepared_reads (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial prepared : Runtime) (d : Dimensions)
    (ready : runPrefix api.api request initial = .ready d prepared)
    (lift : BitVec 32 → ℝ) :
    Views.Reads c.regions (state lift prepared) (inputs lift input) := by
  have hw := prepared_words c input api request initial prepared d ready
  have hb := (prefix_binds api.api request initial prepared d ready).2
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
  · funext n; exact read_word lift prepared _ _ _ (hw.1 n)
  · funext n; exact read_word lift prepared _ _ _ (hw.2.1 n)
  · funext n; exact read_word lift prepared _ _ _ (hw.2.2.1 n)
  · funext n; exact (read_integer lift prepared _ n).trans
      (hb .contiguousStart (by simp [normalization,setup]) n)
  · funext n; exact (read_integer lift prepared _ n).trans
      (hb .contiguousSequence (by simp [normalization,setup]) n)
  · funext n; exact (read_integer lift prepared _ n).trans
      (hb .contiguousPrompt (by simp [normalization,setup]) n)
  · funext n; exact (read_integer lift prepared _ n).trans
      (hb .contiguousRequest (by simp [normalization,setup]) n)
  · funext n; exact (read_integer lift prepared _ n).trans
      (hb .resolvePages (by simp [normalization,setup]) n)

/-- The unchanged finite allocation/coordinate conditions are transferred for
the same c and A. All state-dependent metadata fields are re-derived from the
caller input and preparation, rather than borrowed from the geometry witness. -/
theorem prepared_valid (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial prepared : Runtime) (d : Dimensions)
    (ready : runPrefix api.api request initial = .ready d prepared)
    (lift : BitVec 32 → ℝ) (A : FlatAlloc) (geometryState : BlockState)
    (valid : Host.Valid c A geometryState) : Host.Valid c A (state lift prepared) := by
  have hi := (prepared_reads c input api request initial prepared d ready lift).inputs_eq
  exact { valid with
    start_value := by intro b hb; rw [hi]; exact input.start_value b hb
    length_value := by intro b hb; rw [hi]; exact input.sequence_value b hb
    prompt_value := by intro b hb; rw [hi]; exact input.prompt_value b hb
    request_value := by intro b hb; rw [hi]; exact input.request_value b hb
    page_value := by intro b hb j hj; rw [hi]; exact input.pages_value b hb j hj }

#print axioms prepared_reads
#print axioms prepared_valid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.RealHandoff
