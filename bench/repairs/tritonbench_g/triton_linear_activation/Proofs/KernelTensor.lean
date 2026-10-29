import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelRealArithmetic

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelTensor
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Pre-state storage interpretation, one typed cell at each actual allocation
address. This API boundary contains no post-state or arithmetic-result premise. -/
structure StorageBound (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → α) (s : State α) : Prop where
  cells : ∀ slot, used r slot → ∀ offset, offset<storageElements r.mode (prepared.tensors slot) →
    s.mem (regions slot) offset=Cell.mk (dtype r.mode).toTileDType (ofFloat (dtype r.mode)
      (raw (storageBase r.mode (prepared.tensors slot)+elementBytes r.mode*offset)))

/-- Logical value of a bound actual Tensor. Prepared excludes lazy negation,
so the value is its actual backing word (or ideal scalar interpretation). -/
def logical (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → α) (slot : Fin 5) (i : Nat) : α :=
  raw (storageBase r.mode (prepared.tensors slot)+
    elementBytes r.mode*physicalIndex r.mode (prepared.tensors slot) i)

theorem bound_read (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → α) (s : State α) (bound : StorageBound r regions prepared raw s)
    (slot : Fin 5) (live : used r slot) (i : Nat)
    (valid : i<(metadata r.mode (prepared.tensors slot)).numel) :
    KernelValues.read r.mode s (regions slot) i=logical r regions prepared raw slot i := by
  have physical := physical_contiguous r.mode (prepared.tensors slot)
    (prepared.contiguous slot live) (prepared.offsetZero slot live) i valid
  have cap := capacity r.mode (prepared.tensors slot)
    (prepared.contiguous slot live) (prepared.offsetZero slot live) i valid
  unfold KernelValues.read logical
  rw [bound.cells slot live i cap,physical]
  have roundtrip (d : FloatDType) (value : α) :
      toFloat d ((Cell.mk d.toTileDType (ofFloat d value)).read d.toTileDType)=value := by
    cases d <;> simp [ofFloat,toFloat]
  exact roundtrip _ _

theorem numel (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (slot : Fin 5) (live : used r slot) :
    (metadata r.mode (prepared.tensors slot)).numel=(tensorShape r slot).prod := by
  simp only [VeriTile.Torch.TensorMetadata.numel,prepared.shape slot live]

theorem matrix_index (rows columns m n : Nat) (hm : m<rows) (hn : n<columns) :
    m*columns+n<rows*columns := by nlinarith

theorem x_read (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → ℝ) (s : State ℝ) (bound : StorageBound r regions prepared raw s)
    (m k : Nat) (hm : m<r.batch.rows) (hk : k<r.k) :
    KernelRealArithmetic.x r.mode (parameters r c regions) s m k=
      logical r regions prepared raw 2 (m*r.k+k) := by
  have live : used r 2 := by simp [used]
  have size := numel r regions prepared 2 live
  have index := matrix_index _ _ _ _ hm hk
  have valid : m*r.k+k<(metadata r.mode (prepared.tensors 2)).numel := by
    simpa [size,tensorShape] using index
  simpa [KernelRealArithmetic.x,parameters,max_eq_left (show 1≤r.k by omega)] using
    bound_read r regions prepared raw s bound 2 live (m*r.k+k) valid

theorem w_read (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → ℝ) (s : State ℝ) (bound : StorageBound r regions prepared raw s)
    (n k : Nat) (hn : n<r.n) (hk : k<r.k) :
    KernelRealArithmetic.w r.mode (parameters r c regions) s n k=
      logical r regions prepared raw 3 (n*r.k+k) := by
  have live : used r 3 := by simp [used]
  have size := numel r regions prepared 3 live
  have index := matrix_index _ _ _ _ hn hk
  have valid : n*r.k+k<(metadata r.mode (prepared.tensors 3)).numel := by
    simpa [size,tensorShape] using index
  simpa [KernelRealArithmetic.w,parameters,max_eq_left (show 1≤r.k by omega)] using
    bound_read r regions prepared raw s bound 3 live (n*r.k+k) valid

def densePre (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → ℝ) (m n : Nat) : ℝ :=
  (if r.hasBias then logical r regions prepared raw 4 n else 0)+
    ∑ k∈Finset.range r.k,logical r regions prepared raw 2 (m*r.k+k)*logical r regions prepared raw 3 (n*r.k+k)

def denseOutput (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → ℝ) (m n : Nat) : ℝ := realActivation r.activation (densePre r regions prepared raw m n)

theorem dense_bridge (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → ℝ) (s : State ℝ) (bound : StorageBound r regions prepared raw s)
    (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    KernelRealArithmetic.densePre r.mode (parameters r c regions) s m n=densePre r regions prepared raw m n := by
  unfold KernelRealArithmetic.densePre densePre
  congr 1
  · unfold KernelRealArithmetic.bias
    cases hb : r.hasBias
    · simp [parameters,hb]
    · have live : used r 4 := by simp [used,hb]
      have size := numel r regions prepared 4 live
      have valid : n<(metadata r.mode (prepared.tensors 4)).numel := by simpa [size,tensorShape] using hn
      simpa [parameters,hb] using bound_read r regions prepared raw s bound 4 live n valid
  · apply Finset.sum_congr rfl
    intro k hk
    have kbound : k<r.k := Finset.mem_range.mp hk
    rw [x_read r c regions prepared raw s bound m k hm kbound,
      w_read r c regions prepared raw s bound n k hn kbound]

theorem destination_index (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (slot : Fin 5) (destination : slot=0 ∨ slot=1) (live : used r slot)
    (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    physicalIndex r.mode (prepared.tensors slot) (m*r.n+n)=m*r.n+n ∧
      m*r.n+n<storageElements r.mode (prepared.tensors slot) := by
  have size := numel r regions prepared slot live
  have valid : m*r.n+n<(metadata r.mode (prepared.tensors slot)).numel := by
    rcases destination with rfl | rfl <;> simpa [size,tensorShape] using matrix_index _ _ _ _ hm hn
  exact ⟨physical_contiguous r.mode _ (prepared.contiguous slot live) (prepared.offsetZero slot live) _ valid,
    capacity r.mode _ (prepared.contiguous slot live) (prepared.offsetZero slot live) _ valid⟩

#axiomsClean bound_read
#axiomsClean dense_bridge
#axiomsClean destination_index
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelTensor
