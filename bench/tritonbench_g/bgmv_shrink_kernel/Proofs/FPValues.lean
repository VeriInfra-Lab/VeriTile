import bench.tritonbench_g.bgmv_shrink_kernel.Types
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_FPValues
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Workset
import bench.tritonbench_g.bgmv_expand_slice.Sources.FPValues

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
variable {α : Type} [Inhabited α]
abbrev Reduction := VeriTile.Bench.Repairs.BgmvExpandSlice.FPValues.Reduction

/-- A local reduction backend retains the entire typed tensor, shape, axis,
precision and keepDims. Atomic addition has a distinct local primitive. No
finite, commutative, associative, cast-idempotence or whole-output law is used. -/
abbrev Primitives := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.FPValues.Primitives

def machine (M : Algebra α) (P : Primitives α) : Algebra α :=
  ({M with reduceSumTyped := fun p d {_} ax keep v => some (P.reduction.apply p d ax keep v)
           atomicAdd := fun d => some (P.atomic d)}).withDefaultPrecision .fp32

def dtype : Storage → FloatDType | .fp16 => .fp16 | .bf16 => .bf16

structure Data (α : Type) where
  input : Nat → α
  weights : Nat → α
  output : Nat → α
  scaling : α
  indices : Nat → Int
  other : RegionName → Nat → Cell α
  padding : (d : TileDType) → RegionName → Nat → Value α d

def Data.cells (h : Call) (data : Data α) (r : RegionName) (off : Nat) : Cell α :=
  if r="X" then .mk (dtype h.storage).toTileDType (ofFloat _ (data.input off))
  else if r="W" then .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights off))
  else if r="Y" then .mk h.outputType.toTileDType (ofFloat _ (data.output off))
  else if r="Indices" then .mk .int (data.indices off)
  else data.other r off

def entry (h : Call) (batch split : Nat) (data : Data α) : State α :=
  ({ mem := data.cells h
     regs := fun _ _ _ => none
     pids := fun ax => if ax=0 then split else if ax=1 then batch else 0
     numPids := fun ax => if ax=0 then 64 else if ax=1 then h.batches else 1
     undef := data.padding } : State α).setReg "scaling" .real [] (fun _ => data.scaling)

def loaded (M : Algebra α) (h : Call) (data : Data α) (batch split c : Nat) (e : Fin 256) : α :=
  if rank split c e.val<h.K then data.input (inputOffset h batch (rank split c e.val))
  else M.cast (some .fp32) .real (dtype h.storage) (M.literal (some .fp32) .real 0.0)

def weight (M : Algebra α) (h : Call) (data : Data α) (slot split c : Nat) (n : Fin h.BN) (e : Fin 256) : α :=
  if n.val<h.N ∧ rank split c e.val<h.K then
    data.weights (weightOffset h slot n.val (rank split c e.val))
  else M.cast (some .fp32) .real (dtype h.storage) (M.literal (some .fp32) .real 0.0)

def products (M : Algebra α) (h : Call) (data : Data α) (batch slot split c : Nat) :
    Values α (dtype h.storage).toTileDType [h.BN,256] :=
  fun (n,e,_) => ofFloat _ (M.binary (some .fp32) (dtype h.storage) .mul
    (loaded M h data batch split c e) (weight M h data slot split c n e))

def sums (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (batch slot split c : Nat) :
    Fin h.BN → α := fun n => toFloat _ (P.reduction.apply (some .fp32) (dtype h.storage)
      ⟨1,by simp⟩ Bool.false (products M h data batch slot split c) (n,PUnit.unit))

/-- Exact ordered source recurrence, including the initial fp32 zero and all
half/bfloat16 multiply/reduce steps before fp32 extension and accumulation. -/
def accumulator (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (batch slot split : Nat) : Nat → Fin h.BN → α
  | 0,_ => M.literal (some .fp32) .real 0
  | c+1,n => M.binary (some .fp32) .real .add (accumulator M P h data batch slot split c n)
      (M.cast (some .fp32) (dtype h.storage) .real (sums M P h data batch slot split c n))

/-- Preserve scaling and the implicit atomic-operand conversion separately. -/
def payload (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (batch slot split : Nat) (n : Fin h.BN) : α :=
  M.cast (some .fp32) .real h.outputType (M.binary (some .fp32) .real .mul
    (accumulator M P h data batch slot split h.cycles n) (data.scaling))

/-- Each cell uses its actual atomic order, including K=0 contributions.
No sum-tree normalization or output oracle is used in this FP target. -/
def target (M : Algebra α) (P : Primitives α) (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (data : Data α) (b : Fin h.batches) (n : Fin h.N) : α :=
  if data.indices b.val=-1 then data.output (outputOffset h b.val n.val) else
    (orders.order b n).foldl (fun old split => P.atomic h.outputType old
      (payload M P h data b.val (data.indices b.val).toNat split.val ⟨n.val,n.isLt.trans_le h.rowsCovered⟩))
      (data.output (outputOffset h b.val n.val))

end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues
