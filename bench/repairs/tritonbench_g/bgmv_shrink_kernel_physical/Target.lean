import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPValues
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.CheckedReal
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical
open VeriTile.Triton FP.Structural
namespace Target
noncomputable section
open scoped BigOperators
variable {α : Type} [Inhabited α]
abbrev Data := VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Data
abbrev Primitives := VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Primitives

def format : Storage → FloatDType | .fp16 => .fp16 | .bf16 => .bf16

/-- The independent dense mathematical operation starts from the supplied Y. -/
def real (h : Call) (initial : BlockState) (indices : Nat → Int)
    (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  let old := initial.readMem "Y" (b.val*h.columns+n.val)
  if indices b.val=-1 then old else old+h.scaling*∑ k : Fin h.K,
    initial.readMem "X" (b.val*h.K+k.val)*
    initial.readMem "W" ((indices b.val).toNat*(h.N*h.K)+n.val*h.K+k.val)

def loaded (M : Algebra α) (h : Call) (data : Data α) (batch split c : Nat) (e : Fin 256) : α :=
  if (c*16384+e.val+split*256) < h.K then data.input ((batch*h.K+((c*16384+e.val+split*256))))
  else M.cast (some .fp32) .real (format h.storage) (M.literal (some .fp32) .real 0.0)

def weight (M : Algebra α) (h : Call) (data : Data α) (slot split c : Nat) (n : Fin h.BN) (e : Fin 256) : α :=
  if n.val < h.N ∧ (c*16384+e.val+split*256) < h.K then
    data.weights ((slot*(h.N*h.K)+n.val*h.K+((c*16384+e.val+split*256))))
  else M.cast (some .fp32) .real (format h.storage) (M.literal (some .fp32) .real 0.0)

def products (M : Algebra α) (h : Call) (data : Data α) (batch slot split c : Nat) :
    Values α (format h.storage).toTileDType [h.BN,256] :=
  fun (n,e,_) => ofFloat _ (M.binary (some .fp32) (format h.storage) .mul
    (loaded M h data batch split c e) (weight M h data slot split c n e))

def sums (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) (batch slot split c : Nat) :
    Fin h.BN → α := fun n => toFloat _ (P.reduction.apply (some .fp32) (format h.storage)
      ⟨1,by simp⟩ Bool.false (products M h data batch slot split c) (n,PUnit.unit))

/-- Exact ordered source recurrence, including the initial fp32 zero and all
half/bfloat16 multiply/reduce steps before fp32 extension and accumulation. -/
def accumulator (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (batch slot split : Nat) : Nat → Fin h.BN → α
  | 0,_ => M.literal (some .fp32) .real 0
  | c+1,n => M.binary (some .fp32) .real .add (accumulator M P h data batch slot split c n)
      (M.cast (some .fp32) (format h.storage) .real (sums M P h data batch slot split c n))

/-- Preserve scaling and the implicit atomic-operand conversion separately. -/
def payload (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (batch slot split : Nat) (n : Fin h.BN) : α :=
  M.cast (some .fp32) .real h.outputType (M.binary (some .fp32) .real .mul
    (accumulator M P h data batch slot split h.cycles n) (data.scaling))

/-- Each cell uses its actual atomic order, including K=0 contributions.
No sum-tree normalization or output oracle is used in this FP target. -/
def target (M : Algebra α) (P : Primitives α) (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (data : Data α) (b : Fin h.batches) (n : Fin h.N) : α :=
  if data.indices b.val=-1 then data.output ((b.val*h.columns+n.val)) else
    (orders.order b n).foldl (fun old split => P.atomic h.outputType old
      (payload M P h data b.val (data.indices b.val).toNat split.val ⟨n.val,n.isLt.trans_le h.rowsCovered⟩))
      (data.output ((b.val*h.columns+n.val)))


#specNonCircular real avoiding [VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal.Run,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution.contribution,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Kernels.source,VeriTile.Triton.exec]
#specNonCircular target avoiding [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.target,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Kernels.source,FP.Structural.exec]
#auditModuleAxioms
end
end Target
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical
