import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Backends
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Contract

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Native
open FusedRwkv6Kernel
noncomputable section

/-- The explicit FP32 casts in the structural interpreter are move/barrier
nodes. Input and output casts use the same typed cast primitive. -/
def referenceOps : Ops Term := ⟨.zero,.store32,.store32,.add,.mul,.exp,.load32⟩
theorem source_ops (p : ReductionPlan) : Source.ops (algebra p)=referenceOps := rfl

/-- Triton's raw integer 1 is a constexpr specialization, unlike float 1.0.
Only the designated scalar leaf is folded. This is a binding/topology choice,
not an IEEE identity, and it never tests any tensor or FP32 payload. -/
def specialize (one : Bool) : Term → Term
  | .zero => .zero
  | .word r b bits => .word r b bits
  | .load32 x => .load32 (specialize one x)
  | .store32 x => .store32 (specialize one x)
  | .add x y => .add (specialize one x) (specialize one y)
  | .mul x y => match one,y with
    | Bool.true,.word .outside 0 _ => specialize one x
    | _,_ => .mul (specialize one x) (specialize one y)
  | .exp x => .exp (specialize one x)

theorem specialize_false (t : Term) : specialize Bool.false t=t := by
  induction t <;> simp_all [specialize]

def isOne (raw : PythonNumber) : Bool := decide (raw=.integer 1)
def observeBound (a : Instructions) (compiler : Compiler) (raw : PythonNumber) (t : Term) : BitVec 32 :=
  observe a compiler (specialize (isOne raw) t)

theorem int_one_query (a : Instructions) (compiler : Compiler) (binding : ScalarBinding) (q : Term) :
    observeBound a compiler (.integer 1) (.mul q (scalar binding (.integer 1)))=
      observeBound a compiler (.integer 1) q := by
  simp [observeBound,isOne,specialize,scalar]

theorem float_query (a : Instructions) (compiler : Compiler) (binding : ScalarBinding)
    (bits : BitVec 64) (q : Term) :
    observeBound a compiler (.float bits) (.mul q (scalar binding (.float bits)))=
      observe a compiler (.mul q (scalar binding (.float bits))) := by
  simp [observeBound,isOne,specialize,scalar,specialize_false]

#axiomsClean specialize_false
#axiomsClean int_one_query
#axiomsClean float_query
#specNonCircular specialize avoiding [VeriTile.Triton.FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Native
