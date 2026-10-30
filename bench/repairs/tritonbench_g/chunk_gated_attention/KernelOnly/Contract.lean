import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Tensor
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostStorage

/-! Kernel-only contracts preserve the selected Python guards and ordinary
padding/copy API relations. They do not assume compilation, execution, or any
arithmetic result. The selected wrapper skips both kernels on empty inputs. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
open HostStorage

def innerDescriptors (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before) :=
  r.tensors ++ p.tensors
def prefixDescriptors (r : Admission.Prefix) (before : Memory) (p : PrefixPrepared r before) :=
  [r.input,p.input,p.output]

structure InnerContract (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before) : Prop where
  checked : r.check = Bool.true
  active : r.nonempty
  pointers : p.check = Bool.true

structure PrefixContract (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) : Prop where
  checked : r.check = Bool.true
  active : r.nonempty
  selected : BS = 16 ∨ BS = 32 ∨ BS = 64
  pointers : p.check = Bool.true

theorem inner_empty_history (r : Admission.Inner) (checked : r.check = Bool.true)
    (empty : ¬r.nonempty) : innerCount r = 0 := by
  have he : r.B*r.H*r.T*r.K*r.V = 0 := not_not.mp empty
  have bt := (r.blocks checked).1
  have nt0 (ht : r.T = 0) : r.config.NT = 0 := by
    simp [Admission.Inner.config,HostFP.Inner.NT,HostFP.chunks,ht,
      Nat.div_eq_of_lt (show r.BT-1 < r.BT by omega)]
  rcases mul_eq_zero.mp he with hab|hv
  · rcases mul_eq_zero.mp hab with hab|hk
    · rcases mul_eq_zero.mp hab with hbh|ht
      · simp [innerCount,hbh]
      · simp [innerCount,nt0 ht]
    · simp [innerCount,hk]
  · simp [innerCount,hv]

theorem prefix_empty_output (r : Admission.Prefix) (empty : ¬r.nonempty) : prefixCount r = 0 :=
  not_not.mp empty

#specNonCircular InnerContract avoiding [FP.Structural.exec,VeriTile.Triton.exec]
#specNonCircular PrefixContract avoiding [FP.Structural.exec,VeriTile.Triton.exec]
#axiomsClean inner_empty_history
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
