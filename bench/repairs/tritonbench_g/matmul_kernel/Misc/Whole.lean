import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Admitted
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Host
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Scheduling
namespace VeriTile.Bench.Repairs.MatmulKernel.Public
open Native Physical Host
structure NativeClaim (c : Config) (actual target : Instructions) (initial : Bytes) (result : Result Byte) : Prop where
  returned : result.returned=.none
  identity : result.outputIdentity=.c
  source : load result.memory=MatmulKernel.source c (wordOps actual) (load initial)
  bytes : ∀ w,c.Active w → ∀ i : Fin 4,
    result.memory .c (4*c.C.addr w.1 w.2+i.val)=encode (reference c target (load initial) w.1 w.2) i
  frame : ∀ r b,(∀ w,c.Active w → ¬Writes c w r (b/4)) → result.memory r b=initial r b
structure RealClaim (c : Config) (initial : Memory ℝ) (result : Result ℝ) : Prop where
  returned : result.returned=.none
  identity : result.outputIdentity=.c
  source : result.memory=MatmulKernel.source c realOps initial
  values : ∀ w,c.Active w → result.memory .c (c.C.addr w.1 w.2)=mathematical c initial w.1 w.2
  frame : ∀ r a,(∀ w,c.Active w → ¬Writes c w r a) → result.memory r a=initial r a

theorem whole_real (c : Config) (domain : Admitted c) (initial : Memory ℝ) :
    RealClaim c initial (realCall c initial) := by
  refine ⟨rfl,rfl,real_memory c initial,caller_mathematical c domain.config initial,?_⟩
  intro r a outside
  rw [real_memory,source_run]
  exact run_frame c realOps initial r a outside

theorem whole_native (c : Config) (domain : Admitted c) (p : Instructions) (initial : Bytes) :
    NativeClaim c p p initial (call c p initial) := by
  refine ⟨rfl,rfl,?_,caller_bytes c domain.config p initial,caller_frame c p initial⟩
  rw [memory_native]
  exact native_source c p initial

theorem whole_fp (c : Config) (domain : Admitted c) (actual target : Instructions)
    (ports : Obligations actual target) (initial : Bytes) :
    NativeClaim c actual target initial (call c actual initial) := by
  refine ⟨rfl,rfl,?_,?_,caller_frame c actual initial⟩
  · rw [memory_native]
    exact native_source c actual initial
  · intro w hw i
    rw [caller_local_transport c actual target ports initial]
    exact caller_bytes c domain.config target initial w hw i
#axiomsClean whole_real
#axiomsClean whole_native
#axiomsClean whole_fp
end VeriTile.Bench.Repairs.MatmulKernel.Public
