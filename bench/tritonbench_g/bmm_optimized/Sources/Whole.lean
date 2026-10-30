import bench.tritonbench_g.bmm_optimized.Sources.Host
import bench.tritonbench_g.bmm_optimized.Sources.SourceIndex
namespace VeriTile.Bench.Repairs.BmmOptimized.Public
open Native Host

structure NativeClaim (call : Call) (c : Config) (actual target : Instructions)
    (callerDevice : Nat) (initial : Bytes) (result : Returned Byte) : Prop where
  returned : result.identity=call.freshIdentity
  fresh : result.identity≠call.a.identity ∧ result.identity≠call.b.identity
  shape : result.shape=(call.shape.batch,call.shape.m,call.shape.n)
  strides : result.strides=outputStrides call.shape
  dtype : result.fp32=true
  device : result.device=call.a.device ∧ result.restoredDevice=callerDevice
  source : load result.memory=source call.shape c (scalar actual) (prepare actual.neg call (load initial))
  bytes : ∀ x,call.shape.Active x → ∀ i : Fin 4,
    result.memory .output (4*call.shape.o x.batch x.row x.column+i.val)=expected target call initial x i
  frame : ∀ r a,CallerFrame r a call.shape → result.memory r a=initial r a

theorem whole_native (call : Call) (domain : Valid call) (c : Config) (p : Instructions)
    (callerDevice : Nat) (initial : Bytes) :
    NativeClaim call c p p callerDevice initial (invoke call c p callerDevice initial) := by
  refine ⟨rfl,⟨domain.outputA,domain.outputB⟩,rfl,rfl,rfl,⟨rfl,rfl⟩,?_,?_,?_⟩
  · rw [invoke_memory,load_selected,load_prepare]
  · exact Host.whole_native call c p callerDevice initial
  · exact whole_frame call c p callerDevice initial
theorem whole_fp (call : Call) (domain : Valid call) (c : Config) (actual target : Instructions)
    (ports : Obligations actual target) (callerDevice : Nat) (initial : Bytes) :
    NativeClaim call c actual target callerDevice initial (invoke call c actual callerDevice initial) := by
  rw [←instructions_eq actual target ports]
  exact whole_native call domain c actual callerDevice initial

structure RealClaim (call : Call) (c : Config) (callerDevice : Nat)
    (initial : Memory ℝ) (result : Returned ℝ) : Prop where
  returned : result.identity=call.freshIdentity
  fresh : result.identity≠call.a.identity ∧ result.identity≠call.b.identity
  shape : result.shape=(call.shape.batch,call.shape.m,call.shape.n)
  strides : result.strides=outputStrides call.shape
  dtype : result.fp32=true
  device : result.device=call.a.device ∧ result.restoredDevice=callerDevice
  source : result.memory=source call.shape c realScalar (prepare Neg.neg call initial)
  values : ∀ x,call.shape.Active x →
    result.memory .output (call.shape.o x.batch x.row x.column)=product call initial x
  callerA : ∀ a,result.memory .callerA a=initial .callerA a
  callerB : ∀ a,result.memory .callerB a=initial .callerB a
  other : ∀ id a,result.memory (.other id) a=initial (.other id) a
  outside : ∀ a,call.shape.size≤a → result.memory .output a=initial .output a

theorem whole_real (call : Call) (domain : Valid call) (c : Config)
    (callerDevice : Nat) (initial : Memory ℝ) :
    RealClaim call c callerDevice initial (invokeReal call c callerDevice initial) := by
  refine ⟨rfl,⟨domain.outputA,domain.outputB⟩,rfl,rfl,rfl,⟨rfl,rfl⟩,
    invokeReal_memory call c callerDevice initial,Host.whole_real call c callerDevice initial,?_,?_,?_,?_⟩
  · intro a;rw [invokeReal_memory,source_frame _ c realScalar _ .callerA a (Or.inl (by simp))];rfl
  · intro a;rw [invokeReal_memory,source_frame _ c realScalar _ .callerB a (Or.inl (by simp))];rfl
  · intro id a;rw [invokeReal_memory,source_frame _ c realScalar _ (.other id) a (Or.inl (by simp))];rfl
  · intro a ha;rw [invokeReal_memory,source_frame _ c realScalar _ .output a (Or.inr ha)];rfl

#axiomsClean whole_real
#axiomsClean whole_native
#axiomsClean whole_fp
end VeriTile.Bench.Repairs.BmmOptimized.Public
