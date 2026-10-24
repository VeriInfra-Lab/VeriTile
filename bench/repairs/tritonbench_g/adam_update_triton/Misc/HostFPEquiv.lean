import bench.repairs.tritonbench_g.adam_update_triton.Proofs.PublicFacts
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.HostFPEquiv
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
open Host
noncomputable section

abbrev Word := Native.Word
abbrev Memory := Bytes.Memory
abbrev Driver (rt : Runtime Memory) (native : Words.Instructions) :=
  ChunkRuntime Memory Word (Term Word) Bytes.ops (Machine.nativeModel (Words.ports native)) rt

/-- Independent raw-word recurrence, including exact FMA contraction/order,
neg.f32 versus sub(+0,lr), the unordered test and FP32 boolean multiplicand. -/
def target (native : Words.Instructions) (params : Scalars Word) (call : Call)
    (before : World Memory) (output : Output) (i : Nat) : Word :=
  Native.reference (Words.ports native) params output
    (logical Bytes.ops Bytes.sign before.memory call.request.p i)
    (logical Bytes.ops Bytes.sign before.memory call.request.grad i)
    (logical Bytes.ops Bytes.sign before.memory call.request.momentum i)

def Execution (rt : Runtime Memory) (native : Words.Instructions) (driver : Driver rt native)
    (convert : Scalar.Conversions Word) (compiler : Compilation.Compiler Memory)
    (call : Call) (args : Scalars (BitVec 64)) (params : Scalars Word) (before after : World Memory) : Prop :=
  Host.Execution Bytes.ops (Machine.nativeModel (Words.ports native)) rt driver convert call args params before after
    (fun b queue chunk config state=>Compilation.Binding compiler (Words.ports native)
      (Compilation.invocation params call.request b chunk config queue) state)

structure ResultSpec (rt : Runtime Memory) (native : Words.Instructions) (driver : Driver rt native)
    (convert : Scalar.Conversions Word) (compiler : Compilation.Compiler Memory)
    (call : Call) (args : Scalars (BitVec 64)) (params : Scalars Word) (before after : World Memory) : Prop where
  whole : WholeSpec Bytes.ops Bytes.sign (Machine.nativeModel (Words.ports native) params) call.request before after
  values : ∀ output i,i<call.request.count →
    logical Bytes.ops Bytes.sign after.memory (outputView call.request output) i=target native params call before output i
  bytes : ∀ output i,i<call.request.count → ∀ byte : Fin 4,
    after.memory ((outputView call.request output).key i).owner (((outputView call.request output).key i).address+byte.val)=
      Bytes.unpack (Bytes.sign.encode (outputView call.request output).negative (target native params call before output i)) byte
  gradient : ∀ i,i<call.request.grad.numel →
    logical Bytes.ops Bytes.sign after.memory call.request.grad i=logical Bytes.ops Bytes.sign before.memory call.request.grad i
  surplus : ∀ i,call.request.count≤ i → i<call.request.momentum.numel →
    logical Bytes.ops Bytes.sign after.memory call.request.momentum i=logical Bytes.ops Bytes.sign before.memory call.request.momentum i
  execution : Execution rt native driver convert compiler call args params before after

/-- Exact checked in-place FP32 call, raw words and bytes, caller complement,
unchanged gradient/surplus, and every actual source/restore/compiler invocation.
No finiteness or coefficient-range premise occurs in this main theorem. -/
theorem lion_host_fp32 (rt : Runtime Memory) (lib : Library rt) (native : Words.Instructions)
    (driver : Driver rt native) (convert : Scalar.Conversions Word) (compiler : Compilation.Compiler Memory)
    (call : Call) (h : Admitted call) (tensor : Bindings call)
    (args : Scalars (BitVec 64)) (params : Scalars Word)
    (normalized : call.request.count≠0 → Scalar.normalize convert call.arguments=.ok args)
    (binding : call.request.count≠0 → Scalar.bind convert args=.ok params)
    (compilation : ∀ b queue,Compilation.Local compiler (Words.ports native) params call.request b queue)
    (before : World Memory) :
    Ensures (fun _ after=>ResultSpec rt native driver convert compiler call args params before after)
      (run rt driver convert call before) := by
  cases actual : run rt driver convert call before with
  | error => trivial
  | ok unit after =>
    cases unit
    have whole:=run_spec Bytes.ops Bytes.sign (Machine.nativeModel (Words.ports native)) rt lib driver convert
      call h tensor args params normalized binding before after actual
    have values : ∀ output i,i<call.request.count →
        logical Bytes.ops Bytes.sign after.memory (outputView call.request output) i=target native params call before output i := by
      intro output i active
      simpa only [Machine.native_scalar,target] using whole.values output i active
    refine ⟨whole,values,?_,?_,?_,Host.execution Bytes.ops (Machine.nativeModel (Words.ports native)) rt lib driver convert
      call h tensor args params normalized binding before after actual _
      (fun b queue chunk _ config state=>(compilation b queue).resolve chunk config state)⟩
    · intro output i active byte
      have physical:=congrArg (Bytes.sign.encode (outputView call.request output).negative) (values output i active)
      simp only [logical,Negation.encode_twice] at physical
      rw [Bytes.byte_of_read after.memory ((outputView call.request output).key i) byte]
      exact congrArg (fun word=>Bytes.unpack word byte) physical
    · intro i active
      exact read_only Bytes.ops Bytes.sign Observation.bytes_read (Machine.nativeModel (Words.ports native)) rt lib driver convert
        call h tensor args params normalized binding before after actual whole .third i active (Or.inl rfl)
    · intro i lowerBound upperBound
      exact read_only Bytes.ops Bytes.sign Observation.bytes_read (Machine.nativeModel (Words.ports native)) rt lib driver convert
        call h tensor args params normalized binding before after actual whole .b i upperBound (Or.inr lowerBound)

/-- The historical optional software bridge requires finite operands at every
actual basic-operation site, including intermediates; the main result does not. -/
def NumericDomain (native : Words.Instructions) (params : Scalars Word) (call : Call) (before : World Memory) : Prop :=
  ∀ i,i<call.request.count → Words.NumericDomain native params
    (logical Bytes.ops Bytes.sign before.memory call.request.p i)
    (logical Bytes.ops Bytes.sign before.memory call.request.grad i)
    (logical Bytes.ops Bytes.sign before.memory call.request.momentum i)

def softwareTarget (params : Scalars Word) (call : Call) (before : World Memory) (output : Output) (i : Nat) : Word :=
  Words.reference params output (logical Bytes.ops Bytes.sign before.memory call.request.p i)
    (logical Bytes.ops Bytes.sign before.memory call.request.grad i)
    (logical Bytes.ops Bytes.sign before.memory call.request.momentum i)

theorem software_readback (rt : Runtime Memory) (native : Words.Instructions) (driver : Driver rt native)
    (convert : Scalar.Conversions Word) (compiler : Compilation.Compiler Memory)
    (call : Call) (args : Scalars (BitVec 64)) (params : Scalars Word) (before after : World Memory)
    (result : ResultSpec rt native driver convert compiler call args params before after)
    (profile : Compilation.Profile) (laws : Words.LocalExact profile native)
    (domain : NumericDomain native params call before) (output : Output) (i : Nat) (active : i<call.request.count) :
    logical Bytes.ops Bytes.sign after.memory (outputView call.request output) i=softwareTarget params call before output i := by
  rw [result.values output i active]
  exact Words.reference_exact profile native laws params _ _ _ (domain i active) output

#specNonCircular target avoiding [Native.interpret,Native.sourceScalar,Tile.scalar,Tile.sourceRun,Machine.scalar,Machine.run,run,pipeline]
#specNonCircular softwareTarget avoiding [Native.interpret,Native.sourceScalar,Tile.scalar,Tile.sourceRun,Machine.scalar,Machine.run,run,pipeline]
#axiomsClean lion_host_fp32
#axiomsClean software_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.HostFPEquiv
