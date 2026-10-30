import bench.repairs.tritonbench_g.token_attn_mistral.Bounds
import bench.repairs.tritonbench_g.token_attn_mistral.HostSchedule
import bench.repairs.tritonbench_g.token_attn_mistral.Integers
import bench.repairs.tritonbench_g.token_attn_mistral.Binding
namespace VeriTile.Bench.Repairs.TokenAttnMistral.Host.Historical
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option linter.unusedVariables false

/-- Raw FP32 output specification with a fixed compiler profile. The complete
128-lane trees and fusion sites are constructed, not caller numerical oracles. -/
def nativeTarget (I : Native.Instructions) (profile : Native.Profile)
    (c : Call) (m : Memory Native.Word) (b h d : Nat) (hd : d<c.value.width) :=
  Native.eval I Native.defaultFusion
    (values Native.algebra (Native.reduction profile.reductions) (params c) (Native.entry m)
      ((b,h),(⟨d,hd⟩,PUnit.unit)))

def NativeOutputs (I : Native.Instructions) (profile : Native.Profile)
    (c : Call) (m : Memory Native.Word) (result : Result Native.Term) :=
  ∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
    Native.observe I Native.defaultFusion result.memory c.output.region
      ((b.val*heads c+h.val)*c.value.width+d.val)=
    nativeTarget I profile c m b.val h.val d.val d.isLt

/-- Complete graph and full frame, without any numerical instruction premise. -/
theorem host_native_coverage (profile : Native.Profile)
    (c : Call) (m : Memory Native.Word) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c) (integers : IntegerDomain c)
    (binding : CompilerBinding c profile)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧
    (∀ e : Event (params c),e∈outputEvents c ↔ e.1.1<batches c ∧ e.1.2<heads c) ∧
    ∃ result, invokeActual Native.algebra (Native.reduction profile.reductions) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      result.memory=(outputEvents c).foldl
        (Grid.write (params c) (values Native.algebra (Native.reduction profile.reductions) (params c) (Native.entry m))) (Native.entry m) ∧
      (∀ e∈outputEvents c,result.memory c.output.region (offset (params c) e.1 e.2)=
        .mk .fp32 (values Native.algebra (Native.reduction profile.reductions) (params c) (Native.entry m) e)) ∧
      Frame c (Native.entry m) result.memory := by
  refine ⟨accesses c m layout data storage,formed c m data integers,events_exact c,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_typed Native.algebra (Native.reduction profile.reductions) c (Native.entry m) layout

/-- All logical output words, supplied output identity, None return, and full
memory frame. Native means the pinned ordered source graph; actual compiler
binding is checked separately against the recorded JIT artifacts. -/
theorem host_native (I : Native.Instructions) (profile : Native.Profile)
    (c : Call) (m : Memory Native.Word) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c) (integers : IntegerDomain c)
    (binding : CompilerBinding c profile)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Native.algebra (Native.reduction profile.reductions) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧ NativeOutputs I profile c m result ∧
      Frame c (Native.entry m) result.memory := by
  refine ⟨accesses c m layout data storage,formed c m data integers,?_⟩
  rw [invokeActual_eq _ _ _ _ _ layout legal]
  exact whole_native I profile.reductions Native.defaultFusion c m layout

/-- The only numerical bridge is pointwise agreement of three reached scalar
opcodes. No dot product, reduction, row, block or whole-result equality is
assumed. This lifts those local facts to every output word of the full call. -/
theorem fpequiv (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard) (profile : Native.Profile)
    (c : Call) (m : Memory Native.Word) (work : List PID)
    (layout : Layout c) (data : Data c m) (storage : Storage c) (integers : IntegerDomain c)
    (binding : CompilerBinding c profile)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    Accesses c m ∧ Formed c m ∧ ∃ result,
      invokeActual Native.algebra (Native.reduction profile.reductions) c work (Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output ∧
      (∀ (b : Fin (batches c)) (h : Fin (heads c)) (d : Fin c.value.width),
        Native.observe actual Native.defaultFusion result.memory c.output.region
          ((b.val*heads c+h.val)*c.value.width+d.val)=
        nativeTarget standard profile c m b.val h.val d.val d.isLt) ∧
      Frame c (Native.entry m) result.memory := by
  obtain ⟨bounds,formed,result,run,ret,out,words,frame⟩ :=
    host_native actual profile c m work layout data storage integers binding legal
  refine ⟨bounds,formed,result,run,ret,out,?_,frame⟩
  intro b h d
  rw [words b h d]
  exact Native.eval_agreement actual standard agreement _ _

#axiomsClean host_native_coverage
#axiomsClean host_native
#axiomsClean fpequiv
#auditStmt host_native_coverage
#auditStmt host_native
#auditStmt fpequiv
#specNonCircular nativeTarget avoiding [FP.Structural.exec,Grid.run,invoke,invokeActual]
end
end VeriTile.Bench.Repairs.TokenAttnMistral.Host.Historical
