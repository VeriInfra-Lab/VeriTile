import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Interface
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.NativeExecution

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.NativeExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
open Interface
noncomputable section

theorem io_equiv (native : NativeWords.Instructions) (params : Scalars Native.Word)
    (total n B pid : Nat) :
    (programIO (Native.semantics (NativeWords.ports native)) (Native.arguments params)
      Term.input (Native.interpret (NativeWords.ports native)) total n B pid).Implements
      (fun values => result
        (fun j => Native.reference (NativeWords.ports native) params .parameter
          (parameter values j) (gradient values j) (momentum values j))
        (fun j => Native.reference (NativeWords.ports native) params .momentum
          (parameter values j) (gradient values j) (momentum values j))) := by
  refine Interface.implements (Native.semantics (NativeWords.ports native))
    (Native.arguments params) Term.input (Native.interpret (NativeWords.ports native))
    total n B pid (fun values output j => Native.reference (NativeWords.ports native) params output
      (parameter values j) (gradient values j) (momentum values j)) ?_
  intro values s hp hg hm
  exact kernel_inputs native params total n B pid s
    (parameter values) (gradient values) (momentum values) hp hg hm

#axiomsClean io_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.NativeExecution
