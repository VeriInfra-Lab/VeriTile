import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Observations

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.RawNative
open VeriTile Triton FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000

/-- Independent source-order scalar recurrence over the original input words.
Reduction structure and local lowering are selected before values arrive. -/
def reference (a : Native.Instructions) (compiler : Native.Compiler) (p : Native.ReductionPlan)
    (binding : Native.ScalarBinding) (raw : PythonNumber) (c : Shape) (m : Memory (BitVec 32))
    (r : FusedRwkv6Kernel.Region) (b : Nat) : BitVec 32 :=
  Native.observeBound a compiler raw (target Native.referenceOps c (Native.plans c p)
    (inputs c (FusedRwkv6Kernel.Native.entry m)) (Native.scalar binding raw) r b)

def observed (a : Native.Instructions) (compiler : Native.Compiler) (raw : PythonNumber)
    (out : Source.Cells Native.Term) : Memory (BitVec 32) :=
  fun r b => Native.observeBound a compiler raw (Source.words out r b)

/-- Execute the shared source, then the finite four-byte stores at its kernel
output sites. No host sum, wrapper execution or supplied result relation. -/
def execute (a : Native.Instructions) (compiler : Native.Compiler) (p : Native.ReductionPlan)
    (binding : Native.ScalarBinding) (raw : PythonNumber) (c : Shape)
    (layout : Physical.Layout c) (memory : Bytes) : Option Bytes :=
  (Source.launch (Native.algebra p) c (Native.scalar binding raw)
    (Source.cells (FusedRwkv6Kernel.Native.entry (Physical.loadWords c layout memory)))).map
      (fun out => executeBytes (Physical.byteTrace c layout (observed a compiler raw out)) memory)

theorem execute_no_launch (a : Native.Instructions) (compiler : Native.Compiler)
    (p : Native.ReductionPlan) (binding : Native.ScalarBinding) (raw : PythonNumber)
    (c : Shape) (layout : Physical.Layout c) (memory : Bytes)
    (ht : c.T=0) (hf : c.final=Bool.false) :
    execute a compiler p binding raw c layout memory=some memory := by
  have nils : (programs c).flatMap (fun _ => ([] : List (FusedRwkv6Kernel.Region × Nat)))=[] :=
    List.flatMap_eq_nil_iff.mpr (by intro p hp;rfl)
  simp [execute,Source.launch,Launches,ht,hf,Physical.byteTrace,Physical.wordTrace,
    partialTrace,finalTrace,executeBytes,nils]

/-- Complete native execution on raw caller bytes and a finite output layout.
The scalar ABI, instruction meanings and compiler/reduction choices are unchanged. -/
theorem rwkv_native (a : Native.Instructions) (compiler : Native.Compiler)
    (p : Native.ReductionPlan) (binding : Native.ScalarBinding) (raw : PythonNumber)
    (c : Shape) (hc : Sizes c) (layout : Physical.Layout c) (memory : Bytes) :
    ∃ result, execute a compiler p binding raw c layout memory = some result ∧
      (∀ r b, Output c r b → Physical.loadWords c layout result r b =
        reference a compiler p binding raw c (Physical.loadWords c layout memory) r b) ∧
      (∀ r, InputRegion r → ∀ b, b<extent c r → ∀ j : Fin 4,
        result (layout.base r+4*b+j.val) = memory (layout.base r+4*b+j.val)) ∧
      (∀ addr, Physical.Outside c layout addr → result addr = memory addr) := by
  obtain ⟨out,ho,hw⟩:=Source.launch_memory (Native.algebra p) c hc (Native.plans c p)
    (Source.native_rows p c) (Native.scalar binding raw)
      (Source.cells (FusedRwkv6Kernel.Native.entry (Physical.loadWords c layout memory)))
  simp only [Source.words_cells,Native.source_ops] at hw
  let result:=Physical.scatter c layout memory (observed a compiler raw out)
  have result_word : ∀ r b,Output c r b →
      Physical.loadWords c layout result r b=
        reference a compiler p binding raw c (Physical.loadWords c layout memory) r b := by
    intro r b hb
    rw [Physical.scatter_word c layout memory _ r b hb]
    unfold observed reference
    rw [hw,if_pos hb]
  refine ⟨result,?_,result_word,Physical.input_frame c layout memory _,Physical.scatter_frame c layout memory _⟩
  simp only [execute,ho,Option.map_some,Physical.trace_realizes c hc,result]

#axiomsClean rwkv_native
#axiomsClean execute_no_launch
#axiomsClean Source.pointer_signed_live
#axiomsClean Checked.formed_address
#specNonCircular reference avoiding [execute,Source.launch,Source.runGrid,FP.Structural.exec,runLoop,runProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.RawNative
