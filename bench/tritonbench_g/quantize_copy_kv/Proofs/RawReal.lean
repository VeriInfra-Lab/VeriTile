import bench.tritonbench_g.quantize_copy_kv.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.RawReal
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Complete original and candidate launches compute the independent real
max/scale/code formula over raw input memory and preserve every exterior cell. -/
theorem kernel_correct (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (hw : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (reads : ∀ token head column, token < count → head < p.heads → column < p.width →
      (initial.mem "input" (KernelResult.inputOffset p token head column)).read .real = values token head column) :
    ∃ final,
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) p Bool.false (List.range count) initial = some final ∧
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) p Bool.true (List.range count) initial = some final ∧
      RawKernel.Result p count dest
        (fun token head column => KernelRealModel.code (values token head) p.width column)
        (fun token head => KernelRealModel.scale (values token head) p.width) initial final := by
  apply RawKernel.whole (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal
    p count dest hw covered unique initial metadata
  · intro token i ht hh
    apply KernelRealResult.code rest _ _ hw i.1 hh (values token i.1.val)
    intro c hc
    simpa [KernelGrid.atProgram] using reads token i.1.val c ht hh hc
  · intro token head ht hh
    apply KernelRealResult.scale rest _ _ hw head hh (values token head.val)
    intro c hc
    simpa [KernelGrid.atProgram] using reads token head.val c ht hh hc

#axiomsClean kernel_correct
#specNonCircular KernelRealModel.code avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.RawReal
