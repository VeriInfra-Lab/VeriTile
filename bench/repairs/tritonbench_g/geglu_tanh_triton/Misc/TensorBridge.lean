import bench.repairs.tritonbench_g.geglu_tanh_triton.Misc.PhysicalIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.RawCorrect
import bench.repairs.tritonbench_g.geglu_tanh_triton.Misc.RawNativeApplication
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Kernel
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

/-! Optional checked Tensor addresses, Python admission, and output-byte observations. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical DirectHost Native
noncomputable section

/-- Complete forward/backward kernel grid. Every actual selected source
executes successfully in the same physical state sequence; all logical Tensor
outputs satisfy the independent exact-decimal GELU formula/gradient, and every
exterior byte is preserved. Wrapper execution is not a proof obligation. -/
theorem geglu_kernel_correctness {M β : Type} (ops : Ops M ℝ β)
    (candidate : Bool) (p : Prepared backward) (inputs : InputAdmitted p)
    (plan : Kernel.Plan p) (entry : M) :
    VeriTile.Spec.Real (∃ final,
      Kernel.Executes ops RealExecution.algebra candidate p (input_admitted p inputs) plan entry final ∧
      Kernel.RunSpec ops RealExecution.formula p (input_admitted p inputs) entry final) := by
  refine ⟨_,Kernel.executes ops RealExecution.algebra candidate p (input_admitted p inputs) plan entry,?_⟩
  exact Kernel.run_spec ops (Tile.scalar RealExecution.algebra) RealExecution.formula
    RealExecution.scalar_real p (input_admitted p inputs) plan entry

/-- Complete raw-FP32 kernel-grid output and physical-byte frame for either
actual index variant. The exact retained native graph includes ternary FMA,
libdevice tanh branches/FTZ sites and source-encoded constants. Ordinary
instruction semantics are model boundaries, not deferred experiments. -/
theorem geglu_kernel_fp32 {M : Type} (ops : Ops M Word (BitVec 8))
    (instructions : Instructions) (candidate : Bool)
    (p : Prepared backward) (inputs : InputAdmitted p) (plan : Kernel.Plan p) (entry : M) :
    ∃ final,
      Kernel.NativeExecutes ops instructions candidate p (input_admitted p inputs) plan entry final ∧
      Kernel.RunSpec ops (compiledReference instructions) p (input_admitted p inputs) entry final := by
  refine ⟨_,Kernel.native_executes ops instructions candidate p (input_admitted p inputs) plan entry,?_⟩
  exact Kernel.run_spec ops (sourceScalar (compiledPorts instructions)) (compiledReference instructions)
    (compiled_lowering instructions) p (input_admitted p inputs) plan entry

/-- Python input admission and the public input contract agree exactly.
Forward output allocation metadata is the ordinary supplied buffer binding. -/
theorem python_checks_iff_preconditions (p : Prepared backward) :
    inputCheck p = Bool.true ↔ InputAdmitted p := inputCheck_iff p

/-- The same checked Tensor metadata fixes native signed indices/masks and
actual aligned FP32 pointers, including each Tensor's storage offset. -/
theorem checked_kernel_addresses (p : Prepared backward) (inputs : InputAdmitted p) :
    (∀ pid lane, pid < Grid.grid (p.rows*p.cols) → lane < 128 →
      (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).toInt = (pid*128+lane : Nat) ∧
      (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).slt
        (BitVec.ofNat 64 (p.rows*p.cols)) = decide (pid*128+lane < p.rows*p.cols)) ∧
    (∀ r i, i < p.rows*p.cols →
      (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat + BitVec.ofNat 64 (4*i)).toNat =
        ((p.tensor r).key i).address ∧ ((p.tensor r).key i).address % 4 = 0) :=
  ⟨indices p (input_admitted p inputs),pointer p (input_admitted p inputs)⟩

/-- The ordinary little-endian storage codec exposes every output byte of
these kernel results; no primitive-conformance assumption is added. -/
theorem geglu_kernel_output_bytes {M : Type} (ops : Ops M Word (BitVec 8))
    (codec : SwigluTriton.Words.Codec ops) (instructions : Instructions)
    (p : Prepared backward) (inputs : InputAdmitted p) (plan : Kernel.Plan p) (entry : M)
    (ch : Channel) (on : enabled backward ch) (i : Nat) (hi : i < p.rows*p.cols)
    (byte : Nat) (small : byte < 4) :
    let final := Physical.run ops (sourceScalar (compiledPorts instructions)) backward
      (layout p (input_admitted p inputs)) plan.order entry
    let key := (p.tensor (outputRole ch)).key i
    ops.observe final key.owner (key.address+byte) =
      (compiledReference instructions ch ((p.tensor .a).read ops entry i) ((p.tensor .b).read ops entry i)
        (if backward then (p.tensor .third).read ops entry i else default)).extractLsb' (8*byte) 8 := by
  dsimp only
  rw [codec.byte _ _ byte small]
  have values := (Kernel.run_spec ops _ _ (compiled_lowering instructions) p
    (input_admitted p inputs) plan entry).1 ch on i hi
  change ops.read _ _ = _ at values
  rw [values]

#axiomsClean geglu_kernel_correctness
#axiomsClean geglu_kernel_fp32
#axiomsClean python_checks_iff_preconditions
#axiomsClean checked_kernel_addresses
#axiomsClean geglu_kernel_output_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.TensorBridge
