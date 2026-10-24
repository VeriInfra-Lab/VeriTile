import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.KernelNativeResults

/-! Optional Tensor checks and binding consequences. -/
namespace VeriTile.Bench.Repairs.L2NormBwd.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

theorem python_checks_iff_preconditions (x dy : ArgumentMetadata) :
    checkInputs x dy=.ok () ↔ AdmittedArguments x dy := checkInputs_iff x dy

/-- Native active addresses use the Tensor-bound pointers, have four backed
bytes, and meet scalar FP32 alignment. No restriction on inactive pointers or
on the unused suffix of DY is added. -/
theorem kernel_address_safety (x dy : TensorMetadata) (h : InputContract x dy)
    (input gradient : View) (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (launch : Kernel.LaunchCompatible p) (row col : Nat) (hr : row<p.x.rows) (hc : col<p.x.cols) :
    (Kernel.AccessSafe p.x row col ∧ Kernel.AccessSafe p.dy row col ∧ Kernel.AccessSafe p.dx row col) ∧
    (p.x.key row col).address%4=0 ∧ (p.dy.key row col).address%4=0 ∧ (p.dx.key row col).address%4=0 :=
  ⟨Kernel.access_safe x dy input gradient p binding launch row col hr hc,
    Kernel.access_alignment x dy h input gradient p binding row col hr hc⟩

#axiomsClean python_checks_iff_preconditions
#axiomsClean kernel_address_safety
end
end VeriTile.Bench.Repairs.L2NormBwd.TensorBridge

namespace VeriTile.Bench.Repairs.L2NormBwd.TensorBridge.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

/-- Successful complete source execution, the independent derivative on every
active lane, and every exterior logical-memory cell. -/
theorem kernel_correctness {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (_h : InputContract x dy) (input gradient : View)
    (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : ℝ) (before : M) (row : Nat) (hr : row<p.x.rows) :
    Spec.Real (∃ after, FP.Structural.exec Math.algebra (source p.strides p.x.cols p.block)
      (Tile.rowState row eps (Bridge.sourceState ops id p before))=some after ∧
      (∀ col, col<p.x.cols → (after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real =
        Kernel.formula ops input gradient p.x.cols eps before row col) ∧
      Kernel.RowFrame p row (Bridge.sourceState ops id p before) after)  := by
  obtain ⟨after,execution,values,frame⟩ := VeriTile.Bench.Repairs.L2NormBwd.KernelRealResults.kernel_correctness_raw
    ops (RawKernel.Layout.ofPhysical p) (RawKernel.Layout.ofPhysical_valid p binding.valid) eps before row hr
  dsimp only [RawKernel.Layout.physical,RawKernel.Layout.ofPhysical,RawKernel.Matrix.physical,RawKernel.Matrix.ofPhysical] at values
  refine ⟨after,execution,?_,frame⟩
  intro col hc
  rw [values col hc]
  exact (Math.source_derivative p.x.cols p.block binding.valid.block eps _ _ col).symm.trans
    (Kernel.real_value ops x dy input gradient p binding eps before row col hr hc)

/-- Every row executes from its preceding physical memory; all DX values and
all bytes outside DX follow, including empty grids. -/
theorem grid_correctness {M β : Type} (ops : Ops M ℝ β)
    (x dy : TensorMetadata) (_h : InputContract x dy) (input gradient : View)
    (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : ℝ) (before : M) :
    Spec.Real (Kernel.Certificates ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops Math.algebra id (fun _ => id) p eps (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        Kernel.formula ops input gradient p.x.cols eps before row col) ∧ Kernel.ByteFrame ops p before after)  := by
  obtain ⟨certificates,values,frame⟩ := VeriTile.Bench.Repairs.L2NormBwd.KernelRealResults.grid_correctness_raw
    ops (RawKernel.Layout.ofPhysical p) (RawKernel.Layout.ofPhysical_valid p binding.valid) eps before
  simp only [RawKernel.sourceRows_ofPhysical,RawKernel.certificates_ofPhysical] at certificates values frame
  dsimp only [RawKernel.Layout.physical,RawKernel.Layout.ofPhysical,RawKernel.Matrix.physical,RawKernel.Matrix.ofPhysical] at values
  refine ⟨certificates,?_,frame⟩
  intro row col hr hc
  exact (values row col hr hc).trans ((Math.source_derivative p.x.cols p.block binding.valid.block eps _ _ col).symm.trans
    (Kernel.real_value ops x dy input gradient p binding eps before row col hr hc))

#axiomsClean kernel_correctness
#axiomsClean grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.TensorBridge.Correct

namespace VeriTile.Bench.Repairs.L2NormBwd.TensorBridge.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section

/-- The actual fourteen-statement source succeeds and returns the independently
written native graph on every active lane. All untouched cells are preserved.
The domain includes every raw FP32 input, gradient and epsilon word. -/
theorem kernel_native {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (x dy : TensorMetadata) (_h : InputContract x dy) (input gradient : View)
    (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : Native.Word) (before : M) (row : Nat) (hr : row<p.x.rows) :
    ∃ after, FP.Structural.exec (Native.algebra instructions p.block) (source p.strides p.x.cols p.block)
      (Tile.rowState row (Native.atom eps) (Bridge.sourceState ops Native.atom p before))=some after ∧
      (∀ col, col<p.x.cols → ((after.mem "DX" (row*p.dx.rowStride+col*p.dx.colStride)).read .real).value (col%128) =
        Kernel.nativeFormula ops instructions input gradient p.x.cols p.block eps before row col) ∧
      Kernel.RowFrame p row (Bridge.sourceState ops Native.atom p before) after  := by
  obtain ⟨after,execution,values,frame⟩ := VeriTile.Bench.Repairs.L2NormBwd.KernelNativeResults.kernel_native_raw
    ops instructions (RawKernel.Layout.ofPhysical p) (RawKernel.Layout.ofPhysical_valid p binding.valid) eps before row hr
  dsimp only [RawKernel.Layout.physical,RawKernel.Layout.ofPhysical,RawKernel.Matrix.physical,RawKernel.Matrix.ofPhysical] at values
  refine ⟨after,execution,?_,frame⟩
  intro col hc
  rw [values col hc]
  exact Kernel.native_value ops instructions x dy input gradient p binding eps before row col hr hc

/-- Complete source-grid execution, exact words and all four actual backing
bytes per output, with the full exterior byte frame. The codec is ordinary
little-endian storage semantics; no output correspondence is assumed. -/
theorem grid_native {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (instructions : Native.Instructions) (x dy : TensorMetadata) (_h : InputContract x dy)
    (input gradient : View) (p : Physical.Layout) (binding : Kernel.Buffers x dy input gradient p)
    (eps : Native.Word) (before : M) :
    Kernel.Certificates ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
      p (Native.atom eps) (List.range p.x.rows) before ∧
      let after := Bridge.sourceRows ops (Native.algebra instructions p.block) Native.atom (fun c n => n.value (c%128))
        p (Native.atom eps) (List.range p.x.rows) before
      (∀ row col, row<p.x.rows → col<p.x.cols → ops.read after (p.dx.key row col) =
        Kernel.nativeFormula ops instructions input gradient p.x.cols p.block eps before row col ∧
        ∀ byte, byte<4 → ops.observe after (p.dx.key row col).owner ((p.dx.key row col).address+byte) =
          (Kernel.nativeFormula ops instructions input gradient p.x.cols p.block eps before row col).extractLsb' (8*byte) 8) ∧
      Kernel.ByteFrame ops p before after  := by
  obtain ⟨certificates,values,frame⟩ := VeriTile.Bench.Repairs.L2NormBwd.KernelNativeResults.grid_native_raw
    ops codec instructions (RawKernel.Layout.ofPhysical p) (RawKernel.Layout.ofPhysical_valid p binding.valid) eps before
  simp only [RawKernel.sourceRows_ofPhysical,RawKernel.certificates_ofPhysical] at certificates values frame
  dsimp only [RawKernel.Layout.physical,RawKernel.Layout.ofPhysical,RawKernel.Matrix.physical,RawKernel.Matrix.ofPhysical] at values
  refine ⟨certificates,?_,frame⟩
  intro row col hr hc
  have target := Kernel.native_value ops instructions x dy input gradient p binding eps before row col hr hc
  have result := values row col hr hc
  exact ⟨result.1.trans target,fun byte hb => (result.2 byte hb).trans
    (congrArg (fun w : Native.Word => w.extractLsb' (8*byte) 8) target)⟩

#axiomsClean kernel_native
#axiomsClean grid_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.TensorBridge.FPEquiv
