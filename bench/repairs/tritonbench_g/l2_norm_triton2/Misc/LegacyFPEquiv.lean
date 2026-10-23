import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Binding
import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Compilation
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Forward

/- Selected production storage/input/gradient/output and accumulators are FP32.
All raw words and runtime epsilon words remain admitted. Backward's exact native
graph retains both reductions, operand-order branches, bit sign negation and
final FMA. Ordinary compiler/library boundaries are separate from the five
local primitive numerical obligations. Only extreme global-grid/PID/segmentation
compatibility is a separately recorded runtime deferral. No GPU conformance. -/

namespace VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv.Backward
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation Codec logical_bytes)
open Host Native
open L2NormTriton1.Host (World Result Keep)
noncomputable section
open Classical

/-- Independent FP32 native graph: exact masked zeros, runtime epsilon,
register/FMA and shuffle reduction order, sqrt.approx.ftz.f32, div.full.f32,
ordered scale multiplications, exact sign-bit negation and final FMA.
Both input/gradient and epsilon words, including NaN payloads, remain. -/
def target {M : Type} (ops : Ops M Word Byte) (instructions : Instructions)
    (call : Call Word) (entry : World M) (row col : Nat) : Word :=
  reference ops instructions call entry row col

structure ResultSpec {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (compiler : Compiler M) (instructions : Instructions) (call : Call Word)
    (entry : World M) (out : View) (after : World M) : Prop where
  whole : PublicFacts ops signNegation (callAdapter instructions call) rt (callNode call) entry out after
  values : ∀ row col,row*(call.cols)+col<call.input.numel →
    col<call.cols →
    logical ops signNegation after.memory out (row*(call.cols)+col)=
      target ops instructions call entry row col
  bytes : ∀ row col,row*(call.cols)+col<call.input.numel →
    col<call.cols → ∀ n,n<4 →
    let key := out.key (row*(call.cols)+col)
    ops.observe after.memory key.owner (key.address+n)=
      (target ops instructions call entry row col).extractLsb' (8*n) 8
  execution : ExecutionEvidence ops signNegation (callAdapter instructions call) rt (callNode call) entry out after
  compilation : CompilationEvidence ops rt compiler instructions call entry out after

theorem source_result {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions : Instructions) (compiled : LocalCompilation rt compiler instructions)
    (call : Call Word) (entry : World M) (domain : Domain (callNode call) entry)
    (out : View) (after : World M)
    (success : execute ops (callAdapter instructions call) rt (callNode call) entry=.ok out after) :
    ResultSpec ops rt compiler instructions call entry out after := by
  have facts := execute_success ops signNegation (callAdapter instructions call) rt lib (callNode call)
    entry domain out after success
  have values : ∀ row col,row*(call.cols)+col<call.input.numel →
      col<call.cols →
      logical ops signNegation after.memory out (row*(call.cols)+col)=
        target ops instructions call entry row col := by
    intro row col active hc
    exact (facts.values row col active hc).trans (expected_native ops instructions call entry row col hc)
  refine ⟨facts,values,?_,execution_evidence ops signNegation (callAdapter instructions call) rt lib
    (callNode call) entry domain out after facts.trace,
    compilation_evidence ops rt compiler instructions compiled lib call entry out after domain facts.trace⟩
  intro row col active hc n small
  have bytes := logical_bytes ops codec after.memory out _ _ (values row col active hc) n small
  simpa only [facts.normal,Negation.encode,Bool.false_eq_true,if_false] using bytes

/-- Exactly five optional local numerical-port equalities. This bridge never
assumes a whole reduction/output equality, execution success, address bound,
reassociation, commutativity, or equivalence of FMA with split arithmetic. -/
theorem source_bridge {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions referenceInstructions : Instructions) (compiled : LocalCompilation rt compiler instructions)
    (numerics : InstructionAgreement instructions referenceInstructions)
    (call : Call Word) (entry : World M) (domain : Domain (callNode call) entry)
    (out : View) (after : World M)
    (success : execute ops (callAdapter instructions call) rt (callNode call) entry=.ok out after) :
    ResultSpec ops rt compiler instructions call entry out after ∧
      ∀ row col,row*(call.cols)+col<call.input.numel →
        col<call.cols →
        logical ops signNegation after.memory out (row*(call.cols)+col)=
          target ops referenceInstructions call entry row col := by
  have facts := source_result ops codec rt lib compiler instructions compiled call entry domain out after success
  refine ⟨facts,?_⟩
  intro row col active hc
  rw [facts.values row col active hc]
  simp only [target,reference]
  exact row_agreement instructions referenceInstructions numerics _ _ _ _ _ _ _

theorem source_frame {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (lib : Library ops signNegation rt) (instructions : Instructions) (call : Call Word)
    (entry : World M) (domain : Domain (callNode call) entry) :
    Keep ops entry (execute ops (callAdapter instructions call) rt (callNode call) entry).state :=
  execute_frame ops signNegation (callAdapter instructions call) rt lib (callNode call) entry domain

/-- Checked backward wrapper with the accepted native graph, both direct
input views, original X return shape, readback bytes and all caller-byte frames. -/
theorem l2_host_fp32 {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions : Instructions) (compiled : LocalCompilation rt compiler instructions)
    (x dy : TensorMetadata) (input gradient : View) (eps : Word) (entry : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : input.storage∈entry.allocations) (ld : gradient.storage∈entry.allocations)
    (out : View) (after : World M)
    (success : checkedExecute ops (callAdapter instructions (checkedCall x input gradient eps)) rt
      x dy input gradient (atom eps) entry=.ok (.ok out after)) :
    ResultSpec ops rt compiler instructions (checkedCall x input gradient eps) entry out after := by
  rw [checked_execute _ _ _ _ _ _ _ _ _ h] at success
  exact source_result ops codec rt lib compiler instructions compiled (checkedCall x input gradient eps) entry
    (checked_domain x dy input gradient (atom eps) entry h bx bd lx ld) out after (Except.ok.inj success)

/-- The historical five local numerical ports are unchanged. -/
theorem l2_host_instruction_bridge {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions referenceInstructions : Instructions) (compiled : LocalCompilation rt compiler instructions)
    (numerics : InstructionAgreement instructions referenceInstructions)
    (x dy : TensorMetadata) (input gradient : View) (eps : Word) (entry : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : input.storage∈entry.allocations) (ld : gradient.storage∈entry.allocations)
    (out : View) (after : World M)
    (success : checkedExecute ops (callAdapter instructions (checkedCall x input gradient eps)) rt
      x dy input gradient (atom eps) entry=.ok (.ok out after)) :
    ResultSpec ops rt compiler instructions (checkedCall x input gradient eps) entry out after ∧
      ∀ row col,row*(gradient.shape.getLastD 0)+col< input.numel → col< gradient.shape.getLastD 0 →
        logical ops signNegation after.memory out (row*(gradient.shape.getLastD 0)+col)=
          target ops referenceInstructions (checkedCall x input gradient eps) entry row col := by
  rw [checked_execute _ _ _ _ _ _ _ _ _ h] at success
  exact source_bridge ops codec rt lib compiler instructions referenceInstructions compiled numerics
    (checkedCall x input gradient eps) entry
    (checked_domain x dy input gradient (atom eps) entry h bx bd lx ld) out after (Except.ok.inj success)

theorem l2_fp32_all_outcomes {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (lib : Library ops signNegation rt) (instructions : Instructions)
    (x dy : TensorMetadata) (input gradient : View) (eps : Word) (entry : World M)
    (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : input.storage∈entry.allocations) (ld : gradient.storage∈entry.allocations) :
    match checkedExecute ops (callAdapter instructions (checkedCall x input gradient eps)) rt
      x dy input gradient (atom eps) entry with
    | .error _ => True
    | .ok result => Keep ops entry result.state := by
  rw [checked_execute _ _ _ _ _ _ _ _ _ h]
  exact source_frame ops rt lib instructions (checkedCall x input gradient eps) entry
    (checked_domain x dy input gradient (atom eps) entry h bx bd lx ld)

#axiomsClean l2_host_fp32
#axiomsClean l2_host_instruction_bridge
#axiomsClean l2_fp32_all_outcomes

-- These are constructive execution/readback theorems, not FP.Equiv rewrite
-- statements accepted by #print_fp_assumptions. Numerical deferrals are exactly
-- Native.InstructionAgreement; ordinary compiler contracts are separately typed.
#specNonCircular target avoiding [Tile.scalar,Tile.sourceRun,Host.execute,Physical.run,Host.expected,algebra]
#axiomsClean source_result
#axiomsClean source_bridge
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv.Backward

namespace VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv.Forward
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation Codec logical_bytes)
open L2NormTriton1.Host L2NormTriton1.Native
noncomputable section
open Classical

/-- Independent FP32 native graph: exact masked zeros, runtime epsilon,
register/FMA and shuffle reduction order, sqrt.approx.ftz.f32, div.full.f32,
and final multiply. All input/epsilon words, including NaN payloads, remain. -/
def target {M : Type} (ops : Ops M Word Byte) (instructions : Instructions)
    (call : Call Word) (entry : World M) (row col : Nat) : Word :=
  reference ops instructions call entry row col

structure ResultSpec {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (compiler : Compiler M) (instructions : Instructions) (call : Call Word)
    (entry : World M) (out : View) (after : World M) : Prop where
  whole : PublicFacts ops signNegation (callAdapter instructions call) rt (callNode call) entry out after
  values : ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
    col<call.input.shape.getLastD 0 →
    logical ops signNegation after.memory out (row*(call.input.shape.getLastD 0)+col)=
      target ops instructions call entry row col
  bytes : ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
    col<call.input.shape.getLastD 0 → ∀ n,n<4 →
    let key := out.key (row*(call.input.shape.getLastD 0)+col)
    ops.observe after.memory key.owner (key.address+n)=
      (target ops instructions call entry row col).extractLsb' (8*n) 8
  execution : ExecutionEvidence ops signNegation (callAdapter instructions call) rt (callNode call) entry out after
  compilation : L2NormTriton2.Forward.CompilationEvidence ops rt compiler instructions call entry out after

theorem source_result {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions : Instructions) (compiled : L2NormTriton2.Forward.LocalCompilation rt compiler instructions)
    (call : Call Word) (entry : World M) (domain : Domain (callNode call) entry)
    (out : View) (after : World M)
    (success : execute ops (callAdapter instructions call) rt (callNode call) entry=.ok out after) :
    ResultSpec ops rt compiler instructions call entry out after := by
  have facts := execute_success ops signNegation (callAdapter instructions call) rt lib (callNode call)
    entry domain out after success
  have values : ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
      col<call.input.shape.getLastD 0 →
      logical ops signNegation after.memory out (row*(call.input.shape.getLastD 0)+col)=
        target ops instructions call entry row col := by
    intro row col active hc
    exact (facts.values row col active hc).trans (expected_native ops instructions call entry row col hc)
  refine ⟨facts,values,?_,execution_evidence ops signNegation (callAdapter instructions call) rt lib
    (callNode call) entry domain out after facts.trace,
    L2NormTriton2.Forward.compilation_evidence ops rt compiler instructions compiled lib call entry out after domain facts.trace⟩
  intro row col active hc n small
  have bytes := logical_bytes ops codec after.memory out _ _ (values row col active hc) n small
  simpa only [facts.normal,Negation.encode,Bool.false_eq_true,if_false] using bytes

/-- Exactly five optional local numerical-port equalities. This bridge never
assumes a whole reduction/output equality, execution success, address bound,
reassociation, commutativity, or equivalence of FMA with split arithmetic. -/
theorem source_bridge {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions referenceInstructions : Instructions) (compiled : L2NormTriton2.Forward.LocalCompilation rt compiler instructions)
    (numerics : InstructionAgreement instructions referenceInstructions)
    (call : Call Word) (entry : World M) (domain : Domain (callNode call) entry)
    (out : View) (after : World M)
    (success : execute ops (callAdapter instructions call) rt (callNode call) entry=.ok out after) :
    ResultSpec ops rt compiler instructions call entry out after ∧
      ∀ row col,row*(call.input.shape.getLastD 0)+col<call.input.numel →
        col<call.input.shape.getLastD 0 →
        logical ops signNegation after.memory out (row*(call.input.shape.getLastD 0)+col)=
          target ops referenceInstructions call entry row col := by
  have facts := source_result ops codec rt lib compiler instructions compiled call entry domain out after success
  refine ⟨facts,?_⟩
  intro row col active hc
  rw [facts.values row col active hc]
  simp only [target,reference]
  exact row_agreement instructions referenceInstructions numerics _ _ _ _ _ _

theorem source_frame {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (lib : Library ops signNegation rt) (instructions : Instructions) (call : Call Word)
    (entry : World M) (domain : Domain (callNode call) entry) :
    Keep ops entry (execute ops (callAdapter instructions call) rt (callNode call) entry).state :=
  execute_frame ops signNegation (callAdapter instructions call) rt lib (callNode call) entry domain

/-- Checked direct-input wrapper with the accepted native reduction graph,
raw FP32 words, returned shape/object, source execution and complete byte frame. -/
theorem l2_host_fp32 {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions : Instructions) (compiled : L2NormTriton2.Forward.LocalCompilation rt compiler instructions)
    (metadata : TensorMetadata) (input : View) (eps : Word) (entry : World M)
    (h : ForwardContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈entry.allocations) (out : View) (after : World M)
    (success : checkedExecute ops (callAdapter instructions (checkedCall metadata input eps)) rt
      metadata input (atom eps) entry=.ok (.ok out after)) :
    ResultSpec ops rt compiler instructions (checkedCall metadata input eps) entry out after := by
  rw [checked_execute _ _ _ _ _ _ _ h] at success
  exact source_result ops codec rt lib compiler instructions compiled (checkedCall metadata input eps) entry
    (checked_domain metadata input (atom eps) entry h binding live) out after (Except.ok.inj success)

/-- Exactly the original five local instruction ports; no hidden finiteness,
reassociation, FMA splitting or whole-output numerical hypothesis is introduced. -/
theorem l2_host_instruction_bridge {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (rt : Runtime M) (lib : Library ops signNegation rt) (compiler : Compiler M)
    (instructions referenceInstructions : Instructions) (compiled : L2NormTriton2.Forward.LocalCompilation rt compiler instructions)
    (numerics : InstructionAgreement instructions referenceInstructions)
    (metadata : TensorMetadata) (input : View) (eps : Word) (entry : World M)
    (h : ForwardContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈entry.allocations) (out : View) (after : World M)
    (success : checkedExecute ops (callAdapter instructions (checkedCall metadata input eps)) rt
      metadata input (atom eps) entry=.ok (.ok out after)) :
    ResultSpec ops rt compiler instructions (checkedCall metadata input eps) entry out after ∧
      ∀ row col,row*(input.shape.getLastD 0)+col< input.numel → col< input.shape.getLastD 0 →
        logical ops signNegation after.memory out (row*(input.shape.getLastD 0)+col)=
          target ops referenceInstructions (checkedCall metadata input eps) entry row col := by
  rw [checked_execute _ _ _ _ _ _ _ h] at success
  exact source_bridge ops codec rt lib compiler instructions referenceInstructions compiled numerics
    (checkedCall metadata input eps) entry (checked_domain metadata input (atom eps) entry h binding live)
    out after (Except.ok.inj success)

theorem l2_fp32_all_outcomes {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (lib : Library ops signNegation rt) (instructions : Instructions)
    (metadata : TensorMetadata) (input : View) (eps : Word) (entry : World M)
    (h : ForwardContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈entry.allocations) :
    match checkedExecute ops (callAdapter instructions (checkedCall metadata input eps)) rt
      metadata input (atom eps) entry with
    | .error _ => True
    | .ok result => Keep ops entry result.state := by
  rw [checked_execute _ _ _ _ _ _ _ h]
  exact source_frame ops rt lib instructions (checkedCall metadata input eps) entry
    (checked_domain metadata input (atom eps) entry h binding live)

#axiomsClean l2_host_fp32
#axiomsClean l2_host_instruction_bridge
#axiomsClean l2_fp32_all_outcomes

-- These are constructive execution/readback theorems, not FP.Equiv rewrite
-- statements accepted by #print_fp_assumptions. Numerical deferrals are exactly
-- Native.InstructionAgreement; ordinary compiler contracts are separately typed.
#specNonCircular target avoiding [L2NormTriton1.Tile.scalar,L2NormTriton1.Tile.sourceRun,L2NormTriton1.Host.execute,L2NormTriton1.Physical.run,L2NormTriton1.Host.expected,algebra]
#axiomsClean source_result
#axiomsClean source_bridge
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv.Forward
