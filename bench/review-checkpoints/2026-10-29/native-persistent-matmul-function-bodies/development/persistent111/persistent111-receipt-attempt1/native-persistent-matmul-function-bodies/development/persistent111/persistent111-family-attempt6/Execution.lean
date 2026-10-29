import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Prelude
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Outer

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open GridExecution (Outside Post atProgram)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def outerStatement (p : Parameters) : ComputeStmt :=
  (source p).surfaceBody[15]'(by cases hp : p.problem.profile <;> simp [source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,hp,ComputeKernel.surfaceBody])

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody=prelude p++[outerStatement p] := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile <;> rfl

theorem dynamicOuter (P : Source.Arithmetic α α) (p : Parameters)
    (s : State α) (pid : Nat) (regs : Registers p pid s) :
    step (backend P) (outerStatement p) s=
      FP.Structural.rangeInt (backend P) "tile_ordinal" 0 (p.problem.work pid) 1 (outer p) s := by
  cases hp : p.problem.profile <;>
    simp [outerStatement,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,hp,ComputeKernel.surfaceBody,step,
      evalOp_unfold,evalExpr,evalComputeOp,Option.bind,regs.work]

def program (P : Source.Arithmetic α α) (p : Parameters) (pid : Nat) (s : State α) : Option (State α) :=
  FP.Structural.exec (backend P) (source p) (atProgram s pid)

def ProgramOwns (p : Parameters) (pid : Nat) (x : LogicalIndex p) : Prop :=
  ∃ ordinal<p.problem.work pid,Owns p (p.problem.tile pid ordinal) x

theorem programExecution (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (initial : State α) (entry : Memory α) (pid : Nat)
    (layout : Layout p.problem b) (strides : Strides p b) (inputs : Inputs p b initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero)
    (sms : 0<p.problem.SMS) (pidBound : pid<p.problem.grid) :
    ∃ final,program P p pid initial=some final ∧
      Post "C" (logicalAddress p) (referenceValue P p b entry) (ProgramOwns p pid) initial final := by
  let prepared := preludeExecution P p (atProgram initial pid)
  have regs : Registers p pid prepared.val := by
    simpa [atProgram] using (prelude_invariant P p (atProgram initial pid)).1
  have memory : prepared.val.mem=initial.mem := (prelude_invariant P p (atProgram initial pid)).2
  have ip : Inputs p b prepared.val entry := by
    constructor
    · intro i; rw [memory]; exact inputs.a i
    · intro i; rw [memory]; exact inputs.b i
  have single : ∀ ordinal,ordinal<p.problem.work pid → ∀ current,Outside "C" prepared.val current →
      Registers p pid current → ∃ final,
        Compose.ordinal (backend P) "tile_ordinal" (outer p) ordinal current=some final ∧
        Post "C" (logicalAddress p) (referenceValue P p b entry) (Owns p (p.problem.tile pid ordinal)) current final ∧
        Registers p pid final := by
    intro ordinal bound current outside ready
    have tileBound : p.problem.tile pid ordinal<p.problem.tiles :=
      (Source.tiles_exact p.problem sms _).mp ((Source.tile_mem p.problem _).mpr ⟨pid,pidBound,ordinal,bound,rfl⟩)
    exact outerExecution P p b current entry pid ordinal layout strides
      (inputs_outside p b prepared.val current entry ip outside) zero ready tileBound
  obtain ⟨final,ran,post,ready⟩ := Compose.ready "C" (logicalAddress p)
    (logical_injective p b layout strides) (referenceValue P p b entry)
    (fun ordinal => Owns p (p.problem.tile pid ordinal))
    (Compose.ordinal (backend P) "tile_ordinal" (outer p)) (Registers p pid)
    prepared.val (p.problem.work pid) single (List.range' 0 (p.problem.work pid))
    (by intro ordinal h; simpa [List.mem_range'] using h) prepared.val
    (by intro r o h; rfl) regs
  refine ⟨final,?_,?_⟩
  · unfold program
    change run (backend P) (source p).surfaceBody (atProgram initial pid)=some final
    rw [source_decomposition,run_append,prepared.property,Option.bind_some,run,
      dynamicOuter P p prepared.val pid regs,Compose.range_sequence]
    simp only [Nat.sub_zero,ran,run]
    rfl
  · simpa [Post,ProgramOwns,List.mem_range',memory] using post

def executeGrid (P : Source.Arithmetic α α) (p : Parameters) (initial : State α) : Option (State α) :=
  GridExecution.sequence (program P p) (List.range p.problem.grid) initial

/-- Full readable quote execution, composed over the actual persistent grid.
The numerical target contains neither program identifiers nor executions. -/
theorem gridExecution (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (initial : State α) (entry : Memory α)
    (layout : Layout p.problem b) (strides : Strides p b) (inputs : Inputs p b initial entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero) (sms : 0<p.problem.SMS) :
    ∃ final,executeGrid P p initial=some final ∧
      Post "C" (logicalAddress p) (referenceValue P p b entry) (fun _ => True) initial final := by
  obtain ⟨final,ran,post⟩ := GridExecution.compose "C" (logicalAddress p)
    (logical_injective p b layout strides) (referenceValue P p b entry) (ProgramOwns p)
    (program P p) initial p.problem.grid
    (fun pid hp current outside => programExecution P p b current entry pid layout strides
      (inputs_outside p b initial current entry inputs outside) zero sms hp)
    (List.range p.problem.grid) (by intro pid hp; exact List.mem_range.mp hp) initial (by intro r o h; rfl)
  refine ⟨final,ran,?_,?_⟩
  · intro x hx
    obtain ⟨pid,hp,ordinal,ho,owns⟩ := covers p sms x
    exact post.1 x ⟨pid,List.mem_range.mpr hp,ordinal,ho,owns⟩
  · intro r o miss
    exact post.2 r o (miss.imp id (fun h x _ => h x trivial))

#axiomsClean programExecution
#axiomsClean gridExecution
#specNonCircular Targets.result avoiding [program,executeGrid]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
