import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Compilation

namespace VeriTile.Bench.Repairs.L2NormTriton2.Forward
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation)
open L2NormTriton1.Host L2NormTriton1.Native L2NormTriton1.Physical
noncomputable section
open Classical

def sourceHash : String := "b512d51cc46384b9cf4a7b61e3ec6abea98a6cc4a627917902c92bc4b156fc23"

/-- Forward AST and wrapper semantics equal the reviewed unary helper. Only
the enclosing Python file identity differs; this projection changes that field
explicitly, retaining every instruction, lowering, architecture and option. -/
def compilerView {M : Type} (compiler : Compiler M) : Compiler M :=
  {compiler with resolve := fun p s =>
    let result := compiler.resolve p s
    {result with artifact := {result.artifact with sourceHash := candidateHash}}}

structure LocalCompilation {M : Type} (rt : Runtime M) (compiler : Compiler M)
    (instructions : Instructions) : Prop where
  binding : L2NormTriton1.Host.LocalCompilation rt (compilerView compiler) instructions
  actualHash : ∀ p before plan ready,Valid p → p.block=blockSize p.cols →
    p.input.baseByte%16=0 → p.output.baseByte%16=0 →
    rt.setup p before=.ok plan ready → (compiler.resolve p before).artifact.sourceHash=sourceHash

def BoundCompilation {M : Type} (compiler : Compiler M) (instructions : Instructions)
    (p : Layout) (before : World M) : Prop :=
  L2NormTriton1.Host.BoundCompilation (compilerView compiler) instructions p before ∧
    (compiler.resolve p before).artifact.sourceHash=sourceHash

def CompilationEvidence {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (compiler : Compiler M) (instructions : Instructions) (call : Call Word)
    (entry : World M) (out : View) (after : World M) : Prop :=
  ∃ token entered q ready done,
    rt.enter call.device entry=.ok token entered ∧ prepare rt (callNode call) entered=.ok q ready ∧
    launch ops (callAdapter instructions call) rt q (atom call.eps) ready=.ok () done ∧
    rt.leave token done=.ok () after ∧ rt.reshape q.output call.input.shape=some out ∧
    (q.layout.rows=0 ∨ BoundCompilation compiler instructions q.layout ready)

theorem compilation_evidence {M : Type} (ops : Ops M Word Byte) (rt : Runtime M)
    (compiler : Compiler M) (instructions : Instructions) (contracts : LocalCompilation rt compiler instructions)
    (lib : Library ops signNegation rt) (call : Call Word) (entry : World M) (out : View) (after : World M)
    (domain : Domain (callNode call) entry)
    (trace : SuccessTrace ops (callAdapter instructions call) rt (callNode call) entry out after) :
    CompilationEvidence ops rt compiler instructions call entry out after := by
  obtain ⟨_,token,entered,q,ready,done,he,hp,hk,hl,hr⟩ := trace
  simp only [callNode] at he
  have keep : Keep ops entry entered := by simpa [he,Result.state] using lib.enter call.device entry
  have facts := prepare_sound ops signNegation rt lib (callNode call) entered
    ⟨domain.backed,keep.live _ domain.live,domain.device,domain.direct⟩ q ready hp
  refine ⟨token,entered,q,ready,done,he,hp,hk,hl,hr,?_⟩
  by_cases empty : q.layout.rows=0
  · exact Or.inl empty
  · right
    cases hs : rt.setup q.layout ready with
    | error reason s => simp [launch,empty,hs,Result.bind] at hk
    | ok plan s =>
      have nonempty : 0<q.snapshot.count := Nat.mul_pos (Nat.pos_of_ne_zero empty) facts.valid.positive
      exact ⟨contracts.binding.setup q.layout ready plan s facts.valid facts.block
        (facts.inputAligned nonempty) facts.outputAligned hs,
        contracts.actualHash q.layout ready plan s facts.valid facts.block (facts.inputAligned nonempty) facts.outputAligned hs⟩

#axiomsClean compilation_evidence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Forward
