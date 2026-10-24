import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Evidence
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.NativeSource
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.VectorSafety

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation)
open Native Physical
noncomputable section
open Classical

inductive LoweringPolicy where
  | jitSpecializedFourWarpL2
  | other (name : String)
  deriving DecidableEq

structure Artifact where
  sourceHash : String
  backend : String
  architecture : Nat
  compiler : String
  options : List (String × String)
  signature : List String
  constexprs : List (String × Nat)
  attributes : List (String × Nat)
  block : Nat
  warps : Nat
  ctas : Nat

structure ResolvedCompilation where
  artifact : Artifact
  instructions : Instructions
  lowering : LoweringPolicy

/-- resolve denotes the compiler result used by this setup invocation. Its
local contract is an ordinary compiler/ABI boundary, not a numerical premise. -/
structure Compiler (M : Type) where
  resolve : Layout → World M → ResolvedCompilation
  architecture : Nat → Nat
  version : String
  options : List (String × String)

def candidateHash : String := "9800aebfe7402fb481670b03614f5fda6935991b5480a41d32726f76bc7973c5"

def BoundCompilation {M : Type} (compiler : Compiler M) (instructions : Instructions)
    (p : Layout) (before : World M) : Prop :=
  let result := compiler.resolve p before
  result.instructions=instructions ∧ result.lowering=.jitSpecializedFourWarpL2 ∧
    result.artifact.sourceHash=candidateHash ∧ result.artifact.backend="cuda" ∧
    result.artifact.architecture=compiler.architecture p.input.owner ∧
    result.artifact.compiler=compiler.version ∧ result.artifact.options=compiler.options ∧
    result.artifact.signature=["*fp32","*fp32",if p.cols=1 then "constexpr" else "i32","fp32","constexpr"] ∧
    result.artifact.constexprs=((if p.cols=1 then [("N",1)] else [])++[("BLOCK_N",p.block)]) ∧
    result.artifact.attributes=([("X.divisibility",16),("Y.divisibility",16)]++
      (if p.cols%16=0 then [("N.divisibility",16)] else [])) ∧
    result.artifact.block=p.block ∧ result.artifact.warps=4 ∧ result.artifact.ctas=1 ∧
    ("enable_fp_fusion","true")∈compiler.options

/-- Validity, block selection and allocator alignment are local setup
preconditions already derived by preparation. The conclusion contains only
compiler metadata, not execution, address, numerical or output laws. Offline
PTX is finite supporting evidence for this ordinary compiler contract. -/
structure LocalCompilation {M : Type} (rt : Runtime M) (compiler : Compiler M)
    (instructions : Instructions) : Prop where
  setup : ∀ p before plan ready,Valid p → p.block=blockSize p.cols →
    p.input.baseByte%16=0 → p.output.baseByte%16=0 →
    rt.setup p before=.ok plan ready →
    BoundCompilation compiler instructions p before

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
      exact contracts.setup q.layout ready plan s facts.valid facts.block
        (facts.inputAligned (Nat.mul_pos (Nat.pos_of_ne_zero empty) facts.valid.positive)) facts.outputAligned hs

#axiomsClean compilation_evidence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
