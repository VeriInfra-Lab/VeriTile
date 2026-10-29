import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostForwardExecution
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.SourceAST
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Physical

namespace VeriTile.Bench.Repairs.FastCeLoss.Resident
open Source Host
open CrossEntropyOps.Resident (SourceLanguage)
noncomputable section

inductive Interpretation where
  | numeric | marker

def Scalar : Interpretation → Type
  | .numeric => Floating.Expr
  | .marker => Physical.Marker

def arithmetic : (mode : Interpretation) → Arithmetic (Scalar mode)
  | .numeric => Floating.arithmetic
  | .marker => Physical.markerArithmetic

def scalarInterpreter : (mode : Interpretation) → ScalarInterpreter (Scalar mode)
  | .numeric => fpScalar
  | .marker => ⟨fun _ => .changed,fun _ => .changed⟩

structure Model (σ : Type) where
  interpretation : Interpretation
  runtime : Runtime σ
  stream : Nat

structure Program where
  args : Arguments
  upstream : Descriptor

structure Metadata where
  returned : WholeReturn
  calls : List Call
  allocations : List Descriptor

def run {σ : Type} (model : Model σ) (program : Program) : Action (Scalar model.interpretation) σ WholeReturn :=
  Host.whole (arithmetic model.interpretation) (scalarInterpreter model.interpretation)
    model.runtime model.stream program.args program.upstream

/-- The installed source language has a fixed AST/digest and fixed host/grid
interpreter. Only ordinary runtime effects vary; every one threads native state. -/
def language (σ : Type) : SourceLanguage σ where
  Program := Program
  Syntax := ModuleAST
  Model := Model σ
  State model := World (Scalar model.interpretation) σ
  Result _ := WholeReturn
  ABI _ := CompleteABI σ
  Metadata _ := Metadata
  ast _ := moduleAST
  sourceDigest _ := pythonDigest
  native _ world := world.native
  run := run
  coherent model program abi initial := abi.args=program.args ∧ abi.scalarUpstream=program.upstream ∧
    abi.initial=initial.native ∧ model.stream=abi.stream ∧ abi.enableFPFusion=Bool.true ∧ initial.calls=[] ∧ initial.allocations=[]
  abiMetadata _ abi := ⟨abi.returned,abi.calls,abi.allocationTrace⟩
  metadata _ _ _ result final _ := ⟨result,final.calls,final.allocations⟩

def exports (result : WholeReturn) : List Descriptor :=
  [result.forward.scalar,result.forward.customLoss,result.forward.context.logits,result.forward.context.lse,
    result.forward.context.labels,result.publicGradient,result.upstream]

theorem native_state_threaded {σ : Type} (model : Model σ) (program : Program)
    (initial : World (Scalar model.interpretation) σ) :
    (language σ).run model program initial=Host.whole (arithmetic model.interpretation)
      (scalarInterpreter model.interpretation) model.runtime model.stream program.args program.upstream initial := rfl

#axiomsClean native_state_threaded
#specNonCircular run avoiding [Complete.lseGraph,Complete.lossGraph,Complete.scalarGraph,API.gradientGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Resident
