import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.SourceAST
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Physical
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.FPReference
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentContract

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Resident
open Host
open CrossEntropyOps.Resident (SourceLanguage)
open CrossEntropy1.Host (Step)
noncomputable section

inductive Interpretation where
  | numeric | marker

def Scalar : Interpretation → Type
  | .numeric => Floating.Expr
  | .marker => Physical.Marker

def arithmetic : (mode : Interpretation) → Source.Arithmetic (Scalar mode) (Scalar mode)
  | .numeric => Floating.arithmetic
  | .marker => Physical.arithmetic

structure Model (σ : Type) where
  mode : Interpretation
  runtime : Runtime σ
  live : σ → Nat → Prop
  stream : Nat

/-- Complete concrete launch information: actual returned/input descriptors,
pointer words, widened scalar words, launch grid and both source profiles. -/
structure LaunchABI where
  args : Arguments
  output : Tensor
  sms : Nat
  pointerWords : List (BitVec 64)
  scalarWords : List (BitVec 64)
  grid : Nat × Nat × Nat
  blockM : Nat
  blockN : Nat
  blockK : Nat
  groupM : Nat
  warps : Nat
  stages : Nat

def callABI : Call → Option LaunchABI
  | .launch args sms output =>
      some ⟨args,output,sms,[args.A.pointer,args.B.pointer,output.pointer],
        [args.A.rows,args.B.columns,args.A.columns,args.A.rowStride,args.A.columnStride,
          args.B.rowStride,args.B.columnStride,output.rowStride,output.columnStride].map (BitVec.ofNat 64),
        ((args.problem sms).grid,1,1),128,256,args.A.dtype.blockK,8,8,args.A.dtype.stages⟩
  | _ => none

structure CompleteABI (σ : Type) where
  args : Arguments
  initial : σ
  returned : Tensor
  calls : List Call
  launches : List LaunchABI
  scalarSignatureWidths : List Nat
  target : String
  compilerVersion : String
  stream : Nat
  enableFPFusion : Bool

structure Metadata where
  returned : Tensor
  calls : List Call
  launches : List LaunchABI

def run {σ : Type} (model : Model σ) (program : Arguments) (initial : World (Scalar model.mode) σ) :
    Step (World (Scalar model.mode) σ) Tensor :=
  match Host.run (arithmetic model.mode) model.runtime program initial with
  | none => .error "ordinary API/runtime failure" initial
  | some (result,final) => .ok result final

theorem run_success {σ : Type} (model : Model σ) (program : Arguments)
    (initial final : World (Scalar model.mode) σ) (result : Tensor)
    (success : run model program initial=.ok result final) :
    Host.run (arithmetic model.mode) model.runtime program initial=some (result,final) := by
  cases h : Host.run (arithmetic model.mode) model.runtime program initial with
  | none => simp [run,h] at success
  | some pair =>
    rcases pair with ⟨returned,finished⟩
    simp only [run,h,Step.ok.injEq] at success
    exact congrArg some (Prod.ext success.1 success.2)

/-- Full AST/digest and source host/grid interpreter are installed together.
The fixed interpreter threads actual native state through every runtime call. -/
def language (σ : Type) : SourceLanguage σ where
  Program := Arguments
  Syntax := ModuleAST
  Model := Model σ
  State model := World (Scalar model.mode) σ
  Result _ := Tensor
  ABI _ := CompleteABI σ
  Metadata _ := Metadata
  ast _ := moduleAST
  sourceDigest _ := pythonDigest
  native _ world := world.native
  run := run
  coherent model program abi initial := abi.args=program ∧ abi.initial=initial.native ∧
    model.stream=abi.stream ∧ abi.enableFPFusion=Bool.true ∧ initial.calls=[] ∧
    InputContract model.live program initial.native
  abiMetadata _ abi := ⟨abi.returned,abi.calls,abi.launches⟩
  metadata _ _ _ result final _ := ⟨result,final.calls,final.calls.filterMap callABI⟩

#axiomsClean run_success
#specNonCircular run avoiding [Floating.reference,Floating.Plan,Floating.Plan.graph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Resident
