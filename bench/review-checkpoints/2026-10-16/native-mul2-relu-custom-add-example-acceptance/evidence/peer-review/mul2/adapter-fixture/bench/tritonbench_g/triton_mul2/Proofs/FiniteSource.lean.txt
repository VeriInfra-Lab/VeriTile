import bench.tritonbench_g.triton_mul2.Proofs.FiniteProgram
import bench.tritonbench_g.triton_mul2.Kernels

namespace TritonMul2Finite
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.TritonMul2
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
noncomputable section
open Classical
local instance : DecidableEq ℝ := Classical.decEq _

inductive IndexExpr where
  | literal (n : Nat) | program (ax : Nat) | register (name : RegName) | lane (width : Nat)
  | add (a b : IndexExpr) | mul (a b : IndexExpr)
  deriving DecidableEq
inductive MaskExpr where
  | register (name : RegName) | less (a b : IndexExpr)
  deriving DecidableEq
def lowerIndex {shape : TileShape} : Op .nat shape → Option IndexExpr
  | .constNat n => some (.literal n)
  | .programId ax => some (.program ax)
  | .ref _ _ name => some (.register name)
  | .arange width => some (.lane width)
  | .add _ _ a b => return .add (←lowerIndex a) (←lowerIndex b)
  | .mul _ _ a b => return .mul (←lowerIndex a) (←lowerIndex b)
  | _ => none
def lowerBool {shape : TileShape} : Op .bool shape → Option MaskExpr
  | .ref _ _ name => some (.register name)
  | .lt .nat _ a b => return .less (←lowerIndex a) (←lowerIndex b)
  | _ => none
def lowerMask {shape : TileShape} : MaskOpt .real shape → Option MaskExpr
  | .mask e => lowerBool e
  | _ => none
inductive Instruction where
  | indexAssign (name : RegName) (value : IndexExpr)
  | maskAssign (name : RegName) (value : MaskExpr)
  | load (name : RegName) (region : RegionName) (offset : IndexExpr) (mask : MaskExpr)
  | double (name input : RegName)
  | store (region : RegionName) (offset : IndexExpr) (name : RegName) (mask : MaskExpr)
def lowerStatement : Stmt → Option Instruction
  | .assign .nat _ name e => return .indexAssign name (←lowerIndex e)
  | .assign .bool _ name e => return .maskAssign name (←lowerBool e)
  | .assign .real _ name (.load .real (.region region offset) mask _) =>
      return .load name region (←lowerIndex offset) (←lowerMask mask)
  | .assign .real _ name (.mul .real _ (.const v) (.ref _ _ input)) =>
      if v=2 then some (.double name input) else none
  | .store .real _ (.region region offset) (.ref _ _ name) mask =>
      return .store region (←lowerIndex offset) name (←lowerMask mask)
  | _ => none
def lowerBody : List Stmt → Option (List Instruction)
  | [] => some []
  | stmt::rest => return (←lowerStatement stmt)::(←lowerBody rest)
def sourceCode (input output : RegionName) (n block : Nat) : List Instruction :=
  [.indexAssign "pid" (.program 0),
   .indexAssign "block_start" (.mul (.register "pid") (.literal block)),
   .indexAssign "offsets" (.add (.register "block_start") (.lane block)),
   .maskAssign "mask" (.less (.register "offsets") (.literal n)),
   .load "x" input (.register "offsets") (.register "mask"),
   .double "output" "x",
   .store output (.register "offsets") "output" (.register "mask")]
theorem original_ast_lowering (input output : RegionName) (n block : Nat) :
    lowerBody (mul2_kernel input output n block).body=some (sourceCode input output n block) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerMask,lowerBool,mul2_kernel,
    ComputeKernel.body,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,
    bind,pure,Except.bind,Option.bind]
theorem inplace_ast_lowering (ptr : RegionName) (n block : Nat) :
    lowerBody (mul2_inplace_kernel ptr n block).body=some (sourceCode ptr ptr n block) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerMask,lowerBool,mul2_inplace_kernel,
    ComputeKernel.body,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,
    bind,pure,Except.bind,Option.bind]

def evalIndex (pid lane : Nat) (registers : RegName → Nat → BitVec 32) : IndexExpr → BitVec 32
  | .literal n => BitVec.ofNat 32 n
  | .program ax => if ax=0 then BitVec.ofNat 32 pid else 0
  | .register name => registers name lane
  | .lane _ => BitVec.ofNat 32 lane
  | .add a b => evalIndex pid lane registers a + evalIndex pid lane registers b
  | .mul a b => evalIndex pid lane registers a * evalIndex pid lane registers b
def evalMask (pid lane : Nat) (indices : RegName → Nat → BitVec 32)
    (masks : RegName → Nat → Bool) : MaskExpr → Bool
  | .register name => masks name lane
  | .less a b => decide ((evalIndex pid lane indices a).toInt < (evalIndex pid lane indices b).toInt)

/-- Supported source subset. The final load/multiply/store statements are
fused only after checking register, pointer and mask links. The complete tile
is read before the multiply/store phase, including for the same pointer. -/
def executeBody (b : Numerics) (d : Storage) (p : Layout) (input output : RegionName)
    (block pid : Nat) (state : State) (indices : RegName → Nat → BitVec 32)
    (masks : RegName → Nat → Bool) : List Instruction → Except Issue State
  | [] => .ok state
  | .indexAssign name e :: rest => executeBody b d p input output block pid state
      (fun key lane => if key=name then evalIndex pid lane indices e else indices key lane) masks rest
  | .maskAssign name e :: rest => executeBody b d p input output block pid state indices
      (fun key lane => if key=name then evalMask pid lane indices masks e else masks key lane) rest
  | [.load name region offset mask,.double value operand,.store target outputOffset storedName outputMask] =>
      if region=input ∧ target=output ∧ name=operand ∧ value=storedName ∧ mask=outputMask then do
        let loaded ← loadTile d p state.memory pid
          (fun lane => evalIndex pid lane indices offset)
          (fun lane => evalIndex pid lane indices outputOffset)
          (fun lane => evalMask pid lane indices masks mask) (List.range block)
        let (writes,owners) ← tileWrites d (fun x => stored d (source.eval b d x)) p pid loaded state.owners
        pure ⟨applyWrites writes state.memory,owners⟩
      else .error .sourceElaboration
  | _ => .error .sourceElaboration

theorem source_body_correspondence (b : Numerics) (d : Storage) (p : Layout)
    (input output : RegionName) (n block pid : Nat) (state : State) :
    executeBody b d p input output block pid state (fun _ _ => 0) (fun _ _ => Bool.false)
      (sourceCode input output n block)=executeProgram b d p n block pid state := by
  simp only [sourceCode,executeBody,evalIndex,evalMask,executeProgram,
    if_pos rfl,if_true,String.reduceEq,if_false,and_self]
  rfl

def runOriginalProgram (b : Numerics) (d : Storage) (p : Layout)
    (input output : RegionName) (n block pid : Nat) (state : State) : Except Issue State :=
  match lowerBody (mul2_kernel input output n block).body with
  | none => .error .sourceElaboration
  | some code => executeBody b d p input output block pid state (fun _ _ => 0) (fun _ _ => Bool.false) code
def runInplaceProgram (b : Numerics) (d : Storage) (p : Layout)
    (ptr : RegionName) (n block pid : Nat) (state : State) : Except Issue State :=
  match lowerBody (mul2_inplace_kernel ptr n block).body with
  | none => .error .sourceElaboration
  | some code => executeBody b d p ptr ptr block pid state (fun _ _ => 0) (fun _ _ => Bool.false) code
theorem original_program_correspondence (b : Numerics) (d : Storage) (p : Layout)
    (input output : RegionName) (n block pid : Nat) (state : State) :
    runOriginalProgram b d p input output n block pid state=executeProgram b d p n block pid state := by
  rw [runOriginalProgram,original_ast_lowering]
  exact source_body_correspondence b d p input output n block pid state
theorem inplace_program_correspondence (b : Numerics) (d : Storage) (p : Layout)
    (ptr : RegionName) (n block pid : Nat) (state : State) :
    runInplaceProgram b d p ptr n block pid state=executeProgram b d p n block pid state := by
  rw [runInplaceProgram,inplace_ast_lowering]
  exact source_body_correspondence b d p ptr ptr n block pid state

#axiomsClean original_ast_lowering
#axiomsClean inplace_ast_lowering
#axiomsClean source_body_correspondence
#axiomsClean original_program_correspondence
#axiomsClean inplace_program_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
