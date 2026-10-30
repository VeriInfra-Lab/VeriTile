import bench.tritonbench_g.sin_computation.Proofs.NativeMemory

namespace SinComputationNative
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SinComputation
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
inductive IndexExpr where
  | literal (n : Nat)
  | program (ax : Nat)
  | register (name : RegName)
  | lane (width : Nat)
  | add (a b : IndexExpr)
  | mul (a b : IndexExpr)
  deriving DecidableEq

def lowerIndex {shape : TileShape} : Op .nat shape → Option IndexExpr
  | .constNat n => some (.literal n)
  | .programId ax => some (.program ax)
  | .ref _ _ name => some (.register name)
  | .arange width => some (.lane width)
  | .add _ _ a b => return .add (←lowerIndex a) (←lowerIndex b)
  | .mul _ _ a b => return .mul (←lowerIndex a) (←lowerIndex b)
  | _ => none

inductive MaskExpr where
  | register (name : RegName)
  | less (a b : IndexExpr)
  deriving DecidableEq

def lowerBool {shape : TileShape} : Op .bool shape → Option MaskExpr
  | .ref _ _ name => some (.register name)
  | .lt .nat _ a b => return .less (←lowerIndex a) (←lowerIndex b)
  | _ => none

def lowerMask {shape : TileShape} : MaskOpt .real shape → Option MaskExpr
  | .mask value => lowerBool value
  | _ => none

inductive Instruction where
  | indexAssign (name : RegName) (value : IndexExpr)
  | maskAssign (name : RegName) (value : MaskExpr)
  | load (name : RegName) (region : RegionName) (offset : IndexExpr) (mask : MaskExpr)
  | sinAssign (name a : RegName)
  | store (region : RegionName) (offset : IndexExpr) (value : RegName) (mask : MaskExpr)
  deriving DecidableEq

def lowerStatement : Stmt → Option Instruction
  | .assign .nat _ name value => return .indexAssign name (←lowerIndex value)
  | .assign .bool _ name value => return .maskAssign name (←lowerBool value)
  | .assign .real _ name (.load .real (.region region offset) mask _) =>
      return .load name region (←lowerIndex offset) (←lowerMask mask)
  | .assign .real _ name (.sin (.ref _ _ a)) => some (.sinAssign name a)
  | .store .real _ (.region region offset) (.ref _ _ value) mask =>
      return .store region (←lowerIndex offset) value (←lowerMask mask)
  | _ => none

def lowerBody : List Stmt → Option (List Instruction)
  | [] => some []
  | stmt::rest => return (←lowerStatement stmt)::(←lowerBody rest)

def sourceCode (X O : RegionName) (n : Nat) : List Instruction :=
  [.indexAssign "pid" (.program 0),
   .indexAssign "block_start" (.mul (.register "pid") (.literal 4)),
   .indexAssign "offsets" (.add (.register "block_start") (.lane 4)),
   .maskAssign "mask" (.less (.register "offsets") (.literal n)),
   .load "x" X (.register "offsets") (.register "mask"),
   .sinAssign "output" "x",
   .store O (.register "offsets") "output" (.register "mask")]

theorem original_ast_lowering (X O : RegionName) (n : Nat) :
    lowerBody (sin_kernel X O n 4).body=some (sourceCode X O n) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerBool,lowerMask,sin_kernel,ComputeKernel.body,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,bind,pure,Except.bind,Option.bind]


structure PrefixRegisters where
  indices : RegName → Nat → BitVec 32
  masks : RegName → Nat → Bool

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
  | .less a b => (evalIndex pid lane indices a).slt (evalIndex pid lane indices b)

def stepPrefix (pid : Nat) (s : PrefixRegisters) : Instruction → PrefixRegisters
  | .indexAssign name value => {s with indices := fun key lane => if key=name then evalIndex pid lane s.indices value else s.indices key lane}
  | .maskAssign name value => {s with masks := fun key lane => if key=name then evalMask pid lane s.indices s.masks value else s.masks key lane}
  | _ => s

def runPrefix (pid : Nat) (code : List Instruction) : PrefixRegisters :=
  code.foldl (stepPrefix pid) ⟨fun _ _ => 0,fun _ _ => Bool.false⟩

/-- Extracted from the original body. The get proof eliminates parser failure;
there is no default instruction list or fabricated memory value. -/
def compiledSource (X O : RegionName) (n : Nat) : List Instruction :=
  (lowerBody (sin_kernel X O n 4).body).get (by simp [original_ast_lowering])
theorem compiled_source (X O : RegionName) (n : Nat) :
    compiledSource X O n=sourceCode X O n := by simp [compiledSource,original_ast_lowering]

def sourceActiveTile (n pid : Nat) : List Nat :=
  (List.range 4).filter ((runPrefix pid (compiledSource "x_ptr" "output_ptr" n)).masks "mask")

theorem source_active_tile (n pid : Nat) : sourceActiveTile n pid=activeTile n pid := by
  simp only [sourceActiveTile,compiled_source,sourceCode,runPrefix,List.foldl_cons,List.foldl_nil,
    stepPrefix,evalIndex,evalMask]
  simp only [if_true,String.reduceEq,if_false]
  rfl

def sourceOffset (n pid lane : Nat) : BitVec 32 :=
  (runPrefix pid (compiledSource "x_ptr" "output_ptr" n)).indices "offsets" lane
theorem source_offset (n pid lane : Nat) : sourceOffset n pid lane=offset pid lane := by
  simp only [sourceOffset,compiled_source,sourceCode,runPrefix,List.foldl_cons,List.foldl_nil,
    stepPrefix,evalIndex,evalMask]
  simp only [if_true,String.reduceEq,if_false]
  rfl
def sourcePointer (data width n pid lane : Nat) : Nat :=
  (BitVec.ofNat 64 data + (sourceOffset n pid lane).signExtend 64 * BitVec.ofNat 64 width).toNat
theorem source_pointer (data width n pid lane : Nat) :
    sourcePointer data width n pid lane=pointer data width pid lane := by
  rw [sourcePointer,source_offset];rfl

/-- The same lowered instruction stream contains the one raw masked load,
precision-preserving sine, and one matching masked store in this exact order. -/
theorem complete_source_suffix (X O : RegionName) (n : Nat) :
    (compiledSource X O n).drop 4 =
      [.load "x" X (.register "offsets") (.register "mask"),
       .sinAssign "output" "x",.store O (.register "offsets") "output" (.register "mask")] := by
  rw [compiled_source];rfl

def lowerScalar (d : Storage) : Instruction → Option (Expr d)
  | .sinAssign "output" "x" => some (.convert (.sine .input))
  | _ => none

/-- The expression executed by the source program is obtained from instruction5
of its lowered AST. Its get proof excludes an unsupported source instruction. -/
def compiledScalar (d : Storage) (n : Nat) : Expr d :=
  ((compiledSource "x_ptr" "output_ptr" n)[5]?.bind (lowerScalar d)).get
    (by simp [compiled_source,sourceCode,lowerScalar])

theorem compiled_scalar (d : Storage) (n : Nat) : compiledScalar d n=source d := by
  simp [compiledScalar,compiled_source,sourceCode,lowerScalar,source]

#axiomsClean original_ast_lowering
#axiomsClean compiled_source
#axiomsClean source_active_tile
#axiomsClean source_offset
#axiomsClean source_pointer
#axiomsClean complete_source_suffix
#axiomsClean compiled_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end SinComputationNative
