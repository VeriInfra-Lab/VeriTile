import VeriTile.Triton.Memory.ParameterizedAlgorithmProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.ParameterizedAlgorithmRegression
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def realArgument (value : ℝ) : AlgorithmRegisterBinding :=
  ⟨"sm_scale",.real,[],Tile.scalar (some value)⟩
def natArgument (name : RegName) (value : Nat) : AlgorithmRegisterBinding :=
  ⟨name,.nat,[],Tile.scalar value⟩

def realSource : AlgorithmSource where
  kernel := .fromKernelBody [] ["out"] [
    .store .real [] (.region "out" (.constNat 0)) (.ref .real [] "sm_scale") .none]

/-- The source contains an actual runtime register reference. Its stored
value changes with the binding; no literal-specialized source is executed. -/
theorem runtime_real_read (s : BlockState) (value : ℝ) :
    exec realSource.kernel.toAlgKernel (bindAlgorithmArguments [realArgument value] s)=
      some ((bindAlgorithmArguments [realArgument value] s).writeMem "out" 0 value) := by
  simp [realSource,realArgument,ComputeKernel.toAlgKernel_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,TileShape.allIndices,Tile.scalar,
    BlockState.writeMemTyped,BlockState.writeMem,-Tile.scalar_eta]

/-- Later bindings replace a name even at a different dtype or shape. -/
theorem typed_name_replaced (s : BlockState) (value : ℝ) :
    (bindAlgorithmArguments [natArgument "sm_scale" 7,realArgument value] s).regs .real [] "sm_scale"=
      some (Tile.scalar (some value)) ∧
    (bindAlgorithmArguments [natArgument "sm_scale" 7,realArgument value] s).regs .nat [] "sm_scale"=none ∧
    (bindAlgorithmArguments [realArgument value,
      ⟨"sm_scale",.real,[1],⟨fun _=>some value⟩⟩] s).regs .real [] "sm_scale"=none := by
  simp [bindAlgorithmArguments,List.foldl_cons,realArgument,natArgument,BlockState.setReg]

def two : Grid := ⟨[2]⟩
def index (n : Fin 2) : GridIndex two := fun ax=>
  ⟨n.val,by
    have zero : ax.val=0 := by have h:=ax.isLt; change ax.val<1 at h; omega
    simpa [two,Grid.dim,zero] using n.isLt⟩

def swapSource : AlgorithmSource where
  kernel := .fromKernelBody ["x"] ["x"] [
    .store .nat [] (.region "x" (.programId 0))
      (.add .nat .nil
        (.load .nat (.region "x" (.sub .nat .nil (.constNat 1) (.programId 0))) .none .zero)
        (.ref .nat [] "bump")) .none]
def next (s : BlockState) (bump : Nat) :=
  s.writeMemTyped .nat "x" (s.pids 0) (s.readMemValue .nat "x" (1-s.pids 0)+bump)

theorem swap_runs (s : BlockState) (bump : Nat)
    (bound : s.regs .nat [] "bump"=some (Tile.scalar bump)) :
    exec swapSource.kernel.toAlgKernel s=some (next s bump) := by
  simp [swapSource,ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,
    bound,Tile.bop,Tile.scalar,TileShape.allIndices,next,BlockState.readMemValue,
    BlockState.readMemTyped,BlockState.writeMemTyped,NumericDType.add,NumericDType.sub,Region.cast,-Tile.scalar_eta]

def frames (s : BlockState) (bump : Nat)
    (bound : s.regs .nat [] "bump"=some (Tile.scalar bump)) :
    Kernel.GridFrames swapSource.kernel.toAlgKernel two s :=
  fun idx=>⟨next (s.withGridIndex idx) bump,fun addr=>addr.1="x" ∧ addr.2=idx.toPids 0,
    swap_runs _ bump bound,by
      intro r off outside
      change s.mem r off=(if r="x" ∧ off=idx.toPids 0 then _ else s.mem r off)
      rw [if_neg outside]⟩

theorem frames_disjoint (s : BlockState) (bump : Nat)
    (bound : s.regs .nat [] "bump"=some (Tile.scalar bump)) :
    Kernel.GridWritesDisjoint (frames s bump bound) := by
  intro a b distinct address ha hb
  apply distinct
  have eq : a.toPids 0=b.toPids 0 := ha.2.symm.trans hb.2
  funext ax
  have zero : ax=(⟨0,by decide⟩ : Fin two.rank) := by
    apply Fin.ext
    have h:=ax.isLt
    change ax.val<1 at h
    change ax.val=0
    omega
  subst ax
  apply Fin.ext
  exact eq

def input : BlockState :=
  {(default : BlockState) with mem:=fun _ off=>MemCell.of .nat (if off=0 then 1 else 10)}
def prepared := bindAlgorithmArguments [natArgument "bump" 2] input
theorem bump_bound : prepared.regs .nat [] "bump"=some (Tile.scalar 2) := by
  simp [prepared,natArgument]
def merged := Kernel.mergeFrames two prepared (frames prepared 2 bump_bound)
def serial := next ((next (prepared.withGridIndex (index 0)) 2).withGridIndex (index 1)) 2

/-- Both ordinary CTAs read the same bound initial state. Aliasing makes a
serial substitution observably wrong even though both read the runtime bump. -/
theorem ordinary_grid_keeps_common_initial :
    (AlgorithmStage.ordinary swapSource two).Runs prepared merged ∧
    merged.mem "x" 0=MemCell.of .nat 12 ∧ merged.mem "x" 1=MemCell.of .nat 3 ∧
    serial.mem "x" 1=MemCell.of .nat 14 := by
  refine ⟨⟨⟨frames prepared 2 bump_bound,frames_disjoint prepared 2 bump_bound,rfl⟩⟩,?_,?_,?_⟩
  · have cell := Kernel.mergeFrames_mem_written (frames_disjoint prepared 2 bump_bound)
      (index 0) "x" 0 (by exact ⟨rfl,rfl⟩)
    change merged.mem "x" 0=_ at cell
    rw [cell]
    simp [frames,next,prepared,natArgument,input,index,two,GridIndex.toPids,Grid.rank,
      BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,MemCell.of,MemCell.readAs]
  · have cell := Kernel.mergeFrames_mem_written (frames_disjoint prepared 2 bump_bound)
      (index 1) "x" 1 (by exact ⟨rfl,rfl⟩)
    change merged.mem "x" 1=_ at cell
    rw [cell]
    simp [frames,next,prepared,natArgument,input,index,two,GridIndex.toPids,Grid.rank,
      BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,MemCell.of,MemCell.readAs]
  · simp [serial,next,prepared,natArgument,input,index,two,GridIndex.toPids,Grid.rank,
      BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,MemCell.of,MemCell.readAs]

def indirectSource : AlgorithmSource where
  kernel := .fromKernelBody ["x"] [] [
    .assign .nat [] "value" (.load .nat (.region "x" (.ref .nat [] "address")) .none .zero)]
def addressState (off : Nat) := bindAlgorithmArguments [natArgument "address" off] input

/-- Safety sees the installed runtime address. Checking the old caller's
register value would incorrectly accept the rebound out-of-range access. -/
theorem source_safety_uses_argument :
    (AlgorithmStage.ordinary indirectSource two).Safe (fun _=>1) (addressState 0) ∧
    ¬(AlgorithmStage.ordinary indirectSource two).Safe (fun _=>1)
      (bindAlgorithmArguments [natArgument "address" 1] (addressState 0)) := by
  constructor
  · intro idx
    simp [AlgorithmStage.Safe,Kernel.TraceSafe,indirectSource,ComputeKernel.toAlgKernel_fromKernelBody,
      Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,
      addressState,natArgument,BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,stepStmt.eq_def]
  · intro safe
    have launched := safe (index 0)
    simp only [Kernel.TraceSafe,indirectSource,ComputeKernel.toAlgKernel_fromKernelBody,
      Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def] at launched
    have bound := launched.1.2.2
    have impossible := bound (Tile.scalar 1) (by
      simp [evalOp.eq_def,natArgument,BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids])
        PUnit.unit trivial
    exact Nat.lt_irrefl 1 impossible

def emptyIO (args : List AlgorithmRegisterBinding) (plan : List AlgorithmStage)
    (mode : AlgorithmInitialUndef := .preserve) : ParameterizedAlgorithmProgramIO where
  nIn:=0
  nOut:=0
  nScr:=0
  bufs:=[]
  ity:=Fin.elim0
  iarity:=Fin.elim0
  ibuf:=Fin.elim0
  oty:=Fin.elim0
  oarity:=Fin.elim0
  obuf:=Fin.elim0
  obuf_mem:=fun o=>Fin.elim0 o
  sarity:=Fin.elim0
  sbuf:=Fin.elim0
  iwin:=fun i=>Fin.elim0 i
  imask:=fun i=>Fin.elim0 i
  owin:=fun i=>Fin.elim0 i
  omask:=fun i=>Fin.elim0 i
  swin:=fun i=>Fin.elim0 i
  smask:=fun i=>Fin.elim0 i
  initialUndef:=mode
  stages:=fun _=>plan
  arguments:=fun _=>args

theorem empty_preserves_original_memory (args : List AlgorithmRegisterBinding) (mode : AlgorithmInitialUndef) :
    (emptyIO args [] mode).Implements (fun _ o=>Fin.elim0 o) := by
  intro values _ original _
  refine ⟨(emptyIO args [] mode).initial values original,.nil _,?_,?_,?_⟩
  · intro _ _; trivial
  · intro o; exact Fin.elim0 o
  · intro r off _ _
    exact congrFun (congrFun (ParameterizedAlgorithmProgramIO.initial_mem _ _ _) r) off

theorem undef_modes_and_memory (args : List AlgorithmRegisterBinding) (s : BlockState) :
    ((emptyIO args [] .preserve).initial (fun i=>Fin.elim0 i) s).undef=s.undef ∧
    ((emptyIO args [] .zero).initial (fun i=>Fin.elim0 i) s).undef=(fun _ _=>0) ∧
    ((emptyIO args [] .zero).initial (fun i=>Fin.elim0 i) s).mem=s.mem := by
  simp [ParameterizedAlgorithmProgramIO.initial,emptyIO,AlgorithmInitialUndef.initial]

example (io : AlgorithmProgramIO) (domain : io.Ctx → Prop)
    (formula : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) :
    (ParameterizedAlgorithmProgramIO.ofProgram io).ImplementsOn3D domain formula ↔
      io.ImplementsOn3D domain formula := Iff.rfl

#axiomsClean runtime_real_read
#axiomsClean typed_name_replaced
#axiomsClean ordinary_grid_keeps_common_initial
#axiomsClean source_safety_uses_argument
#axiomsClean empty_preserves_original_memory
#axiomsClean undef_modes_and_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Triton.ParameterizedAlgorithmRegression
