import VeriTile.Triton.Memory.AlgorithmProgramIO

namespace VeriTile.Triton.AlgorithmProgramRegression
noncomputable section
set_option linter.unusedSimpArgs false

def unchangedIDs : AlgorithmLaunch where
  pids := fun _=>0
  setPids := fun _=>false

@[simp] theorem unchanged_initial (s : BlockState) : unchangedIDs.initial s=s := rfl

def addSource : AlgorithmSource where
  kernel := .fromKernelBody ["x"] ["tmp"] [
    .store .nat [] (.region ⟨"tmp"⟩ (.constNat 0))
      (.add .nat .nil (.load .nat (.region ⟨"x"⟩ (.constNat 0)) .none .zero) (.constNat 1)) .none]
def twiceSource : AlgorithmSource where
  kernel := .fromKernelBody ["tmp"] ["out"] [
    .store .nat [] (.region ⟨"out"⟩ (.constNat 0))
      (.mul .nat .nil (.load .nat (.region ⟨"tmp"⟩ (.constNat 0)) .none .zero) (.constNat 2)) .none]
def added (s : BlockState) := s.writeMemTyped .nat "tmp" 0 (s.readMemValue .nat "x" 0+1)
def doubled (s : BlockState) := s.writeMemTyped .nat "out" 0 (s.readMemValue .nat "tmp" 0*2)

theorem add_runs (s : BlockState) :
    (AlgorithmStage.single addSource unchangedIDs).Runs s (added s) := by
  rw [AlgorithmStage.single_runs]
  simp [addSource,ComputeKernel.fromKernelBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,Except.bind,bind,unchangedIDs,AlgorithmLaunch.initial,
    exec,stepStmts,stepStmt,evalOp,Tile.bop,Tile.scalar,TileShape.allIndices,added,
    BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,NumericDType.add,Region.cast,-Tile.scalar_eta]

theorem twice_runs (s : BlockState) :
    (AlgorithmStage.single twiceSource unchangedIDs).Runs s (doubled s) := by
  rw [AlgorithmStage.single_runs]
  simp [twiceSource,ComputeKernel.fromKernelBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,Except.bind,bind,unchangedIDs,AlgorithmLaunch.initial,
    exec,stepStmts,stepStmt,evalOp,Tile.bop,Tile.scalar,TileShape.allIndices,doubled,
    BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,NumericDType.mul,Region.cast,-Tile.scalar_eta]

def input : BlockState :=
  { (default : BlockState) with mem := fun r _ => MemCell.of .nat (if r="x" then 3 else if r="tmp" then 10 else 0) }

/-- The second distinct source observes the first source's actual stored value. -/
theorem successor_reads_predecessor :
    Nonempty (AlgorithmExecution [.single addSource unchangedIDs,.single twiceSource unchangedIDs]
      input (doubled (added input))) ∧
    (doubled (added input)).readMemValue .nat "out" 0=8 := by
  refine ⟨⟨.cons (add_runs input) (.cons (twice_runs _) (.nil _))⟩,?_⟩
  simp [added,doubled,input,BlockState.readMemValue,BlockState.readMemTyped,
    BlockState.writeMemTyped,MemCell.of,MemCell.readAs]

/-- Reversing the concrete stage order changes the observed result. -/
theorem reordered_sources_differ :
    Nonempty (AlgorithmExecution [.single twiceSource unchangedIDs,.single addSource unchangedIDs]
      input (added (doubled input))) ∧
    (added (doubled input)).readMemValue .nat "out" 0=20 ∧
    (added (doubled input)).mem "out" 0≠(doubled (added input)).mem "out" 0 := by
  refine ⟨⟨.cons (twice_runs input) (.cons (add_runs _) (.nil _))⟩,?_,?_⟩
  all_goals simp [added,doubled,input,BlockState.readMemValue,BlockState.readMemTyped,
    BlockState.writeMemTyped,MemCell.of,MemCell.readAs]

def outsideRead : AlgorithmSource where
  kernel := .fromKernelBody ["x"] [] [.assign .nat [] "v"
    (.load .nat (.region ⟨"x"⟩ (.constNat 1)) .none .zero)]

theorem unsafe_single_rejected (s : BlockState) :
    ¬(AlgorithmStage.single outsideRead unchangedIDs).Safe (fun _=>1) s := by
  intro safe
  change Kernel.TraceSafe (fun _=>1) _ s at safe
  simp only [Kernel.TraceSafe,outsideRead,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def] at safe
  have bound := safe.1.2.2
  have impossible := bound (Tile.scalar 1) (by simp [evalOp.eq_def]) PUnit.unit trivial
  exact Nat.lt_irrefl 1 impossible

def missingRegister : AlgorithmSource where
  kernel := .fromKernelBody [] [] [.assign .nat [] "v" (.ref .nat [] "missing")]

theorem failing_single_rejected (final : BlockState) :
    ¬(AlgorithmStage.single missingRegister unchangedIDs).Runs (default : BlockState) final := by
  rw [AlgorithmStage.single_runs]
  have absent : (default : BlockState).regs .nat [] "missing"=none := rfl
  simp [missingRegister,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,unchangedIDs,AlgorithmLaunch.initial,absent]

/-- No capacity witness is required to execute an empty plan or preserve its
entire caller memory; the initializer still does not write memory/registers. -/
theorem empty_plan_and_initializer (mode : AlgorithmInitialUndef) (s : BlockState) :
    Nonempty (AlgorithmExecution [] (mode.initial s) (mode.initial s)) ∧
    (AlgorithmExecution.nil (mode.initial s)).Safe (fun _=>0) ∧
    (mode.initial s).mem=s.mem ∧ (mode.initial s).regs=s.regs ∧
    (mode.initial s).pids=s.pids ∧ (mode.initial s).numPids=s.numPids :=
  ⟨⟨.nil _⟩,trivial,mode.initial_mem s,mode.initial_regs s,mode.initial_pids s,mode.initial_numPids s⟩

/-- Zeroing undef is a once-only initializer; individual stages preserve the
predecessor's undef rather than silently reinstalling it. -/
theorem zero_undef_once (s : BlockState) :
    (AlgorithmInitialUndef.zero.initial s).undef=(fun _ _=>0) ∧
    (unchangedIDs.initial s).undef=s.undef ∧
    (added s).undef=s.undef ∧ (doubled s).undef=s.undef := ⟨rfl,rfl,rfl,rfl⟩


def two : Grid := ⟨[2]⟩
def index (n : Fin 2) : GridIndex two := fun ax =>
  ⟨n.val,by have zero : ax.val=0 := by have h:=ax.isLt; change ax.val<1 at h; omega
            simpa [two,Grid.dim,zero] using n.isLt⟩

def swapSource : AlgorithmSource where
  kernel := .fromKernelBody ["x"] ["x"] [
    .store .nat [] (.region ⟨"x"⟩ (.programId 0))
      (.load .nat (.region ⟨"x"⟩ (.sub .nat .nil (.constNat 1) (.programId 0))) .none .zero) .none]
def swapNext (s : BlockState) := s.writeMemTyped .nat "x" (s.pids 0)
  (s.readMemValue .nat "x" (1-s.pids 0))

theorem swap_runs (s : BlockState) : exec swapSource.kernel.toAlgKernel s=some (swapNext s) := by
  simp [swapSource,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,Tile.bop,Tile.scalar,TileShape.allIndices,swapNext,
    BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,NumericDType.sub,Region.cast,-Tile.scalar_eta]

def swapFrames (s : BlockState) : Kernel.GridFrames swapSource.kernel.toAlgKernel two s :=
  fun idx => ⟨swapNext (s.withGridIndex idx),fun addr=>addr.1="x" ∧ addr.2=idx.toPids 0,
    swap_runs _,by
      intro r o outside
      change s.mem r o=(if r="x" ∧ o=idx.toPids 0 then _ else s.mem r o)
      rw [if_neg outside]⟩

theorem swap_disjoint (s : BlockState) : Kernel.GridWritesDisjoint (swapFrames s) := by
  intro a b distinct addr ha hb
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

def swapInput : BlockState :=
  { (default : BlockState) with mem := fun _ off => MemCell.of .nat (if off=0 then 1 else 10) }
def merged := Kernel.mergeFrames two swapInput (swapFrames swapInput)

def launchAt (n : Nat) : AlgorithmLaunch where
  pids := fun _=>n

def serial := swapNext ((launchAt 1).initial (swapNext ((launchAt 0).initial swapInput)))

/-- Ordinary grids preserve the existing same-initial-state interpretation.
The two programs swap aliased input/output cells; serial execution cannot do
so because its second load observes the first program's overwritten input. -/
theorem ordinary_alias_differs_from_serial :
    (AlgorithmStage.ordinary swapSource two).Runs swapInput merged ∧
    Nonempty (AlgorithmExecution [.single swapSource (launchAt 0),.single swapSource (launchAt 1)] swapInput serial) ∧
    merged.mem "x" 1=MemCell.of .nat 1 ∧ serial.mem "x" 1=MemCell.of .nat 10 := by
  refine ⟨⟨⟨swapFrames swapInput,swap_disjoint swapInput,rfl⟩⟩,?_,?_,?_⟩
  · exact ⟨.cons ((AlgorithmStage.single_runs _ _ _ _).mpr (swap_runs _))
      (.cons ((AlgorithmStage.single_runs _ _ _ _).mpr (swap_runs _)) (.nil _))⟩
  · have cell := Kernel.mergeFrames_mem_written (swap_disjoint swapInput) (index 1) "x" 1
      (by exact ⟨rfl,rfl⟩)
    change merged.mem "x" 1=_ at cell
    rw [cell]
    simp [swapFrames,swapNext,swapInput,index,two,GridIndex.toPids,Grid.rank,
      BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,MemCell.of,MemCell.readAs]
  · simp [serial,swapNext,swapInput,launchAt,AlgorithmLaunch.initial,
      BlockState.readMemValue,BlockState.readMemTyped,BlockState.writeMemTyped,MemCell.of,MemCell.readAs]

/-- An ordinary stage with zero declared source outputs still checks every
launched source load; absence of result channels does not erase safety. -/
theorem unsafe_outputless_ordinary_rejected (s : BlockState) :
    ¬(AlgorithmStage.ordinary outsideRead two).Safe (fun _=>1) s := by
  intro safe
  have launched := safe (index 0)
  exact unsafe_single_rejected (s.withGridIndex (index 0)) launched

def rewriteIndex : AlgorithmSource where
  kernel := .fromKernelBody [] ["index"] [
    .store .nat [] (.region ⟨"index"⟩ (.constNat 0)) (.constNat 1) .none]

def indirectRead : AlgorithmSource where
  kernel := .fromKernelBody ["index","x"] [] [
    .assign .nat [] "v" (.load .nat
      (.region ⟨"x"⟩ (.load .nat (.region ⟨"index"⟩ (.constNat 0)) .none .zero)) .none .zero)]

def indexInput : BlockState :=
  { (default : BlockState) with mem := fun _ _=>MemCell.of .nat 0 }
def indexWritten := indexInput.writeMemTyped .nat "index" 0 1
def indirectFinal := indexWritten.setReg "v" .nat [] (Tile.scalar 0)

theorem rewrite_index_runs :
    (AlgorithmStage.single rewriteIndex unchangedIDs).Runs indexInput indexWritten := by
  rw [AlgorithmStage.single_runs]
  simp [rewriteIndex,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,unchangedIDs,AlgorithmLaunch.initial,indexWritten,
    BlockState.writeMemTyped,TileShape.allIndices,Tile.scalar,Region.cast,-Tile.scalar_eta]

theorem indirect_read_runs :
    (AlgorithmStage.single indirectRead unchangedIDs).Runs indexWritten indirectFinal := by
  rw [AlgorithmStage.single_runs]
  simp [indirectRead,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,unchangedIDs,AlgorithmLaunch.initial,indirectFinal,
    indexWritten,indexInput,BlockState.readMemValue,BlockState.readMemTyped,
    BlockState.writeMemTyped,TileShape.allIndices,Tile.scalar,Region.cast,MemCell.readAs,MemCell.of,-Tile.scalar_eta]

def indirectTrace : AlgorithmExecution
    [.single rewriteIndex unchangedIDs,.single indirectRead unchangedIDs] indexInput indirectFinal :=
  .cons rewrite_index_runs (.cons indirect_read_runs (.nil _))

/-- Safety for the second source uses the actual predecessor memory: its
indirect load would be safe on the original state, but the first source changes
the index to one, outside the declared one-element input. -/
theorem successor_safety_uses_predecessor :
    (AlgorithmStage.single indirectRead unchangedIDs).Safe (fun _=>1) indexInput ∧
    ¬indirectTrace.Safe (fun _=>1) := by
  constructor
  · simp [AlgorithmStage.Safe,unchangedIDs,AlgorithmLaunch.initial,Kernel.TraceSafe,
      indirectRead,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
      Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,stepStmt.eq_def,
      indexInput,BlockState.readMemValue,BlockState.readMemTyped,MemCell.readAs,MemCell.of,Region.cast,Tile.scalar,-Tile.scalar_eta]
  · intro safe
    have later := safe.2.1
    change Kernel.TraceSafe (fun _=>1) _ indexWritten at later
    simp only [Kernel.TraceSafe,indirectRead,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,
      Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def] at later
    have bound := later.1.2.2
    have impossible := bound (Tile.scalar 1) (by
      simp [evalOp.eq_def,indexWritten,indexInput,BlockState.readMemValue,BlockState.readMemTyped,
        BlockState.writeMemTyped,MemCell.readAs,MemCell.of,Region.cast,Tile.scalar,-Tile.scalar_eta]) PUnit.unit trivial
    exact Nat.lt_irrefl 1 impossible

end
end VeriTile.Triton.AlgorithmProgramRegression
