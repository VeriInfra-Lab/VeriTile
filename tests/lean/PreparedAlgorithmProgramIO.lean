import VeriTile.Triton.Memory.PreparedAlgorithmProgramIO
import VeriTile.Meta.StatementAudit

namespace PreparedAlgorithmRegression
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def source (body : List Stmt) : AlgorithmSource where
  kernel := ComputeKernel.fromKernelBody [] [] body
  projection := by simp
def job (body : List Stmt) (clear zero : Bool := false) : PreparedAlgorithmInvocation where
  source := source body
  launch := {pids:=fun _=>0,setPids:=fun _=>false}
  clearRegisters := clear
  zeroUndef := zero
def seed : BlockState :=
  ({(default : BlockState) with undef:=fun _ _=>7}).setReg "arg" .real [] (Tile.scalar (some (3 : ℝ)))
def copyReg : List Stmt := [.assign .real [] "result" (.ref .real [] "arg")]
def omitted : List Stmt :=
  [.assign .real [] "result" (.load .real (.region "x" (.constNat 0)) (.mask (.constBool false)) .unspecified)]

example (j : PreparedAlgorithmInvocation) (s : BlockState) : (j.initial s).mem=s.mem := rfl

/-- Preserving the input register executes; clearing it fails at the read. -/
example : ((job copyReg).run seed).map (fun s=>s.regs .real [] "result")=
    some (some (Tile.scalar (some (3 : ℝ)))) := by
  simp [PreparedAlgorithmInvocation.run,PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,
    job,source,copyReg,seed,ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,BlockState.setReg]
example : (job copyReg true).run seed=none := by
  simp [PreparedAlgorithmInvocation.run,PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,
    job,source,copyReg,seed,ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,BlockState.setReg]

/-- The omitted masked-off load observes preserved nonzero undef or selected zero. -/
example : ((job omitted).run seed).map (fun s=>s.regs .real [] "result")=
    some (some (Tile.scalar (some (7 : ℝ)))) := by
  simp [PreparedAlgorithmInvocation.run,PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,
    job,source,omitted,seed,ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,BlockState.setReg]
example : ((job omitted false true).run seed).map (fun s=>s.regs .real [] "result")=
    some (some (Tile.scalar (some (0 : ℝ)))) := by
  simp [PreparedAlgorithmInvocation.run,PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,
    job,source,omitted,seed,ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,BlockState.setReg]

def write : List Stmt := [.store .real [] (.region "x" (.constNat 0)) (.const 9) .none]
def read : List Stmt := [.assign .real [] "result" (.load .real (.region "x" (.constNat 0)) .none)]
/-- The second stage reads the first stage's memory, even with fresh registers. -/
example : (runPreparedAlgorithmSchedule [job write true true,job read true true] seed).map
    (fun s=>s.regs .real [] "result")=some (some (Tile.scalar (some (9 : ℝ)))) := by
  simp [runPreparedAlgorithmSchedule,PreparedAlgorithmInvocation.run,PreparedAlgorithmInvocation.initial,
    AlgorithmLaunch.initial,job,source,write,read,seed,ComputeKernel.toAlgKernel_fromKernelBody,
    exec,stepStmts,stepStmt,evalOp,BlockState.setReg,TileShape.allIndices,BlockState.writeMemTyped,
    BlockState.readMemTyped,BlockState.readMem,BlockState.writeMem,MemCell.readAs,MemCell.real,MemCell.of]

def emptyIO (j : PreparedAlgorithmInvocation) : PreparedAlgorithmProgramIO where
  nIn:=0
  nOut:=0
  nScr:=0
  bufs:=[]
  ity:=fun i=>Fin.elim0 i
  iarity:=fun i=>Fin.elim0 i
  ibuf:=fun i=>Fin.elim0 i
  oty:=fun i=>Fin.elim0 i
  oarity:=fun i=>Fin.elim0 i
  obuf:=fun i=>Fin.elim0 i
  obuf_mem:=fun i=>Fin.elim0 i
  sarity:=fun i=>Fin.elim0 i
  sbuf:=fun i=>Fin.elim0 i
  iwin:=fun i=>Fin.elim0 i
  imask:=fun i=>Fin.elim0 i
  owin:=fun i=>Fin.elim0 i
  omask:=fun i=>Fin.elim0 i
  swin:=fun i=>Fin.elim0 i
  smask:=fun i=>Fin.elim0 i
  schedule:=fun _=>[j]
def emptyResult (j : PreparedAlgorithmInvocation) : (emptyIO j).Ctx →
    (o : Fin (emptyIO j).nOut) → Fin ((emptyIO j).oarity o) → ((emptyIO j).oty o).carrier :=
  fun _ i=>Fin.elim0 i

theorem failed_execution_rejected : ¬(emptyIO (job copyReg true)).Implements (emptyResult _) := by
  intro h
  obtain ⟨_,final,ran,_,_⟩:=h (fun i=>Fin.elim0 i) trivial seed (fun i=>Fin.elim0 i)
  simpa [emptyIO,runPreparedAlgorithmSchedule,PreparedAlgorithmInvocation.run,
    PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,job,source,copyReg,seed,
    ComputeKernel.toAlgKernel_fromKernelBody,exec,stepStmts,stepStmt,evalOp,BlockState.setReg] using ran

theorem unbacked_read_rejected : ¬(emptyIO (job read)).Implements (emptyResult _) := by
  intro h
  have safe:=(h (fun i=>Fin.elim0 i) trivial seed (fun i=>Fin.elim0 i)).1 (fun _=>0)
    ⟨fun i=>Fin.elim0 i,fun i=>Fin.elim0 i,fun i=>Fin.elim0 i⟩
  have access:=safe.1
  simp [PreparedAlgorithmInvocation.Safe,Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt,
    PreparedAlgorithmInvocation.initial,AlgorithmLaunch.initial,job,source,read,
    ComputeKernel.toAlgKernel_fromKernelBody,evalOp,MemAccess.ActiveAddressSafe,MaskOpt.Active,memAccessActiveAddressSafe] at access

#axiomsClean failed_execution_rejected
#axiomsClean unbacked_read_rejected
end
end PreparedAlgorithmRegression
