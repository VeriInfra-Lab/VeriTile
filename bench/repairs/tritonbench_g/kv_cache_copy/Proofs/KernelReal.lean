import bench.repairs.tritonbench_g.kv_cache_copy.Kernels
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Semantics.Integer
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
open VeriTile.Triton Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def position (s : BlockState) : Int := s.readMemValue .int "lengths" (s.pids 0)-1
def selectedBlock (p : Parameters) (s : BlockState) : Int :=
  s.readMemValue .int "table" (s.pids 0*p.strideBts+((position s).tdiv p.blockSize*p.strideBtb).toNat)
def sourceOffset (p : Parameters) (ids : Nat → Nat) (key : Bool) (q : Nat)
    (i : TileIndex [p.width]) : Nat :=
  if key then ids 0*p.strideKt+ids 1*p.strideKh+(q*p.width+i.1.val)*p.strideKd
  else ids 0*p.strideVt+ids 1*p.strideVh+(q*p.width+i.1.val)*p.strideVd

def targetInt (p : Parameters) (s : BlockState) (key : Bool)
    (q : Nat) (i : TileIndex [p.width]) : Int :=
  if key then selectedBlock p s*p.strideKcb+(s.pids 1:Int)*p.strideKch+
    (q:Int)*p.strideKcsplit+(position s).tmod p.blockSize*p.strideKcs+i.1.val
  else selectedBlock p s*p.strideVcb+(s.pids 1:Int)*p.strideVch+
    (position s).tmod p.blockSize*p.strideVcs+(q*p.width+i.1.val)*p.strideVcd

def outputRegion (key : Bool) : RegionName := if key then "keyCache" else "valueCache"
def inputRegion (key : Bool) : RegionName := if key then "key" else "value"

def scatterOne (p : Parameters) (s : BlockState) (key : Bool) (q : Nat) (t : BlockState) : BlockState :=
  (TileShape.allIndices [p.width]).foldl (fun z i =>
    z.writeMem (outputRegion key) (targetInt p s key q i).toNat
      (s.readMem (inputRegion key) (sourceOffset p s.pids key q i))) t

def copyStep (p : Parameters) (s : BlockState) (q : Nat) (t : BlockState) : BlockState :=
  scatterOne p s Bool.false q (scatterOne p s Bool.true q t)

def copied (p : Parameters) (s : BlockState) : Nat → BlockState
  | 0 => s
  | q+1 => copyStep p s q (copied p s q)

def preludeState (p : Parameters) (s : BlockState) : BlockState :=
  let z := s.setReg "cur_seq_idx" .nat [] (Tile.scalar (s.pids 0))
  let z := z.setReg "cur_kv_head_idx" .nat [] (Tile.scalar (s.pids 1))
  let z := z.setReg "past_kv_seq_len" .int [] (Tile.scalar (position s))
  let z := z.setReg "last_bt_block_idx" .int [] (Tile.scalar ((position s).tdiv p.blockSize))
  let z := z.setReg "block_table_ptr" .ptr [] (Tile.scalar ("table",s.pids 0*p.strideBts))
  let z := z.setReg "block_id" .int [] (Tile.scalar (selectedBlock p s))
  let z := z.setReg "offsets_in_last_block" .int [] (Tile.scalar ((position s).tmod p.blockSize))
  let z := z.setReg "range_x" .nat [p.width] ⟨fun i => i.1.val⟩
  z.setReg "offsets_dmodel_x_partition" .nat [p.width] ⟨fun i => i.1.val⟩

structure Controls (p : Parameters) (s t : BlockState) : Prop where
  seq : t.regs .nat [] "cur_seq_idx"=some (Tile.scalar (s.pids 0))
  head : t.regs .nat [] "cur_kv_head_idx"=some (Tile.scalar (s.pids 1))
  block : t.regs .int [] "block_id"=some (Tile.scalar (selectedBlock p s))
  slot : t.regs .int [] "offsets_in_last_block"=some (Tile.scalar ((position s).tmod p.blockSize))
  lanes : t.regs .nat [p.width] "range_x"=some ⟨fun i => i.1.val⟩

def Frozen (s t : BlockState) : Prop := t.mem "key"=s.mem "key" ∧ t.mem "value"=s.mem "value"

def preparedState (p : Parameters) (s : BlockState) (q : Nat) (t : BlockState) : BlockState :=
  let z := t.setReg "split_x" .nat [] (Tile.scalar q)
  let z := z.setReg "offsets_dmodel_x_partition" .nat [p.width] ⟨fun i => q*p.width+i.1.val⟩
  let z := z.setReg "offsets_k" .nat [p.width] ⟨sourceOffset p s.pids Bool.true q⟩
  let z := z.setReg "k" .real [p.width] ⟨fun i => some (s.readMem "key" (sourceOffset p s.pids Bool.true q i))⟩
  let z := z.setReg "offsets_v" .nat [p.width] ⟨sourceOffset p s.pids Bool.false q⟩
  let z := z.setReg "v" .real [p.width] ⟨fun i => some (s.readMem "value" (sourceOffset p s.pids Bool.false q i))⟩
  z.setReg "offsets_kcache" .int [p.width] ⟨targetInt p s Bool.true q⟩

def finalStep (p : Parameters) (s : BlockState) (q : Nat) (t : BlockState) : BlockState :=
  let z := scatterOne p s Bool.true q (preparedState p s q t)
  let z := z.setReg "offsets_vcache" .int [p.width] ⟨targetInt p s Bool.false q⟩
  scatterOne p s Bool.false q z

def source (p : Parameters) := (originalKernel p).toAlgKernel

theorem projection (p : Parameters) : (originalKernel p).toAlgorithm? = .ok (source p) := by
  simp [source,originalKernel,kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals rfl

def prelude (p : Parameters) := (source p).body.take 9
def body (p : Parameters) := match (source p).body[9]? with
  | some (.forLoop _ _ body) => body
  | _ => []

theorem decomposition (p : Parameters) :
    (source p).body=prelude p ++ [.forLoop "split_x" (p.dims/p.width) (body p)] := rfl

theorem prelude_run (p : Parameters) (s : BlockState) :
    stepStmts (prelude p) s=some (preludeState p s) := by
  simp [prelude,preludeState,selectedBlock,position,source,originalKernel,kernel,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Region.cast,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.uop,Tile.cop,Tile.select,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.add,NumericDType.mul,NumericDType.sub,IntegralDType.floorDiv,IntegralDType.mod,ComparableDType.lt,
    VeriTile.Triton.eval_truncDiv,VeriTile.Triton.eval_truncMod,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp [Tile.bop,Tile.uop,Tile.cop,Tile.select,NumericDType.add,NumericDType.mul,NumericDType.sub,
    ComparableDType.lt,Broadcast.leftIndex,Broadcast.rightIndex,←IntegerExpr.tdiv_formula,Int.tmod_def]
  all_goals rfl

theorem body_run (p : Parameters) (s t : BlockState) (q : Nat)
    (hc : Controls p s t) (hf : Frozen s t) :
    stepStmts (body p) (t.setReg "split_x" .nat [] (Tile.scalar q))=some (finalStep p s q t) := by
  rcases hc with ⟨hs,hh,hb,hslot,hl⟩
  rcases hf with ⟨hk,hv⟩
  simp [body,finalStep,preparedState,scatterOne,sourceOffset,targetInt,outputRegion,inputRegion,
    source,originalKernel,kernel,ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Region.cast,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.uop,NumericDType.add,NumericDType.mul,NumericDType.sub,
    Broadcast.leftIndex,Broadcast.rightIndex,hs,hh,hb,hslot,hl,hk,hv,BlockState.readMemValue,BlockState.readMem]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals congr 20
  all_goals funext
  all_goals try simp [Tile.bop,Tile.uop,NumericDType.add,NumericDType.mul,sourceOffset,targetInt,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  all_goals ring

#axiomsClean projection
#axiomsClean prelude_run
#axiomsClean body_run
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
