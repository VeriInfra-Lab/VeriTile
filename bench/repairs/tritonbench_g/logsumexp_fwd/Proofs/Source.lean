import bench.repairs.tritonbench_g.logsumexp_fwd.Kernels

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Source
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def inputAddress (row block D B lane : Nat) : Nat := row*D+(block*B+lane)
def outputAddress (row block D B : Nat) : Nat := row*((D+B-1)/B)+block

/-- Exact eager source graph: masked loads, optional FP32 multiplication,
mask restoration, actual max and sum reductions, infinite-shift selection,
exp/sub/log/add in their original order. -/
def rowValue [Inhabited A] (M : Algebra A) (equal : A → A → Bool)
    (input : Nat → A) (scale : A) (row block D B : Nat) (hasScale : Bool) : A :=
  let loaded : TileIndex [B] → A := fun i => M.fp32Load
    (if block*B+i.1.val<D then input (inputAddress row block D B i.1.val) else M.negInf)
  let scaled := if hasScale then fun i => M.binary (some .fp32) .real .mul (loaded i) scale else loaded
  let restored := fun i : TileIndex [B] => if block*B+i.1.val<D then scaled i else M.negInf
  let maximum := M.reduceMax (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false restored PUnit.unit
  let shift := if equal maximum (M.fp32Bits ⟨BitVec.ofNat 32 2139095040⟩) || equal maximum M.negInf
    then M.literal (some .fp32) .real 0.0 else maximum
  let exponentials := fun i => M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (restored i) shift)
  let total := M.reduceSum (some .fp32) (shape := [B]) ⟨0,by simp⟩ Bool.false exponentials PUnit.unit
  M.binary (some .fp32) .real .add (M.unary (some .fp32) .log total) shift

theorem row_exec [Inhabited A] (M : Algebra A) (equal : A → A → Bool)
    (comparisons : ∀ p d,M.compareEq p d=some equal)
    (x z : RegionName) (D B : Nat) (hasScale : Bool) (scale : A)
    (s : State A) (hB : 0<B) (parameter : s.regs .real [] "scale"=some (fun _ => scale)) :
    ∃ final,FP.Structural.exec M (rowBlock x z D B hasScale) s=some final ∧
      final.mem=(s.write z (outputAddress (s.pids 0) (s.pids 1) D B)
        (.mk .real (rowValue M equal (fun i => (s.mem x i).read .real) scale
          (s.pids 0) (s.pids 1) D B hasScale))).mem := by
  have hsum : 1≤D+B := by omega
  cases hasScale
  all_goals simp [rowBlock,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,store,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,numericEq,Region.cast,rowValue,inputAddress,outputAddress,
    TileShape.allIndices,TileShape.axisDim,hB,State.write,State.setReg,parameter,comparisons]
  all_goals simp only [← Int.natCast_one,← Int.natCast_sub hsum,← Int.natCast_add,
    ← Int.natCast_mul,← Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]
  all_goals simp [bop,Function.comp_def]
  all_goals rfl

#axiomsClean row_exec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Source
