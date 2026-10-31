import bench.tritonbench_g.context_attn_nopad.FPSymbolic

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPReference
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Formula Program
open FPSymbolic (parts)
open FP.ProfiledRewrite (engine)
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem unit_index (i : TileIndex []) : i=PUnit.unit := Subsingleton.elim _ _

structure Inputs (α : Type) where
  (query key value : Nat → α)
  (head block : Nat)
  (start length : Int)

def inputs (r : Regions) (s : State α) : Inputs α where
  query := fun off => (s.mem r.query off).read .real
  key := fun off => (s.mem r.key off).read .real
  value := fun off => (s.mem r.value off).read .real
  head := s.pids 1
  block := s.pids 2
  start := (s.mem r.start (s.pids 0)).read .int
  length := (s.mem r.sequence (s.pids 0)).read .int

def row (p : Layout) (a : Inputs α) (i : Nat) := a.block*p.BM+i

def blockMask (p : Layout) (a : Inputs α) : Nat := if (p.BM*a.block : Int) < a.length then 1 else 0

def queryTile (p : Layout) (M : Algebra α) (a : Inputs α) : Values α .real [p.BM,p.D] :=
  fun i => M.fp32Load (if (row p a i.1.val : Int) < a.length then
    a.query (((a.start+row p a i.1.val)*p.qRow+(a.head*p.qHead : Nat)+(i.2.1.val*p.qDim : Nat)).toNat)
    else M.literal (some .fp32) .real 0.0)

def keyTile (p : Layout) (M : Algebra α) (a : Inputs α) (cur : Int) : Values α .real [p.D,p.BN] :=
  fun i => M.fp32Load (if cur+i.2.1.val < a.length then
    a.key (i.2.1.val*p.kRow+a.head*p.kHead+i.1.val*p.kDim+((a.start+cur)*p.kRow).toNat)
    else M.literal (some .fp32) .real 0.0)

def valueTile (p : Layout) (M : Algebra α) (a : Inputs α) (cur : Int) : Values α .real [p.BN,p.D] :=
  fun i => M.fp32Load (if cur+i.1.val < a.length then
    a.value (i.1.val*p.vRow+a.head*p.vHead+i.2.1.val*p.vDim+((a.start+cur)*p.vRow).toNat)
    else M.literal (some .fp32) .real 0.0)

structure Acc (p : Layout) (α : Type) where
  (center mass : Values α .real [p.BM])
  weighted : Values α .real [p.BM,p.D]

def initial (p : Layout) (M : Algebra α) : Acc p α where
  center := fun _ => M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) M.negInf
  mass := fun _ => M.literal (some .fp32) .real 0
  weighted := fun _ => M.literal (some .fp32) .real 0

/-- Independent normalized online attention recurrence with explicit source
rounding locations, probability cast, negative-infinity mask and dot seeds. -/
def advance (p : Layout) (o : FPBackend.Ops α) (scale : ℝ)
    (a : Inputs α) (cur : Int) (old : Acc p α) : Acc p α :=
  let M := o.algebra
  let scores := o.dot (some .fp32) (batch := []) (queryTile p M a) (keyTile p M a cur)
  let masked : Values α .real [p.BM,p.BN] := fun i =>
    if cur+i.2.1.val ≤ (row p a i.1.val : Int) then
      M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) (scores i))
        (M.literal (some .fp32) .real scale)
    else M.negInf
  let tileCenter := M.reduceMax (some .fp32) (shape := [p.BM,p.BN]) ⟨1,by simp⟩ Bool.false masked
  let probability : Values α .real [p.BM,p.BN] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (masked i) (tileCenter (i.1,())))
  let tileMass := M.reduceSum (some .fp32) (shape := [p.BM,p.BN]) ⟨1,by simp⟩ Bool.false probability
  let center : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .max (old.center i) (tileCenter i)
  let alpha : Values α .real [p.BM] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (old.center i) (center i))
  let beta : Values α .real [p.BM] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (tileCenter i) (center i))
  let mass : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (alpha i) (old.mass i))
    (M.binary (some .fp32) .real .mul (beta i) (tileMass i))
  let castP : Values α .real [p.BM,p.BN] := fun i => M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (probability i)
      (M.binary (some .fp32) .real .div (beta (i.1,())) (mass (i.1,()))))
  let pv := o.dot (some .fp32) (batch := []) castP (valueTile p M a cur)
  { center := center
    mass := mass
    weighted := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.weighted i)
        (M.binary (some .fp32) .real .mul
          (M.binary (some .fp32) .real .div (old.mass (i.1,())) (mass (i.1,()))) (alpha (i.1,()))))
      (pv i) }

def iterate (p : Layout) (o : FPBackend.Ops α) (scale : ℝ)
    (a : Inputs α) (cur stop : Int) (old : Acc p α) : Acc p α :=
  if hz : p.BN=0 then old else
  if hc : cur < stop then iterate p o scale a (cur+p.BN) stop (advance p o scale a cur old)
  else old
termination_by (stop-cur).toNat
decreasing_by omega

def output (p : Layout) (o : FPBackend.Ops α) (scale : ℝ) (a : Inputs α) : Values α .real [p.BM,p.D] :=
  let final := iterate p o scale a 0 ((blockMask p a*(a.block+1)*p.BM : Nat) : Int) (initial p o.algebra)
  fun i => o.algebra.cast (some .fp32) .real .real (final.weighted i)

def Matches (p : Layout) (r : Regions) (M : Algebra α) (a : Inputs α) (acc : Acc p α) (s : State α) : Prop :=
  s.regs .nat [] "cur_head"=some (fun _ => a.head) ∧
  s.regs .nat [] "start_m"=some (fun _ => a.block) ∧
  s.regs .int [] "cur_batch_in_all_start_index"=some (fun _ => a.start) ∧
  s.regs .int [] "cur_batch_seq_len"=some (fun _ => a.length) ∧
  s.regs .nat [p.BM] "offs_m"=some (fun i => row p a i.1.val) ∧
  s.regs .nat [p.BN] "offs_n"=some (fun i => i.1.val) ∧
  s.regs .nat [p.D] "offs_d"=some (fun i => i.1.val) ∧
  s.regs .real [p.BM,p.D] "q"=some (queryTile p M a) ∧
  s.regs .ptr [p.D,p.BN] "k_ptrs"=some (fun i => (r.key,i.2.1.val*p.kRow+a.head*p.kHead+i.1.val*p.kDim)) ∧
  s.regs .ptr [p.BN,p.D] "v_ptrs"=some (fun i => (r.value,i.1.val*p.vRow+a.head*p.vHead+i.2.1.val*p.vDim)) ∧
  s.regs .real [p.BM] "m_i"=some acc.center ∧
  s.regs .real [p.BM] "l_i"=some acc.mass ∧
  s.regs .real [p.BM,p.D] "acc"=some acc.weighted ∧
  s.regs .nat [] "block_mask"=some (fun _ => blockMask p a)

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α) (s : State α) :
    ∃ t,run (engine o.algebra) (parts p r scale).before s=some t ∧
      Matches p r o.algebra (inputs r s) (initial p o.algebra) t ∧ t.mem=s.mem := by
  simp (config := {maxSteps := 500000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Matches,inputs,initial,blockMask,queryTile,row,bop]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,queryTile,row]
  all_goals rfl

theorem body (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0 < p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o.algebra (inputs r initialState) old s) (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (engine o.algebra) (parts p r scale).body
        (s.setReg "start_n" .int [] (fun _ => cur))=some t ∧
      Matches p r o.algebra (inputs r initialState) (advance p o scale (inputs r initialState) cur old) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp (config := {maxSteps := 1400000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,
    TileShape.axisDim,hBN,Matches,advance,keyTile,valueTile,
    memory,inputs,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,keyTile,valueTile,Function.comp_def]
  refine ⟨?_,?_,?_⟩ <;> intros <;> rfl

end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPReference
