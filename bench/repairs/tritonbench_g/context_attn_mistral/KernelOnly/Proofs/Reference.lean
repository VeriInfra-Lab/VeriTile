import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.FPSymbolic
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.FPBackend

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula
open VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program (Regions)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev program (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) :=
  VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel p r storage Bool.false scale

def parts (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (program p r storage scale).surfaceBody "start_n" with
  | some q => q
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (p : Layout) (r : Regions) (scale : ℝ) :
    (program p r .fp32 scale).surfaceBody=(parts p r .fp32 scale).before ++
      [.forRangeInt "start_n" (parts p r .fp32 scale).start (parts p r .fp32 scale).stop
        (parts p r .fp32 scale).stride (parts p r .fp32 scale).body] ++
      (parts p r .fp32 scale).after := rfl

@[simp] theorem region_eta (r : RegionName) : (⟨r.name⟩ : RegionName)=r := rfl
@[simp] theorem unit_index (i : TileIndex []) : i=PUnit.unit := Subsingleton.elim _ _

/-- Initial input arrays and scalar launch metadata only. -/
structure Inputs (α : Type) where
  (query key value : Nat → α)
  (head kvHead block : Nat)
  (start length : Int)

def inputs (p : Layout) (r : Regions) (s : State α) : Inputs α where
  query := fun off => (s.mem r.query off).read .real
  key := fun off => (s.mem r.key off).read .real
  value := fun off => (s.mem r.value off).read .real
  head := s.pids 1
  kvHead := s.pids 1/p.kvGroup
  block := s.pids 2
  start := (s.mem r.start (s.pids 0)).read .int
  length := (s.mem r.sequence (s.pids 0)).read .int

def row (p : Layout) (a : Inputs α) (i : Nat) := a.block*p.BM+i
def blockMask (p : Layout) (a : Inputs α) : Nat := if (a.block*p.BM : Int)<a.length then 1 else 0
def stop (p : Layout) (a : Inputs α) : Int := (blockMask p a : Int)*((a.block : Int)+1)*p.BM

def queryTile (p : Layout) (o : FPBackend.Ops α) (a : Inputs α) : Values α .real [p.BM,p.D] :=
  fun i => o.scalar.fp32Load (if (row p a i.1.val : Int)<a.length then
    a.query (((a.start+row p a i.1.val)*p.qRow+(a.head*p.qHead : Nat)+(i.2.1.val*p.qDim : Nat)).toNat)
    else o.scalar.literal (some .fp32) .real 0.0)

def keyPointer (p : Layout) (r : Regions) (a : Inputs α) : Values α .ptr [p.D,p.BN] :=
  fun i => ⟨r.key,i.2.1.val*p.kRow+a.kvHead*p.kHead+i.1.val*p.kDim⟩

def valuePointer (p : Layout) (r : Regions) (a : Inputs α) : Values α .ptr [p.BN,p.D] :=
  fun i => ⟨r.value,i.1.val*p.vRow+a.kvHead*p.vHead+i.2.1.val*p.vDim⟩

def keyTile (p : Layout) (o : FPBackend.Ops α) (a : Inputs α) (cur : Int) : Values α .real [p.D,p.BN] :=
  fun i => o.scalar.fp32Load (if cur+i.2.1.val<a.length then
    a.key (i.2.1.val*p.kRow+a.kvHead*p.kHead+i.1.val*p.kDim+((a.start+cur)*p.kRow).toNat)
    else o.scalar.literal (some .fp32) .real 0.0)

def valueTile (p : Layout) (o : FPBackend.Ops α) (a : Inputs α) (cur : Int) : Values α .real [p.BN,p.D] :=
  fun i => o.scalar.fp32Load (if cur+i.1.val<a.length then
    a.value (i.1.val*p.vRow+a.kvHead*p.vHead+i.2.1.val*p.vDim+((a.start+cur)*p.vRow).toNat)
    else o.scalar.literal (some .fp32) .real 0.0)

structure Acc (p : Layout) (α : Type) where
  (center mass : Values α .real [p.BM])
  weighted : Values α .real [p.BM,p.D]

def initial (p : Layout) (o : FPBackend.Ops α) : Acc p α where
  center := fun _ => o.scalar.binary (some .fp32) .real .add (o.scalar.literal (some .fp32) .real 0) o.scalar.negInf
  mass := fun _ => o.scalar.literal (some .fp32) .real 0
  weighted := fun _ => o.scalar.literal (some .fp32) .real 0

/-- Independent online recurrence with finite masks, both scalar guards,
the original probability cast, live accumulator, and ordered reductions. -/
def advance (p : Layout) (o : FPBackend.Ops α) (scale : ℝ)
    (a : Inputs α) (cur : Int) (old : Acc p α) : Acc p α :=
  let M := o.scalar
  let raw := o.dot (some .fp32) (batch:=[])
    (m:=p.BM) (k:=p.D) (n:=p.BN) (queryTile p o a) (keyTile p o a cur)
  let scores : Values α .real [p.BM,p.BN] := fun i =>
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) (raw i))
      (M.literal (some .fp32) .real scale)
  let causal : Values α .real [p.BM,p.BN] := fun i =>
    if cur+i.2.1.val≤(row p a i.1.val : Int) then scores i else M.literal (some .fp32) .real (-1e9)
  let masked : Values α .real [p.BM,p.BN] := fun i =>
    if cur+i.2.1.val>(row p a i.1.val : Int)-p.window then causal i else M.literal (some .fp32) .real (-1e9)
  let reduced := M.reduceMax (some .fp32) (shape:=[p.BM,p.BN]) ⟨1,by simp⟩ Bool.false masked
  let localCenter : Values α .real [p.BM] := fun i =>
    if o.eq (some .fp32) .real (reduced i) (M.literal (some .fp32) .real (-1e9))
      then M.literal (some .fp32) .real 0.0 else reduced i
  let probabilities : Values α .real [p.BM,p.BN] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (masked i) (localCenter (i.1,())))
  let sums := M.reduceSum (some .fp32) (shape:=[p.BM,p.BN]) ⟨1,by simp⟩ Bool.false probabilities
  let centers : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .max (old.center i) (localCenter i)
  let alpha : Values α .real [p.BM] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (old.center i) (centers i))
  let beta : Values α .real [p.BM] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (localCenter i) (centers i))
  let unguarded : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (alpha i) (old.mass i))
    (M.binary (some .fp32) .real .mul (beta i) (sums i))
  let mass : Values α .real [p.BM] := fun i =>
    if o.eq (some .fp32) .real (unguarded i) (M.literal (some .fp32) .real 0.0)
      then M.literal (some .fp32) .real 1e-9 else unguarded i
  let scaled : Values α .real [p.BM,p.BN] := fun i => M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (probabilities i)
      (M.binary (some .fp32) .real .div (beta (i.1,())) (mass (i.1,()))))
  let factor : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .div (old.mass i) (mass i)) (alpha i)
  let contribution := o.dot (some .fp32) (batch:=[])
    (m:=p.BM) (k:=p.BN) (n:=p.D) scaled (valueTile p o a cur)
  ⟨centers,mass,fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (old.weighted i) (factor (i.1,()))) (contribution i)⟩

def iterate (p : Layout) (o : FPBackend.Ops α) (scale : ℝ)
    (a : Inputs α) (cur endKey : Int) (old : Acc p α) : Acc p α :=
  if hz : p.BN=0 then old else
  if hc : cur<endKey then iterate p o scale a (cur+p.BN) endKey (advance p o scale a cur old)
  else old
termination_by (endKey-cur).toNat
decreasing_by omega

def output (p : Layout) (o : FPBackend.Ops α) (scale : ℝ) (a : Inputs α) : Values α .real [p.BM,p.D] :=
  let final := iterate p o scale a 0 (stop p a) (initial p o)
  fun i => o.scalar.cast (some .fp32) .real .real (final.weighted i)

def Matches (p : Layout) (r : Regions) (o : FPBackend.Ops α) (a : Inputs α) (acc : Acc p α) (s : State α) : Prop :=
  s.regs .nat [] "cur_head"=some (fun _ => a.head) ∧
  s.regs .nat [] "start_m"=some (fun _ => a.block) ∧
  s.regs .int [] "cur_batch_in_all_start_index"=some (fun _ => a.start) ∧
  s.regs .int [] "cur_batch_seq_len"=some (fun _ => a.length) ∧
  s.regs .nat [p.BM] "offs_m"=some (fun i => row p a i.1.val) ∧
  s.regs .nat [p.BN] "offs_n"=some (fun i => i.1.val) ∧
  s.regs .nat [p.D] "offs_d"=some (fun i => i.1.val) ∧
  s.regs .real [p.BM,p.D] "q"=some (queryTile p o a) ∧
  s.regs .ptr [p.D,p.BN] "k_ptrs"=some (keyPointer p r a) ∧
  s.regs .ptr [p.BN,p.D] "v_ptrs"=some (valuePointer p r a) ∧
  s.regs .real [p.BM] "m_i"=some acc.center ∧
  s.regs .real [p.BM] "l_i"=some acc.mass ∧
  s.regs .real [p.BM,p.D] "acc"=some acc.weighted ∧
  s.regs .nat [] "block_mask"=some (fun _ => blockMask p a)

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α) (s : State α) :
    ∃ t,run (engine o.algebra) (parts p r .fp32 scale).before s=some t ∧
      Matches p r o (inputs p r s) (initial p o) t ∧ t.mem=s.mem := by
  simp (config := {maxSteps := 500000}) [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Matches,inputs,initial,stop,blockMask,queryTile,keyPointer,valuePointer,row,bop]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,queryTile,keyPointer,valuePointer,row,Int.mul_comm]

end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
