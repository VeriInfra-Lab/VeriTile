import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPTotalBackend
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPGrid

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
open VeriTile.Triton FP.Structural FPSuccess
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem unit_index (i : TileIndex []) : i=PUnit.unit := Subsingleton.elim _ _

/-- Only initial input arrays and scalar launch metadata enter the reference.
There is no source AST, execution, final register or returned result here. -/
structure Inputs (α : Type) where
  (query key value : Nat → α)
  (batch head kvHead block : Nat)
  pages : Nat → Int
  (start length prompt request : Int)

def inputs (p : Layout) (r : Regions) (s : State α) : Inputs α where
  query := fun off => (s.mem r.query off).read .real
  key := fun off => (s.mem r.key off).read .real
  value := fun off => (s.mem r.value off).read .real
  pages := fun off => (s.mem r.pages off).read .int
  request := (s.mem r.request (s.pids 1 / p.heads)).read .int
  batch := s.pids 1 / p.heads
  head := s.pids 1 % p.heads
  kvHead := (s.pids 1 % p.heads) / p.kvGroup
  block := s.pids 0
  start := (s.mem r.start (s.pids 1 / p.heads)).read .int
  length := (s.mem r.sequence (s.pids 1 / p.heads)).read .int -
    (s.mem r.prompt (s.pids 1 / p.heads)).read .int
  prompt := (s.mem r.prompt (s.pids 1 / p.heads)).read .int

def row (p : Layout) (a : Inputs α) (i : Nat) := p.BM*a.block+i

def keyEnd (p : Layout) (a : Inputs α) : Int :=
  min ((p.BM*a.block+p.BM : Nat)+a.prompt) (a.length+a.prompt)

def blockMask (p : Layout) (a : Inputs α) : Nat := if (p.BM*a.block : Int) < a.length then 1 else 0

def queryTile (p : Layout) (M : Algebra α) (a : Inputs α) : Values α .real [p.BM,p.D] :=
  fun i => M.fp32Load (if (row p a i.1.val : Int) < a.length then
    a.query (((a.start+row p a i.1.val)*p.qRow+(a.head*p.qHead : Nat)+(i.2.1.val*p.qDim : Nat)).toNat)
    else M.literal (some .fp32) .real 0.0)

def page (p : Layout) (a : Inputs α) (cur : Int) (lane : Nat) : Int :=
  if cur+lane < keyEnd p a then
    a.pages (((p.reqRow : Int)*a.request+(p.reqToken : Int)*(cur+lane)).toNat) else 0

def keyTile (p : Layout) (M : Algebra α) (a : Inputs α) (cur : Int) : Values α .real [p.D,p.BN] :=
  fun i => M.fp32Load (if cur+i.2.1.val < keyEnd p a then
    a.key (((page p a cur i.2.1.val)*p.kRow+
      (a.kvHead*p.kHead : Nat)+(i.1.val*p.kDim : Nat)).toNat)
    else M.literal (some .fp32) .real 0.0)

def valueTile (p : Layout) (M : Algebra α) (a : Inputs α) (cur : Int) : Values α .real [p.BN,p.D] :=
  fun i => M.fp32Load (if cur+i.1.val < keyEnd p a then
    a.value (((page p a cur i.1.val)*p.vRow+
      (a.kvHead*p.vHead : Nat)+(i.2.1.val*p.vDim : Nat)).toNat)
    else M.literal (some .fp32) .real 0.0)

structure Acc (p : Layout) (α : Type) where
  (center mass : Values α .real [p.BM])
  weighted : Values α .real [p.BM,p.D]

def initial (p : Layout) (M : Algebra α) : Acc p α where
  center := fun _ => M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) M.negInf
  mass := fun _ => M.literal (some .fp32) .real 0
  weighted := fun _ => M.literal (some .fp32) .real 0

/-- Independent tile recurrence. The probability cast, the live PV dot seed,
finite sentinel, ordered max/sum callbacks and each scalar operation are explicit. -/
def advance (p : Layout) (M : Algebra α) (dots : DotSupport p M) (scale : ℝ)
    (a : Inputs α) (cur : Int) (old : Acc p α) : Acc p α :=
  let scores := dots.qk (queryTile p M a) (keyTile p M a cur)
  let masked : Values α .real [p.BM,p.BN] := fun i =>
    if cur+i.2.1.val ≤ (row p a i.1.val : Int)+a.prompt then
      M.binary (some .fp32) .real .mul (scores i) (M.literal (some .fp32) .real scale)
    else M.literal (some .fp32) .real (-1.0e8)
  let centers : Values α .real [p.BM] := fun i => M.binary (some .fp32) .real .max
    (old.center i) (M.reduceMax (some .fp32) (shape := [p.BM,p.BN]) ⟨1,by simp⟩ Bool.false masked i)
  let probabilities : Values α .real [p.BM,p.BN] := fun i => M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (masked i) (centers (i.1,())))
  let sums := M.reduceSum (some .fp32) (shape := [p.BM,p.BN]) ⟨1,by simp⟩ Bool.false probabilities
  let factor : Values α .real [p.BM] := fun i => M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (old.center i) (centers i))
  { center := centers
    mass := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.mass i) (factor i)) (sums i)
    weighted := dots.pv (fun i => M.cast (some .fp32) .real .real (probabilities i))
      (valueTile p M a cur) (fun i => M.binary (some .fp32) .real .mul
        (old.weighted i) (factor (i.1,()))) }

/-- Recursion over key tiles, independent of the typed source interpreter. -/
def iterate (p : Layout) (M : Algebra α) (dots : DotSupport p M) (scale : ℝ)
    (a : Inputs α) (cur stop : Int) (old : Acc p α) : Acc p α :=
  if hz : p.BN = 0 then old else
  if hc : cur < stop then iterate p M dots scale a (cur+p.BN) stop (advance p M dots scale a cur old)
  else old
termination_by (stop-cur).toNat
decreasing_by omega

def output (p : Layout) (M : Algebra α) (dots : DotSupport p M) (scale : ℝ)
    (a : Inputs α) : Values α .real [p.BM,p.D] :=
  let final := iterate p M dots scale a 0 ((blockMask p a : Int)*keyEnd p a) (initial p M)
  fun i => M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .div (final.weighted i) (final.mass (i.1,())))

/-- The exact loop live-register invariant connecting source to recurrence. -/
def Matches (p : Layout) (M : Algebra α) (a : Inputs α) (acc : Acc p α) (s : State α) : Prop :=
  s.regs .nat [] "cur_kv_head"=some (fun _ => a.kvHead) ∧
  s.regs .int [] "cur_batch_req_idx"=some (fun _ => a.request) ∧
  s.regs .int [] "block_end_loc"=some (fun _ => keyEnd p a) ∧
  s.regs .nat [p.BN] "offs_n"=some (fun i => i.1.val) ∧
  s.regs .nat [p.D] "offs_d"=some (fun i => i.1.val) ∧
  s.regs .nat [p.BM] "offs_m"=some (fun i => row p a i.1.val) ∧
  s.regs .int [] "prompt_cache_len"=some (fun _ => a.prompt) ∧
  s.regs .real [p.BM,p.D] "q"=some (queryTile p M a) ∧
  s.regs .real [p.BM] "m_i"=some acc.center ∧
  s.regs .real [p.BM] "l_i"=some acc.mass ∧
  s.regs .real [p.BM,p.D] "acc"=some acc.weighted ∧
  s.regs .int [] "cur_batch_in_all_start_index"=some (fun _ => a.start) ∧
  s.regs .nat [] "cur_head"=some (fun _ => a.head) ∧
  s.regs .int [] "cur_batch_seq_len"=some (fun _ => a.length) ∧
  s.regs .nat [] "block_mask"=some (fun _ => blockMask p a)

theorem matches_ready (p : Layout) (M : Algebra α) (a : Inputs α) (acc : Acc p α)
    (s : State α) (h : Matches p M a acc s) : Ready p s := by
  rcases h with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp only [Ready,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,Option.isSome_some,and_self]

theorem prelude (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α) (s : State α) :
    ∃ t,run (M.withDefaultPrecision .fp32) (parts p r scale).before s=some t ∧
      Matches p M (inputs p r s) (initial p M) t ∧ t.mem=s.mem := by
  simp (config := {maxSteps := 500000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Matches,inputs,initial,keyEnd,blockMask,queryTile,row,bop]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,queryTile,row]
  split_ifs with h
  · exact (min_eq_left (le_of_lt h)).symm
  · exact (min_eq_right (le_of_not_gt h)).symm


theorem body (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0 < p.BN) (initialState s : State α)
    (old : Acc p α) (live : Matches p M (inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (M.withDefaultPrecision .fp32) (parts p r scale).body
        (s.setReg "start_n" .int [] (fun _ => cur))=some t ∧
      Matches p M (inputs p r initialState) (advance p M dots scale (inputs p r initialState) cur old) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp (config := {maxSteps := 1000000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    dots.qk_bound,dots.pv_bound,TileShape.axisDim,hBN,Matches,advance,keyTile,valueTile,
    memory,inputs,page,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,keyTile,valueTile,page,Function.comp_def]
  refine ⟨?_,?_,?_⟩ <;> intros
  all_goals
    congr 12
    all_goals
      try unfold keyTile
      try unfold valueTile
      try unfold page
      simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,ite_mul]

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
