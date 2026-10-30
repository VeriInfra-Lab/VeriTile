import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.Kernels
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FPSuccess

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

/-- These two local primitive implementations need only return their typed
result. Their values are entirely opaque: no exact arithmetic, exp, reduction,
final-output or kernel-success law is assumed. All other reached floating
operations are total functions in `Structural.Algebra`. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
      Values α left.toTileDType (batch++[m,k]) →
      Values α right.toTileDType (batch++[k,n]) →
      Values α out.toTileDType (batch++[m,n])

def withDot (M : Algebra α) (D : Dot α) : Algebra α :=
  { M with
    dot := fun pr {_} {_} {_} {_} a b => some (D.apply pr .real .real .real {} a b)
    dotWithOptions := fun pr options {_} {_} {_} {_} a b => some (D.apply pr .real .real .real options a b)
    dotTyped := fun pr l r o options {_} {_} {_} {_} a b => some (D.apply pr l r o options a b) }

def parts (p : Layout) (r : Regions) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (kernel p r scale).surfaceBody "start_n" with
  | some x => x
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (p : Layout) (r : Regions) (scale : ℝ) :
    (kernel p r scale).surfaceBody =
    (parts p r scale).before ++
      [.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body] ++ (parts p r scale).after := rfl

/-- Exactly the registers read across the source loop boundary. -/
def Ready (p : Layout) (s : State α) : Prop :=
  (s.regs .nat [] "cur_kv_head").isSome ∧
  (s.regs .int [] "cur_batch_req_idx").isSome ∧
  (s.regs .int [] "block_end_loc").isSome ∧
  (s.regs .nat [p.BN] "offs_n").isSome ∧
  (s.regs .nat [p.D] "offs_d").isSome ∧
  (s.regs .nat [p.BM] "offs_m").isSome ∧
  (s.regs .int [] "prompt_cache_len").isSome ∧
  (s.regs .real [p.BM,p.D] "q").isSome ∧
  (s.regs .real [p.BM] "m_i").isSome ∧
  (s.regs .real [p.BM] "l_i").isSome ∧
  (s.regs .real [p.BM,p.D] "acc").isSome ∧
  (s.regs .int [] "cur_batch_in_all_start_index").isSome ∧
  (s.regs .nat [] "cur_head").isSome ∧
  (s.regs .int [] "cur_batch_seq_len").isSome ∧
  (s.regs .nat [] "block_mask").isSome

def start (p : Layout) (r : Regions) (s : State α) : Int :=
  (s.mem r.start (s.pids 0)).read .int

def length (p : Layout) (r : Regions) (s : State α) : Int :=
  (s.mem r.sequence (s.pids 0)).read .int -
  (s.mem r.prompt (s.pids 0)).read .int

def row (p : Layout) (s : State α) (i : TileIndex [p.BM,p.D]) : Nat :=
  p.BM * s.pids 2 + i.1.val

def offset (p : Layout) (r : Regions) (s : State α)
    (i : TileIndex [p.BM,p.D]) : Nat :=
  ((start p r s + row p s i) * p.outRow +
    ((s.pids 1) * p.outHead : Nat) + (i.2.1.val * p.outDim : Nat)).toNat

def active (p : Layout) (r : Regions) (s : State α)
    (i : TileIndex [p.BM,p.D]) : Prop := (row p s i : Int) < length p r s ∧ i.2.1.val < p.headDim

def Fixed (p : Layout) (s t : State α) : Prop :=
  t.regs .int [] "cur_batch_in_all_start_index" = s.regs .int [] "cur_batch_in_all_start_index" ∧
  t.regs .nat [] "cur_head" = s.regs .nat [] "cur_head" ∧
  t.regs .nat [p.BM] "offs_m" = s.regs .nat [p.BM] "offs_m" ∧
  t.regs .nat [p.D] "offs_d" = s.regs .nat [p.D] "offs_d" ∧
  t.regs .int [] "cur_batch_seq_len" = s.regs .int [] "cur_batch_seq_len"

theorem Fixed.refl (p : Layout) (s : State α) : Fixed p s s := by simp [Fixed]
theorem Fixed.trans {p : Layout} {s t u : State α} (hst : Fixed p s t) (htu : Fixed p t u) :
    Fixed p s u := by rcases hst with ⟨a,b,c,d,e⟩; rcases htu with ⟨f,g,h,i,j⟩; simp_all [Fixed]

def Controls (p : Layout) (r : Regions) (s t : State α) : Prop :=
  t.regs .int [] "cur_batch_in_all_start_index" = some (fun _ => start p r s) ∧
  t.regs .nat [] "cur_head" = some (fun _ => s.pids 1) ∧
  t.regs .nat [p.BM] "offs_m" = some (fun i => p.BM * s.pids 2 + i.1.val) ∧
  t.regs .nat [p.D] "offs_d" = some (fun i => i.1.val) ∧
  t.regs .int [] "cur_batch_seq_len" = some (fun _ => length p r s)

theorem Controls.fixed {p : Layout} {r : Regions} {s t u : State α}
    (h : Controls p r s t) (hf : Fixed p t u) : Controls p r s u := by
  rcases h with ⟨a,b,c,d,e⟩; rcases hf with ⟨f,g,h,i,j⟩; simp_all [Controls]

theorem prelude_success (p : Layout) (r : Regions) (scale : ℝ)
    (M : Algebra α) (s : State α) :
    ∃ t, run (M.withDefaultPrecision .fp32) (parts p r scale).before s = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Controls p r s t := by
  simp (config := {maxSteps := 500000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,Ready,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,Controls,start,length,bop]
  constructor
  · funext i
    change s.pids 2 * p.BM + i.1.val = p.BM * s.pids 2 + i.1.val
    rw [Nat.mul_comm]
  · rfl

/-- All body assignments are checked from actual live-register availability;
the two dot callbacks are the only optional floating primitives on this branch. -/
theorem body_success (p : Layout) (r : Regions) (scale : ℝ)
    (base : Algebra α) (D : Dot α) (hBN : 0 < p.BN) (s : State α) (hs : Ready p s) (cur : Int) :
    ∃ t, run ((withDot base D).withDefaultPrecision .fp32) (parts p r scale).body
        (s.setReg "start_n" .int [] (fun _ => cur)) = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Fixed p s t := by
  rcases hs with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases Option.isSome_iff_exists.mp h0 with ⟨v0,h0⟩
  rcases Option.isSome_iff_exists.mp h1 with ⟨v1,h1⟩
  rcases Option.isSome_iff_exists.mp h2 with ⟨v2,h2⟩
  rcases Option.isSome_iff_exists.mp h3 with ⟨v3,h3⟩
  rcases Option.isSome_iff_exists.mp h4 with ⟨v4,h4⟩
  rcases Option.isSome_iff_exists.mp h5 with ⟨v5,h5⟩
  rcases Option.isSome_iff_exists.mp h6 with ⟨v6,h6⟩
  rcases Option.isSome_iff_exists.mp h7 with ⟨v7,h7⟩
  rcases Option.isSome_iff_exists.mp h8 with ⟨v8,h8⟩
  rcases Option.isSome_iff_exists.mp h9 with ⟨v9,h9⟩
  rcases Option.isSome_iff_exists.mp h10 with ⟨v10,h10⟩
  rcases Option.isSome_iff_exists.mp h11 with ⟨v11,h11⟩
  rcases Option.isSome_iff_exists.mp h12 with ⟨v12,h12⟩
  rcases Option.isSome_iff_exists.mp h13 with ⟨v13,h13⟩
  rcases Option.isSome_iff_exists.mp h14 with ⟨v14,h14⟩
  simp (config := {maxSteps := 1000000}) [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,Ready,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,
    withDot,TileShape.axisDim,hBN,Fixed]

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
