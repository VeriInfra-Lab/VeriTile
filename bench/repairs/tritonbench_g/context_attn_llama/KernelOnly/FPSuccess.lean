import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Kernels
import bench.tritonbench_g.context_attn_llama.Proofs.HostInterface
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Reciprocal
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

/-- These two local primitive implementations need only return their typed
result. Their values are entirely opaque: no exact arithmetic, exp, reduction,
final-output or kernel-success law is assumed. All other reached floating
operations are total functions in `Structural.Algebra`. -/
structure DotSupport (p : Layout) (M : Algebra α) where
  qk : Values α .real [p.BM,p.D] → Values α .real [p.D,p.BN] → Values α .real [p.BM,p.BN]
  pv : Values α .real [p.BM,p.BN] → Values α .real [p.BN,p.D] → Values α .real [p.BM,p.D] → Values α .real [p.BM,p.D]
  qk_bound : ∀ q k, M.dot (some .fp32) (batch := []) q k = some (qk q k)
  pv_bound : ∀ p v acc, M.dotAcc (some .fp32) .real .real .real {} (batch := []) p v acc = some (pv p v acc)

def parts (p : Layout) (r : Regions) (scale : ℝ) : Source.IntRange :=
  match Source.intRange? (kernel p r .fp32 scale).surfaceBody "start_n" with
  | some x => x
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split (p : Layout) (r : Regions) (scale : ℝ) :
    (kernel p r .fp32 scale).surfaceBody =
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
  (s.mem r.start (s.pids 1 / p.heads)).read .int

def length (p : Layout) (r : Regions) (s : State α) : Int :=
  (s.mem r.sequence (s.pids 1 / p.heads)).read .int -
  (s.mem r.prompt (s.pids 1 / p.heads)).read .int

def row (p : Layout) (s : State α) (i : TileIndex [p.BM,p.D]) : Nat :=
  p.BM * s.pids 0 + i.1.val

def offset (p : Layout) (r : Regions) (s : State α)
    (i : TileIndex [p.BM,p.D]) : Nat :=
  ((start p r s + row p s i) * p.outRow +
    ((s.pids 1 % p.heads) * p.outHead : Nat) + (i.2.1.val * p.outDim : Nat)).toNat

def active (p : Layout) (r : Regions) (s : State α)
    (i : TileIndex [p.BM,p.D]) : Prop := (row p s i : Int) < length p r s

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
