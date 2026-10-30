import bench.tritonbench_g.chunk_delta_fwd.Sources.Range
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)

theorem foldRange_observe {β γ : Type} (f : Nat → β → β) (get : β → γ)
    (n : Nat) (a : β) (h : ∀ k, k < n → ∀ b, get (f k b)=get b) :
    get (foldRange f 0 n 1 a)=get a := by
  induction n with
  | zero => rw [foldRange_zero]
  | succ n ih =>
    rw [foldRange_succ,h n (by omega)]
    exact ih (fun k hk => h k (by omega))

theorem foldRange_last {β γ : Type} (f : Nat → β → β) (get : β → γ)
    (n aidx : Nat) (a : β) (v : γ) (ha : aidx < n)
    (hit : ∀ b, get (f aidx b)=v)
    (later : ∀ k, aidx < k → k < n → ∀ b, get (f k b)=get b) :
    get (foldRange f 0 n 1 a)=v := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [foldRange_succ]
    by_cases he : aidx=n
    · subst aidx
      exact hit _
    · rw [later n (by omega) (by omega)]
      exact ih (by omega) (fun k hk hn => later k hk (by omega))

#axiomsClean foldRange_observe
#axiomsClean foldRange_last
end VeriTile.Bench.Repairs.ChunkDeltaFwd
