import bench.tritonbench_g.matmul_dequant_int4.Sources.Physical
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Autotune
set_option maxHeartbeats 4000000

def dequantTrials (s : Shape) (p : DequantOps α) : List (List Work) → State α → State α
  | [],m => m
  | order::rest,m => dequantTrials s p rest (dequantRun s p order m)
structure FusedTrial where
  tile : Tile
  fusion : Nat → Nat → Fusion
  order : List Work
def fusedTrials (s : Shape) (p : FusedOps α β) (ng : Bool) : List FusedTrial → State α → State α
  | [],m => m
  | trial::rest,m => fusedTrials s p ng rest (fusedRun s trial.tile p ng trial.fusion trial.order m)

theorem dequant_readonly (s : Shape) (p : DequantOps α) (trials : List (List Work))
    (initial current : State α) (same : Writes.Readonly .dequant initial current) :
    Writes.Readonly .dequant initial (dequantTrials s p trials current) := by
  induction trials generalizing current with
  | nil => exact same
  | cons order rest ih =>
    apply ih
    rw [dequant_run_writes]
    exact Writes.run_readonly _ _ _ order initial current same
theorem fused_readonly (s : Shape) (p : FusedOps α β) (ng : Bool) (trials : List FusedTrial)
    (initial current : State α) (same : Writes.Readonly .output initial current) :
    Writes.Readonly .output initial (fusedTrials s p ng trials current) := by
  induction trials generalizing current with
  | nil => exact same
  | cons trial rest ih =>
    apply ih
    rw [fused_run_writes]
    exact Writes.run_readonly _ _ _ trial.order initial current same
theorem dequant_absorbed (s : Shape) (p : DequantOps α) (trials : List (List Work))
    (g : Grid s.k s.n) (order : List Work) (complete : g.Complete order)
    (initial current : State α) (same : Writes.Readonly .dequant initial current)
    (k n : Nat) (hk : k<s.k) (hn : n<s.n) :
    (dequantRun s p order (dequantTrials s p trials current)).values .dequant (k*s.n+n)=
      dequantValue s p initial ⟨k,n⟩ := by
  rw [dequant_complete s p g order complete _ k n hk hn]
  exact dequant_stable s p initial _ (dequant_readonly s p trials initial current same) _
theorem fused_absorbed (s : Shape) (p : FusedOps α β) (ng : Bool) (trials : List FusedTrial)
    (t : Tile) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n) (order : List Work)
    (complete : g.Complete order) (initial : State α) (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (fusedRun s t p ng fusion order (fusedTrials s p ng trials initial)).values .output (r*s.n+n)=
      fusedValue s t p ng fusion initial ⟨r,n⟩ := by
  rw [fused_complete s t p ng fusion g order complete _ r n hr hn]
  exact fused_stable s t p ng fusion initial _
    (fused_readonly s p ng trials initial initial (Writes.reflexive _ _)) _

theorem dequant_frame (s : Shape) (p : DequantOps α) (trials : List (List Work))
    (active : ∀ order∈trials,∀ w∈order,w.Active s.k s.n) (initial : State α)
    (r : Region) (a : Nat) (outside : r≠.dequant ∨ s.k*s.n≤a) :
    (dequantTrials s p trials initial).values r a=initial.values r a := by
  induction trials generalizing initial with
  | nil => rfl
  | cons order rest ih =>
    change (dequantTrials s p rest (dequantRun s p order initial)).values r a=_
    rw [ih (fun o ho => active o (by simp [ho])),dequant_run_writes]
    apply Writes.frame
    intro w hw ⟨hr,ha⟩
    have h := active order (by simp) w hw
    have hb := rowMajor_bound _ _ _ _ h.1 h.2
    rcases outside with h|h
    · exact h hr
    · omega

#axiomsClean dequant_readonly
#axiomsClean fused_readonly
#axiomsClean dequant_absorbed
#axiomsClean fused_absorbed
#axiomsClean dequant_frame
end VeriTile.Bench.Repairs.MatmulDequantInt4.Autotune
