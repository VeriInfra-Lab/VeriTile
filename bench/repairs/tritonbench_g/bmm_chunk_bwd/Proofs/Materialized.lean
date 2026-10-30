import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Arithmetic
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeAdmission

namespace VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def Materialized : Expr → Prop
  | .leaf .. | .literal .. => True
  | .cast _ _ _ x => Materialized x
  | .binary _ _ _ x y => Materialized x ∧ Materialized y
  | .pendingDot .. => False
  | .dotAcc _ _ _ _ _ _ _ _ _ _ a b acc _ =>
      (∀ i,Materialized (a i)) ∧ (∀ i,Materialized (b i)) ∧ Materialized acc

def Good (s : State Expr) : Prop := ∀ r off,
  s.mem r off=.mk .fp32 ((s.mem r off).read .fp32) ∧ Materialized ((s.mem r off).read .fp32)

theorem write_good (s : State Expr) (hs : Good s) (r : RegionName) (off : Nat) (v : Expr)
    (hv : Materialized v) : Good (s.write r off (.mk .fp32 v)) := by
  intro t n
  by_cases eq : t=r ∧ n=off
  · rcases eq with ⟨rfl,rfl⟩
    rw [State.write_same]
    exact ⟨rfl,hv⟩
  · simpa [State.write,eq] using hs t n

theorem left_materialized (rest : Algebra Expr) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (hs : Good s) (step : Nat) (i : TileIndex [c.BM,c.BCS]) :
    Materialized (NativeExecution.left (algebra rest) h c a alloc q s step i) := by
  dsimp only [NativeExecution.left,algebra]
  change Materialized (if _ then _ else _)
  split_ifs
  · exact (hs _ _).2
  · trivial

theorem right_materialized (rest : Algebra Expr) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (hs : Good s) (step : Nat) (i : TileIndex [c.BCS,c.BN]) :
    Materialized (NativeExecution.right (algebra rest) h c a alloc q s step i) := by
  dsimp only [NativeExecution.right,algebra]
  change Materialized (if _ then _ else _)
  split_ifs
  · exact (hs _ _).2
  · trivial

theorem carry_materialized (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (hs : Good s) (steps : Nat) (i : TileIndex [(tuning choice).BM,(tuning choice).BN]) :
    Materialized (NativeExecution.carry (algebra rest) (dot choice) h (tuning choice) a alloc q s steps i) := by
  induction steps with
  | zero => trivial
  | succ n ih =>
    change (∀ j,Materialized (NativeExecution.left (algebra rest) h (tuning choice) a alloc q s n j)) ∧
      (∀ j,Materialized (NativeExecution.right (algebra rest) h (tuning choice) a alloc q s n j)) ∧ _
    exact ⟨left_materialized rest h _ a alloc q s hs n,right_materialized rest h _ a alloc q s hs n,ih⟩

/-- The actual native carry update interprets as the complete fixed-K8 plan,
with the loaded words, reached carry and live=true. No numerical premise. -/
theorem carry_k8 (arith : Arithmetic) (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (steps : Nat) (i : TileIndex [(tuning choice).BM,(tuning choice).BN]) :
    interpret arith (NativeExecution.carry (algebra rest) (dot choice) h (tuning choice) a alloc q s (steps+1) i)=
      BmmChunkFwd.K8.plan arith.mma8 (config choice) [] (tuning choice).BM (tuning choice).BCS (tuning choice).BN
        (fun j => interpret arith (NativeExecution.left (algebra rest) h (tuning choice) a alloc q s steps j))
        (fun j => interpret arith (NativeExecution.right (algebra rest) h (tuning choice) a alloc q s steps j))
        (interpret arith (NativeExecution.carry (algebra rest) (dot choice) h (tuning choice) a alloc q s steps i)) Bool.true i := rfl

theorem value_materialized (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (hs : Good s) (i : TileIndex [(tuning choice).BM,(tuning choice).BN]) :
    Materialized (NativeExecution.value (algebra rest) (dot choice) h (tuning choice) a alloc q s i) := by
  unfold NativeExecution.value
  by_cases present : h.hasResidual=Bool.true
  · simp only [present,↓reduceIte,algebra,consume,Materialized]
    refine ⟨carry_materialized rest h choice a alloc q s hs _ i,?_⟩
    split_ifs
    · exact (hs _ _).2
    · trivial
  · simp only [present,↓reduceIte,algebra,Materialized]
    exact carry_materialized rest h choice a alloc q s hs _ i

theorem fold_good {ι : Type} (l : List ι) (s : State Expr) (hs : Good s)
    (r : RegionName) (off : ι → Nat) (value : ι → Expr) (act : ι → Bool)
    (hv : ∀ i∈l,Materialized (value i)) :
    Good (l.foldl (fun t i => if act i then t.write r (off i) (.mk .fp32 (value i)) else t) s) := by
  induction l generalizing s with
  | nil => exact hs
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    · split_ifs
      · exact write_good s hs r (off i) (value i) (hv i (by simp))
      · exact hs
    · intro j hj; exact hv j (by simp [hj])

theorem tile_good (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (q : HostFP.Pid)
    (s : State Expr) (hs : Good s) :
    Good (NativeExecution.tile (algebra rest) (dot choice) h (tuning choice) a alloc q s) := by
  unfold NativeExecution.tile
  apply fold_good _ s hs
  intro i _
  exact value_materialized rest h choice a alloc q s hs i

theorem grid_good (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (qs : List HostFP.Pid)
    (s : State Expr) (hs : Good s) :
    Good (NativeExecution.grid (algebra rest) (dot choice) h (tuning choice) a alloc qs s) := by
  induction qs generalizing s with
  | nil => exact hs
  | cons q qs ih => exact ih _ (tile_good rest h choice a alloc q s hs)

#axiomsClean carry_k8
#axiomsClean grid_good
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
