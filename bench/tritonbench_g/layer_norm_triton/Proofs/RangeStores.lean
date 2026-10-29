import bench.tritonbench_g.layer_norm_triton.Proofs.ConcurrentValues

namespace VeriTile.Bench.TritonBenchG.LayerNormLock
open VeriTile.Triton
open FP.Equational (ReductionPlan)

/-- Finite execution of the actual signed-i32 range test and increment. Fuel
exhaustion returns None; there is no forced exit or unbounded-Nat substitution.
This retains possible wraparound/nontermination outside defined host shapes. -/
def rangeTrace (limit step current : I32) : Nat → Option (List I32)
  | 0 => if current.slt limit then none else some []
  | fuel + 1 =>
    if current.slt limit then
      (rangeTrace limit step (current + step) fuel).map (current :: ·)
    else some []

inductive RangeExecution (limit step : I32) : I32 → List I32 → Prop
  | done (cur : I32) (h : cur.slt limit = false) : RangeExecution limit step cur []
  | next (cur : I32) (rest : List I32) (h : cur.slt limit = true)
      (tail : RangeExecution limit step (cur + step) rest) : RangeExecution limit step cur (cur :: rest)

theorem range_trace_sound (limit step current : I32) (fuel : Nat) (offsets : List I32)
    (h : rangeTrace limit step current fuel = some offsets) :
    RangeExecution limit step current offsets := by
  induction fuel generalizing current offsets with
  | zero =>
    simp only [rangeTrace] at h
    split at h
    · contradiction
    · cases h; exact .done _ (by simp_all)
  | succ fuel ih =>
    simp only [rangeTrace] at h
    split at h
    · rename_i hc
      cases hr : rangeTrace limit step (current + step) fuel with
      | none => simp [hr] at h
      | some rest =>
        simp only [hr, Option.map_some, Option.some.injEq] at h
        subst offsets
        exact .next _ _ hc (ih _ _ hr)
    · cases h; exact .done _ (by simp_all)

/-- Every terminating source range is accepted with a sufficiently large budget. -/
theorem range_trace_complete (limit step current : I32) (offsets : List I32)
    (h : RangeExecution limit step current offsets) :
    rangeTrace limit step current offsets.length = some offsets := by
  induction h with
  | done cur hc => simp [rangeTrace, hc]
  | next cur rest hc ht ih => simp [rangeTrace, hc, ih]

structure Write (V : Type) where
  region : Region
  address : Int
  value : V

def applyWrite (w : Write V) (mem : Memory V) : Memory V :=
  put mem w.region (put (mem w.region) w.address w.value)

def writeAll : List (Write V) → Memory V → Memory V
  | [], mem => mem
  | w :: ws, mem => writeAll ws (applyWrite w mem)

theorem writeAll_frame (ws : List (Write V)) (mem : Memory V) (region : Region) (address : Int)
    (h : ∀ w ∈ ws, w.region ≠ region ∨ w.address ≠ address) :
    writeAll ws mem region address = mem region address := by
  induction ws generalizing mem with
  | nil => rfl
  | cons w ws ih =>
    rw [writeAll, ih (applyWrite w mem) (fun v hv => h v (List.mem_cons_of_mem w hv))]
    rcases h w (List.mem_cons_self ..) with hr | ha
    · simp [applyWrite, put, Ne.symm hr]
    · by_cases hr : region = w.region
      · simp [applyWrite, put, hr, Ne.symm ha]
      · simp [applyWrite, put, hr]

def forwardWrites (b : Backend V) (c : Config) (p : Nat) (offsets : List I32)
    (v : ForwardValues V) : List (Write V) :=
  [⟨.mean, (BitVec.ofNat 32 p).toInt, b.cast .fp32 .fp32 v.mean⟩,
   ⟨.rstd, (BitVec.ofNat 32 p).toInt, b.cast .fp32 .fp32 v.rstd⟩] ++
  offsets.flatMap (fun off => (List.range c.block).filterMap (fun lane =>
    if live c off lane then some ⟨.y, rowAddr c p (colAt off lane), v.y off lane⟩ else none))

def forwardKernel (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p fuel : Nat) : Option (Memory V) :=
  (rangeTrace c.n (BitVec.ofNat 32 c.block) 0 fuel).map fun offsets =>
    writeAll (forwardWrites b c p offsets (forwardSource b c plan mem undef eps p offsets)) mem

def forwardMemoryReference (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p fuel : Nat) : Option (Memory V) :=
  (rangeTrace c.n (BitVec.ofNat 32 c.block) 0 fuel).map fun offsets =>
    writeAll (forwardWrites b c p offsets (forwardReference b c plan mem undef eps p offsets)) mem

theorem forward_kernel_value (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p fuel : Nat) :
    forwardKernel b c plan mem undef eps p fuel = forwardMemoryReference b c plan mem undef eps p fuel := by
  simp only [forwardKernel, forwardMemoryReference, forward_value]

def finalWrites (c : Config) (bn pid : Nat) (dw db : Nat → V) : List (Write V) :=
  (List.range bn).flatMap fun lane =>
    let col := BitVec.ofNat 32 pid * BitVec.ofNat 32 bn + BitVec.ofNat 32 lane
    if col.slt c.n then [⟨.finalDW, col.toInt, dw lane⟩, ⟨.finalDB, col.toInt, db lane⟩] else []

def finalKernel (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) : Option (Memory V) :=
  (rangeTrace m (BitVec.ofNat 32 bm) 0 fuel).map fun offsets =>
    writeAll (finalWrites c bn pid
      (finalSource b c mem m bm bn pid plan offsets .dw)
      (finalSource b c mem m bm bn pid plan offsets .db)) mem

def finalMemoryReference (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) : Option (Memory V) :=
  (rangeTrace m (BitVec.ofNat 32 bm) 0 fuel).map fun offsets =>
    writeAll (finalWrites c bn pid
      (finalReference b c mem m bm bn pid plan offsets .dw)
      (finalReference b c mem m bm bn pid plan offsets .db)) mem

theorem final_kernel_value (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    finalKernel b c mem m bm bn pid plan fuel = finalMemoryReference b c mem m bm bn pid plan fuel := by
  simp only [finalKernel, finalMemoryReference, final_value]

#axiomsClean range_trace_sound
#axiomsClean range_trace_complete
#axiomsClean writeAll_frame
#axiomsClean forward_kernel_value
#axiomsClean final_kernel_value
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
