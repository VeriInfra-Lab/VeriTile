import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Lowering

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Lowering
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {M A B : Type} {iw ow : Nat} (ops : Memory.Primitives M A B iw ow)

def programs (twice : Bool) (cast : A → B) (input output count : Nat) : List Nat → M → M
  | [],before => before
  | pid::rest,before => programs twice cast input output count rest
      (program ops twice cast input output count pid before)

theorem programs_eq (twice : Bool) (cast : A → B)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (start count : Nat) (small : count<2^31) (pids : List Nat)
    (launched : ∀ pid∈pids,pid<NativeIndex.grid count)
    (globalActive : ∀ i,i<count → start+i<n) (before : M) :
    programs ops twice cast (x.base+start*iw) (y.base+start*ow) count pids before=
      Memory.run ops twice cast x.base y.base (pids.map (jobs start count)) before := by
  induction pids generalizing before with
  | nil => rfl
  | cons pid rest ih =>
    rw [programs,program_eq ops twice cast x y start count pid small (launched pid (by simp)) globalActive,
      ih (fun p hp => launched p (by simp [hp]))]
    rfl

def grid (twice : Bool) (cast : A → B) (input output count : Nat) (before : M) : M :=
  programs ops twice cast input output count (List.range (NativeIndex.grid count)) before

theorem grid_eq (twice : Bool) (cast : A → B)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (start count : Nat) (small : count<2^31)
    (globalActive : ∀ i,i<count → start+i<n) (before : M) :
    grid ops twice cast (x.base+start*iw) (y.base+start*ow) count before=
      Memory.run ops twice cast x.base y.base
        ((sourceBatches count).map (fun lanes => lanes.map (start+·))) before := by
  rw [grid,programs_eq ops twice cast x y start count small _ (fun _ member => List.mem_range.mp member) globalActive]
  congr 1
  unfold sourceBatches
  rw [List.map_map]
  apply List.map_congr_left
  intro pid member
  exact jobs_eq start count pid small (List.mem_range.mp member)

def chain (step : Nat → M → M) : List Nat → M → M
  | [],before => before
  | c::rest,before => chain step rest (step c before)

private theorem chain_eq (twice : Bool) (cast : A → B) (input output : Nat)
    (step : Nat → M → M) (groups : Nat → List (List Nat))
    (oneStep : ∀ c state,step c state=Memory.run ops twice cast input output (groups c) state)
    (cs : List Nat) (before : M) :
    chain step cs before=Memory.run ops twice cast input output (cs.flatMap groups) before := by
  induction cs generalizing before with
  | nil => rfl
  | cons c rest ih =>
    rw [chain,oneStep,ih,List.flatMap_cons,Memory.run_append]

def allChunks (twice : Bool) (cast : A → B) (input output n : Nat) (cs : List Nat) (before : M) : M :=
  chain (fun c state => grid ops twice cast (input+Chunking.chunkStart c*iw) (output+Chunking.chunkStart c*ow)
    (Chunking.chunkLength n c) state) cs before

theorem allChunks_eq (twice : Bool) (cast : A → B)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow) (cs : List Nat) (before : M) :
    allChunks ops twice cast x.base y.base n cs before=
      Memory.run ops twice cast x.base y.base (cs.flatMap (Chunking.chunkBatches n)) before := by
  apply chain_eq
  intro c state
  exact grid_eq ops twice cast x y (Chunking.chunkStart c) (Chunking.chunkLength n c)
    (Chunking.chunk_small n c) (Chunking.chunk_active n c) state

/-- Every actual finite grid in every host chunk executes the same complete
memory transition; this equation is proved from masked native programs. -/
theorem complete (twice : Bool) (cast : A → B)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow) (before : M) :
    allChunks ops twice cast x.base y.base n (launchChunks n) before=
      Memory.run ops twice cast x.base y.base (Chunking.batches n) before := by
  rw [allChunks_eq]
  rfl

#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Lowering
