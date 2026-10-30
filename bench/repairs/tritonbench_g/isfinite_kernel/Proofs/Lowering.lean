import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Memory
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Finite
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Lowering

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Lowering
open VeriTile.Bench.TritonBenchG.VectorAddition
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {M A : Type} {width : Nat} (ops : Memory.Primitives M A width)

def scatter (mask : Nat → Bool) (pointer : Nat → Nat) (registers : Nat → Bool) : List Nat → M → M
  | [],s => s
  | lane::rest,s => scatter mask pointer registers rest
      (if mask lane then ops.write s (pointer lane) (registers lane) else s)

theorem scatter_eq (mask : Nat → Bool) (pointer index output : Nat → Nat)
    (registers values : Nat → Bool) (lanes : List Nat) (before : M)
    (addresses : ∀ lane∈lanes,mask lane=Bool.true → pointer lane=output (index lane))
    (loaded : ∀ lane∈lanes,mask lane=Bool.true → registers lane=values (index lane)) :
    scatter ops mask pointer registers lanes before=
      Memory.stores ops output values ((lanes.filter mask).map index) before := by
  induction lanes generalizing before with
  | nil => rfl
  | cons lane rest ih =>
    have addr := addresses lane (by simp)
    have value := loaded lane (by simp)
    have tailA : ∀ k∈rest,mask k=Bool.true → pointer k=output (index k) :=
      fun k hk => addresses k (by simp [hk])
    have tailV : ∀ k∈rest,mask k=Bool.true → registers k=values (index k) :=
      fun k hk => loaded k (by simp [hk])
    cases hm : mask lane
    · simp [scatter,hm,ih _ tailA tailV]
    · simp only [scatter,hm,↓reduceIte,List.filter_cons_of_pos hm,List.map_cons,Memory.stores]
      rw [addr hm,value hm]
      exact ih _ tailA tailV

abbrev jobs := F8ConversionUtils.Lowering.jobs

/-- Literal finite native program: masked physical loads precede masked byte
stores. The stride multiplication uses signed 64-bit encodings. -/
def program (classify : A → Bool) (input output : View) (count pid : Nat) (before : M) : M :=
  scatter ops (Finite.mask count pid) (Finite.pointer output pid)
    (fun lane => classify (ops.read before (Finite.pointer input pid lane))) (List.range 1024) before

theorem program_eq (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (start count pid : Nat) (small : count<2^31) (launched : pid<Finite.grid count)
    (globalActive : ∀ i,i<count → start+i<x.n) (before : M) :
    program ops classify (x.view.shift start) (y.outputView.shift start) count pid before=
      Memory.run ops classify x.view.address y.outputView.address [jobs start count pid] before := by
  let values := fun i => classify (ops.read before (x.view.address i))
  have active (lane : Nat) (member : lane∈List.range 1024)
      (masked : Finite.mask count pid lane=Bool.true) : lane<1024 ∧ start+(pid*1024+lane)<x.n := by
    have inTile := List.mem_range.mp member
    have bound := (Native.bounded_arithmetic count pid lane small launched inTile).2
    change Native.mask count pid lane=Bool.true at masked
    have inside : pid*1024+lane<count := by rw [bound] at masked;exact of_decide_eq_true masked
    exact ⟨inTile,globalActive _ inside⟩
  change scatter ops _ _ _ _ before=Memory.stores ops y.outputView.address values (jobs start count pid) before
  apply scatter_eq
  · intro lane member masked
    obtain ⟨inTile,inside⟩ := active lane member masked
    have exactPointer := Finite.checked_pointer Bool.true y h.output x.n start count pid lane h.shape
      small launched inTile inside
    simpa only [Finite.output_view y h.output] using exactPointer
  · intro lane member masked
    obtain ⟨inTile,inside⟩ := active lane member masked
    rw [Finite.checked_pointer Bool.false x h.input x.n start count pid lane rfl small launched inTile inside]

def programs (classify : A → Bool) (input output : View) (count : Nat) : List Nat → M → M
  | [],before => before
  | pid::rest,before => programs classify input output count rest (program ops classify input output count pid before)

theorem programs_eq (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (start count : Nat) (small : count<2^31) (pids : List Nat)
    (launched : ∀ pid∈pids,pid<Finite.grid count) (globalActive : ∀ i,i<count → start+i<x.n) (before : M) :
    programs ops classify (x.view.shift start) (y.outputView.shift start) count pids before=
      Memory.run ops classify x.view.address y.outputView.address (pids.map (jobs start count)) before := by
  induction pids generalizing before with
  | nil => rfl
  | cons pid rest ih =>
    rw [programs,program_eq ops classify x y h start count pid small (launched pid (by simp)) globalActive,
      ih (fun p hp => launched p (by simp [hp]))]
    rfl

def grid (classify : A → Bool) (input output : View) (count : Nat) (before : M) : M :=
  programs ops classify input output count (List.range (Finite.grid count)) before

theorem grid_eq (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (start count : Nat) (small : count<2^31) (globalActive : ∀ i,i<count → start+i<x.n) (before : M) :
    grid ops classify (x.view.shift start) (y.outputView.shift start) count before=
      Memory.run ops classify x.view.address y.outputView.address
        ((VeriTile.Bench.TritonBenchG.VectorAddition.Finite.sourceBatches count).map
          (fun lanes => lanes.map (start+·))) before := by
  rw [grid,programs_eq ops classify x y h start count small _ (fun _ member => List.mem_range.mp member) globalActive]
  congr 1
  unfold VeriTile.Bench.TritonBenchG.VectorAddition.Finite.sourceBatches
  rw [List.map_map]
  apply List.map_congr_left
  intro pid member
  exact F8ConversionUtils.Lowering.jobs_eq start count pid small (List.mem_range.mp member)

def chain (step : Nat → M → M) : List Nat → M → M
  | [],before => before
  | c::rest,before => chain step rest (step c before)

theorem run_append (classify : A → Bool) (input output : Nat → Nat) (xs ys : List (List Nat)) (s : M) :
    Memory.run ops classify input output (xs++ys) s=
      Memory.run ops classify input output ys (Memory.run ops classify input output xs s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons jobs rest ih => exact ih _

private theorem chain_eq (classify : A → Bool) (input output : Nat → Nat)
    (step : Nat → M → M) (groups : Nat → List (List Nat))
    (oneStep : ∀ c state,step c state=Memory.run ops classify input output (groups c) state)
    (cs : List Nat) (before : M) :
    chain step cs before=Memory.run ops classify input output (cs.flatMap groups) before := by
  induction cs generalizing before with
  | nil => rfl
  | cons c rest ih => rw [chain,oneStep,ih,List.flatMap_cons,run_append]

def allChunks (classify : A → Bool) (input output : View) (n : Nat) (cs : List Nat) (before : M) : M :=
  chain (fun c state => grid ops classify (input.shift (chunkStart c)) (output.shift (chunkStart c))
    (chunkLength n c) state) cs before

theorem allChunks_eq (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (cs : List Nat) (before : M) :
    allChunks ops classify x.view y.outputView x.n cs before=
      Memory.run ops classify x.view.address y.outputView.address (cs.flatMap (Finite.chunkBatches x.n)) before := by
  apply chain_eq
  intro c state
  exact grid_eq ops classify x y h (chunkStart c) (chunkLength x.n c)
    (Finite.small x.n c) (Finite.active x.n c) state

theorem complete (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y) (before : M) :
    allChunks ops classify x.view y.outputView x.n (List.range (chunkCount x.n)) before=
      Memory.run ops classify x.view.address y.outputView.address (Finite.batches x.n) before := by
  rw [allChunks_eq ops classify x y h]
  rfl

#axiomsClean program_eq
#axiomsClean complete
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Lowering
