import bench.tritonbench_g.f8_conversion_utils.Proofs.Native
import bench.tritonbench_g.f8_conversion_utils.Proofs.Observation

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Lowering
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {M A B : Type} {iw ow : Nat} (ops : Memory.Primitives M A B iw ow)

/-- One masked source store, using already loaded/converted registers. The
actual finite pointer expression is evaluated only for active lanes. -/
def scatter (mask : Nat → Bool) (pointer : Nat → Nat) (registers : Nat → B) : List Nat → M → M
  | [],s => s
  | lane::rest,s => scatter mask pointer registers rest
      (if mask lane then ops.writeOutput s (pointer lane) (registers lane) else s)

theorem scatter_eq (mask : Nat → Bool) (pointer index : Nat → Nat)
    (registers values : Nat → B) (base : Nat) (lanes : List Nat) (before : M)
    (addresses : ∀ lane∈lanes,mask lane=Bool.true → pointer lane=Memory.address base ow (index lane))
    (loaded : ∀ lane∈lanes,mask lane=Bool.true → registers lane=values (index lane)) :
    scatter ops mask pointer registers lanes before=
      Memory.stores ops base values ((lanes.filter mask).map index) before := by
  induction lanes generalizing before with
  | nil => rfl
  | cons lane rest ih =>
    have addr := addresses lane (by simp)
    have value := loaded lane (by simp)
    have tailA : ∀ k∈rest,mask k=Bool.true → pointer k=Memory.address base ow (index k) :=
      fun k hk => addresses k (by simp [hk])
    have tailV : ∀ k∈rest,mask k=Bool.true → registers k=values (index k) :=
      fun k hk => loaded k (by simp [hk])
    cases hm : mask lane
    · simp [scatter,hm,ih _ tailA tailV]
    · simp only [scatter,hm,↓reduceIte,List.filter_cons_of_pos hm,List.map_cons,Memory.stores]
      rw [addr hm,value hm]
      exact ih _ tailA tailV

def lanes (count pid : Nat) : List Nat :=
  (List.range 1024).filter (NativeIndex.mask count pid)

def jobs (start count pid : Nat) : List Nat :=
  (lanes count pid).map (fun lane => start+(pid*1024+lane))

/-- Literal operation sequence of either original JIT after scalar binding:
one masked load, then one encode store or both decode stores. The conversion
port is local to one scalar destination conversion and retains the configured
load/store ABI; there is no kernel-result or successful-execution callback. -/
def program (twice : Bool) (cast : A → B) (input output count pid : Nat) (before : M) : M :=
  let mask := NativeIndex.mask count pid
  let registers := fun lane => cast (ops.readInput before (NativeIndex.pointer input iw pid lane))
  let first := scatter ops mask (NativeIndex.pointer output ow pid) registers (List.range 1024) before
  if twice then scatter ops mask (NativeIndex.pointer output ow pid) registers (List.range 1024) first
  else first

/-- All active native loads/stores evaluate to the dense scalar addresses.
The allocation and actual mask derive these equations for every admitted
scalar width, independently of the selected numerical conversion. -/
theorem program_eq (twice : Bool) (cast : A → B)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (start count pid : Nat) (small : count<2^31) (launched : pid<NativeIndex.grid count)
    (globalActive : ∀ i,i<count → start+i<n) (before : M) :
    program ops twice cast (x.base+start*iw) (y.base+start*ow) count pid before=
      Memory.run ops twice cast x.base y.base [jobs start count pid] before := by
  let values := fun i => cast (ops.readInput before (Memory.address x.base iw i))
  have active (lane : Nat) (member : lane∈List.range 1024)
      (masked : NativeIndex.mask count pid lane=Bool.true) :
      lane<1024 ∧ start+(pid*1024+lane)<n := by
    have inTile := List.mem_range.mp member
    have bound := (VeriTile.Bench.TritonBenchG.VectorAddition.Native.bounded_arithmetic
      count pid lane small launched inTile).2
    change NativeIndex.mask count pid lane=decide (pid*1024+lane<count) at bound
    have inside : pid*1024+lane<count := by rw [bound] at masked;exact of_decide_eq_true masked
    exact ⟨inTile,globalActive _ inside⟩
  have stores (current : M) :
      scatter ops (NativeIndex.mask count pid) (NativeIndex.pointer (y.base+start*ow) ow pid)
        (fun lane => cast (ops.readInput before (NativeIndex.pointer (x.base+start*iw) iw pid lane)))
        (List.range 1024) current=
      Memory.stores ops y.base values (jobs start count pid) current := by
    apply scatter_eq
    · intro lane member masked
      obtain ⟨inTile,inside⟩ := active lane member masked
      exact Native.pointer y start count pid lane small launched inTile inside
    · intro lane member masked
      obtain ⟨inTile,inside⟩ := active lane member masked
      rw [Native.pointer x start count pid lane small launched inTile inside]
  cases twice <;> simp [program,Memory.run,Memory.writes,stores,Memory.stores_append,values]

private theorem map_filter (f : Nat → Nat) (p : Nat → Bool) (xs : List Nat) :
    (xs.filter (fun x => p (f x))).map f=(xs.map f).filter p := by
  induction xs with
  | nil => rfl
  | cons x xs ih => cases h:p (f x) <;> simp [h,ih]

theorem jobs_eq (start count pid : Nat) (small : count<2^31)
    (launched : pid<NativeIndex.grid count) :
    jobs start count pid=(programLanes count pid).map (start+·) := by
  have masks : (List.range 1024).filter (NativeIndex.mask count pid)=
      (List.range 1024).filter (fun lane => decide (pid*1024+lane<count)) := by
    apply List.filter_congr
    intro lane member
    exact (VeriTile.Bench.TritonBenchG.VectorAddition.Native.bounded_arithmetic
      count pid lane small launched (List.mem_range.mp member)).2
  unfold jobs lanes
  rw [masks]
  have mapped := map_filter (fun lane => pid*1024+lane) (fun i => decide (i<count)) (List.range 1024)
  simpa only [List.map_map,Function.comp_def,programLanes] using congrArg (List.map (start+·)) mapped

#axiomsClean program_eq
#axiomsClean jobs_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Lowering
