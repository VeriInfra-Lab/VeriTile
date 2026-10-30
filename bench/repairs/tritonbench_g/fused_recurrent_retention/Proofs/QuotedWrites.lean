import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForward
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackward

/-! Typed store bookkeeping for the already-proved source executions. A Write
contains no arithmetic operation or callback. The source-to-write facts are
proved from the exact Structural states; none is a compiler/runtime premise. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

structure Write (α : Type) where
  region : RegionName
  offset : Nat
  value : Cell α

def writes (items : List (Write α)) (u : State α) : State α :=
  items.foldl (fun v w => v.write w.region w.offset w.value) u

def covers : List (Write α) → RegionName → Nat → Prop
  | [],_,_ => False
  | w::rest,r,i => (r=w.region ∧ i=w.offset) ∨ covers rest r i

@[simp] theorem writes_nil (u : State α) : writes [] u=u := rfl
@[simp] theorem writes_cons (w : Write α) (rest : List (Write α)) (u : State α) :
    writes (w::rest) u=writes rest (u.write w.region w.offset w.value) := rfl

@[simp] theorem writes_append (xs ys : List (Write α)) (u : State α) :
    writes (xs++ys) u=writes ys (writes xs u) := by simp [writes,List.foldl_append]

theorem writes_mem_congr (items : List (Write α)) (u v : State α) (he : u.mem=v.mem) :
    (writes items u).mem=(writes items v).mem := by
  induction items generalizing u v with
  | nil => exact he
  | cons w rest ih =>
    apply ih
    funext r i
    simp [State.write,he]

@[simp] theorem writes_regs (items : List (Write α)) (u : State α) : (writes items u).regs=u.regs := by
  induction items generalizing u with
  | nil => rfl
  | cons w rest ih => simpa only [writes_cons,State.write_regs] using ih (u.write w.region w.offset w.value)

theorem scatter_writes {ι : Type} (items : List ι) (u : State α) (active : ι → Prop)
    [DecidablePred active] (event : ι → Write α) :
    writes (items.filterMap (fun i => if active i then some (event i) else none)) u =
    items.foldl (fun v i => if active i then
      v.write (event i).region (event i).offset (event i).value else v) u := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp [List.filterMap_cons,hi,ih,writes_cons]

/-- A local store identity: once each actual store's scalar value is known,
only its finite address coverage determines the resulting memory and frame. -/
theorem writes_read (items : List (Write α)) (u : State α) (target : RegionName → Nat → Cell α)
    (values : ∀ w∈items,w.value=target w.region w.offset) (r : RegionName) (i : Nat) :
    (writes items u).mem r i=if covers items r i then target r i else u.mem r i := by
  induction items generalizing u with
  | nil => simp [covers]
  | cons w rest ih =>
    rw [writes_cons,ih _ (fun v hv => values v (by simp [hv]))]
    have hw := values w (by simp)
    by_cases hh : r=w.region ∧ i=w.offset
    · rcases hh with ⟨rfl,rfl⟩
      simp [covers,State.write,hw]
    · by_cases ht : covers rest r i <;> simp [covers,State.write,hh,ht]

theorem writes_frame (items : List (Write α)) (u : State α) (r : RegionName) (i : Nat)
    (untouched : ¬covers items r i) : (writes items u).mem r i=u.mem r i := by
  induction items generalizing u with
  | nil => rfl
  | cons w rest ih =>
    simp only [covers,not_or] at untouched
    rw [writes_cons,ih _ untouched.2]
    simp [State.write,untouched.1]

theorem covers_filterMap {ι : Type} (items : List ι) (active : ι → Prop)
    [DecidablePred active] (event : ι → Write α) (r : RegionName) (i : Nat) :
    covers (items.filterMap (fun j => if active j then some (event j) else none)) r i ↔
      ∃ j∈items,active j ∧ r=(event j).region ∧ i=(event j).offset := by
  induction items with
  | nil => simp [covers]
  | cons j rest ih =>
    by_cases hj : active j <;> simp [List.filterMap_cons,hj,covers,ih]

@[simp] theorem covers_append (xs ys : List (Write α)) (r : RegionName) (i : Nat) :
    covers (xs++ys) r i ↔ covers xs r i ∨ covers ys r i := by
  induction xs with
  | nil => simp [covers]
  | cons w rest ih => simp [covers,ih,or_assoc]

#axiomsClean scatter_writes
#axiomsClean writes_read
#axiomsClean writes_frame
#axiomsClean covers_filterMap
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
