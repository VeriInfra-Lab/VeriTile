import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Source
import Mathlib.Tactic

/-! Ordinary four-byte FP32 copy semantics. Every supplied view retains its
object identifier and address map. Copy jobs are ordered by output role, then
row-major logical coordinate. The byte target covers internal overlap, overlap
between output roles, lazy-negative encoding and unmodified storage holes.
No source execution or output formula is assumed here.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.Copyback
noncomputable section
variable {β : Type}
abbrev Heap (β : Type) := Nat → β

structure Job (β : Type) where
  target : Nat
  bytes : Fin 4 → β

def Job.at (job : Job β) (addr : Nat) : Option β :=
  if h : job.target≤addr ∧ addr<job.target+4 then
    some (job.bytes ⟨addr-job.target,by omega⟩) else none

def write (job : Job β) (heap : Heap β) : Heap β :=
  fun addr => (job.at addr).getD (heap addr)

def commit (jobs : List (Job β)) (heap : Heap β) : Heap β := jobs.foldl (fun u job => write job u) heap

def lastValue : List (Job β) → Nat → Option β
  | [],_ => none
  | job::jobs,addr => (lastValue jobs addr).orElse (fun _ => job.at addr)

theorem exact_last_write (jobs : List (Job β)) (heap : Heap β) (addr : Nat) :
    commit jobs heap addr=(lastValue jobs addr).getD (heap addr) := by
  induction jobs generalizing heap with
  | nil => rfl
  | cons job jobs ih =>
    change commit jobs (write job heap) addr=_
    rw [ih]
    cases hlast : lastValue jobs addr <;> simp [lastValue,hlast,write]

theorem frame (jobs : List (Job β)) (heap : Heap β) (addr : Nat)
    (outside : ∀ job∈jobs, ¬(job.target≤addr ∧ addr<job.target+4)) :
    commit jobs heap addr=heap addr := by
  induction jobs generalizing heap with
  | nil => rfl
  | cons job jobs ih =>
    change commit jobs (write job heap) addr=_
    rw [ih _ (fun j hj => outside j (List.mem_cons_of_mem _ hj))]
    simp [write,Job.at,outside job (by simp)]

theorem word_readback (job : Job β) (heap : Heap β) (byte : Fin 4) :
    write job heap (job.target+byte.val)=job.bytes byte := by
  simp only [write,Job.at]
  rw [dif_pos (by have hb := byte.isLt; omega)]
  simp

theorem last_job (jobs : List (Job β)) (job : Job β) (heap : Heap β) (byte : Fin 4) :
    commit (jobs++[job]) heap (job.target+byte.val)=job.bytes byte := by
  simp only [commit,List.foldl_append,List.foldl_cons,List.foldl_nil]
  exact word_readback job _ byte

structure View (n : Nat) where
  object : Nat
  address : Fin n → Nat
  negative : Bool

/-- Encoding includes the ordinary logical-copy handling of a negative view;
it is not assumed to leave its underlying payload unchanged. -/
def jobs {α : Type} {n : Nat} (view : View n) (encode : Bool → α → Fin 4 → β)
    (values : Fin n → α) : List (Job β) :=
  (List.finRange n).map fun i => ⟨view.address i,encode view.negative (values i)⟩

def touches {n : Nat} (view : View n) (addr : Nat) : Prop :=
  ∃ i : Fin n, view.address i≤addr ∧ addr<view.address i+4

theorem view_frame {α : Type} {n : Nat} (view : View n) (encode : Bool → α → Fin 4 → β)
    (values : Fin n → α) (heap : Heap β) (addr : Nat) (outside : ¬touches view addr) :
    commit (jobs view encode values) heap addr=heap addr := by
  apply frame
  intro job member inside
  obtain ⟨i,_,rfl⟩ := List.mem_map.mp member
  exact outside ⟨i,inside⟩

/-- This is the ordinary bulk-copy contract used only after Torch reports
nonoverlap. Its byte result equals scalar row-major copying. -/
def Nonoverlap {n : Nat} (view : View n) : Prop :=
  ∀ (i j : Fin n) (a b : Fin 4), view.address i+a.val=view.address j+b.val → i=j ∧ a=b

theorem view_readback {α : Type} {n : Nat} (view : View n) (encode : Bool → α → Fin 4 → β)
    (values : Fin n → α) (heap : Heap β) (unique : Nonoverlap view) (i : Fin n) (byte : Fin 4) :
    commit (jobs view encode values) heap (view.address i+byte.val)=encode view.negative (values i) byte := by
  have member := List.mem_finRange i
  obtain ⟨before,after,eq⟩ := List.append_of_mem member
  have nodup := List.nodup_finRange n
  rw [eq,List.nodup_append,List.nodup_cons] at nodup
  have absent : i∉after := nodup.2.1.1
  unfold jobs commit
  rw [eq,List.map_append,List.map_cons,List.foldl_append,List.foldl_cons]
  have outside : ∀ job∈after.map (fun j : Fin n => (⟨view.address j,encode view.negative (values j)⟩ : Job β)),
      ¬(job.target≤view.address i+byte.val ∧ view.address i+byte.val<job.target+4) := by
    intro job hj bounds
    obtain ⟨j,jmember,rfl⟩ := List.mem_map.mp hj
    let b : Fin 4 := ⟨view.address i+byte.val-view.address j,by dsimp at bounds; omega⟩
    have address : view.address i+byte.val=view.address j+b.val := by dsimp [b]; dsimp at bounds; omega
    have same := (unique i j byte b address).1
    exact absent (same ▸ jmember)
  change commit (after.map (fun j : Fin n => (⟨view.address j,encode view.negative (values j)⟩ : Job β))) _ (view.address i+byte.val)=_
  rw [frame _ _ _ outside]
  exact word_readback _ _ byte

#axiomsClean exact_last_write
#axiomsClean view_frame
#axiomsClean view_readback
end
end VeriTile.Bench.Repairs.DecayCumsum.Copyback
