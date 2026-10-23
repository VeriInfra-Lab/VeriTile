import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.StorageOwnership
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Plans
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Physical

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Binding
open Checked Machine
open VeriTile.Bench.Repairs.SwigluTriton.Memory (Key)
abbrev API := VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops
noncomputable section
open Classical

structure Cell where
  region : Region
  program : Nat
  time : Nat
  lane : Nat
  deriving DecidableEq,Repr

def tileCells (region : Region) (programs times lanes : Nat) : List Cell :=
  (List.range programs).flatMap fun p => (List.range times).flatMap fun t =>
    (List.range lanes).map fun j => ⟨region,p,t,j⟩

def cells (c : Shape) (seed : Bool) : List Cell :=
  tileCells .output c.programs c.N 1024 ++ tileCells .dx c.programs c.N 1024 ++
  tileCells .tileGate c.programs c.N 1 ++
  (if 0 < c.N ∨ seed=Bool.true then tileCells .dseed c.programs 1 1024 else []) ++
  tileCells .dgate c.heads c.N 1

theorem tileCells_mem (region : Region) (programs times lanes : Nat) (x : Cell) :
    x∈tileCells region programs times lanes ↔
      x.region=region ∧ x.program < programs ∧ x.time < times ∧ x.lane < lanes := by
  cases x
  simp [tileCells,List.mem_flatMap,Cell.mk.injEq,and_comm,and_left_comm,and_assoc,eq_comm]

theorem cells_mem (c : Shape) (seed : Bool) (x : Cell) :
    x∈cells c seed ↔ Source.writes c seed x.region x.program x.time x.lane := by
  by_cases hs : 0 < c.N ∨ seed=Bool.true <;>
    simp only [cells,hs,↓reduceIte,List.mem_append,tileCells_mem,List.not_mem_nil,
      or_false,Source.writes,Nat.lt_one_iff]
  · cases x.region <;> simp
  · cases x.region <;> simp

def key (c : Shape) (seed : Bool) (runtime : Runtime c seed) (x : Cell) : Key :=
  ⟨(runtime.tensor x.region).device,
    (runtime.tensor x.region).base+4*(offset c runtime.abi x.region x.program x.time x.lane).toNat⟩

def mirror {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool) (runtime : Runtime c seed) (m : M) : Memory α :=
  fun r p t j => api.read m (key c seed runtime ⟨r,p,t,j⟩)

theorem key_address (c : Shape) (seed : Bool) (runtime : Runtime c seed) (x : Cell)
    (active : Active c seed x.region x.program x.time x.lane) :
    ((key c seed runtime x).address : Int)=
      (runtime.abi.allocation x.region).dataPointer+4*offset c runtime.abi x.region x.program x.time x.lane := by
  have nonnegative := (active_offset c seed runtime.abi x.region x.program x.time x.lane active).1
  change (((runtime.tensor x.region).base+4*(offset c runtime.abi x.region x.program x.time x.lane).toNat : Nat) : Int)=_
  push_cast
  rw [Int.toNat_of_nonneg nonnegative]
  rfl

theorem key_contained (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (x : Cell)
    (active : Active c seed x.region x.program x.time x.lane) :
    (runtime.tensor x.region).allocationBase ≤ (key c seed runtime x).address ∧
    (key c seed runtime x).address+4 ≤ (runtime.tensor x.region).allocationBase+(runtime.tensor x.region).allocationBytes := by
  have contains := (active_physical c seed runtime accepted x.region x.program x.time x.lane active).1
  change ((runtime.tensor x.region).allocationBase : Int) ≤
      (runtime.abi.allocation x.region).dataPointer+4*offset c runtime.abi x.region x.program x.time x.lane ∧
    (runtime.abi.allocation x.region).dataPointer+4*offset c runtime.abi x.region x.program x.time x.lane+4 ≤
      (runtime.tensor x.region).allocationBase+(runtime.tensor x.region).allocationBytes at contains
  rw [←key_address c seed runtime x active] at contains
  exact_mod_cast contains

theorem keys_disjoint (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (x y : Cell)
    (hx : x∈cells c seed) (hy : y∈cells c seed) (different : x≠y) :
    (key c seed runtime x).disjoint (key c seed runtime y) := by
  have ax := active_written c seed x.region x.program x.time x.lane ((cells_mem c seed x).mp hx)
  have ay := active_written c seed y.region y.program y.time y.lane ((cells_mem c seed y).mp hy)
  by_cases regions : x.region=y.region
  · have distinct : (offset c runtime.abi x.region x.program x.time x.lane).toNat≠
        (offset c runtime.abi y.region y.program y.time y.lane).toNat := by
      intro same
      have a := (active_offset c seed runtime.abi x.region x.program x.time x.lane ax.2).1
      have b := (active_offset c seed runtime.abi y.region y.program y.time y.lane ay.2).1
      have equal := congrArg (fun n : Nat => (n : Int)) same
      dsimp only at equal
      rw [Int.toNat_of_nonneg a,Int.toNat_of_nonneg b] at equal
      have fields := offset_injective c seed runtime.abi x.region x.program x.time x.lane y.program y.time y.lane ax.1 ax.2
        (by simpa only [regions] using ay.2) (by simpa only [regions] using equal)
      apply different
      cases x; cases y
      simp_all
    simp only [Key.disjoint,key]
    by_cases before : (offset c runtime.abi x.region x.program x.time x.lane).toNat <
        (offset c runtime.abi y.region y.program y.time y.lane).toNat
    · exact Or.inr (Or.inl (by rw [regions] at *; omega))
    · exact Or.inr (Or.inr (by rw [regions] at *; omega))
  · have px := key_contained c seed runtime accepted x ax.2
    have py := key_contained c seed runtime accepted y ay.2
    have fresh := runtime.freshOutput x.region y.region ax.1 ay.1 regions
    change (runtime.tensor x.region).allocationBase+(runtime.tensor x.region).allocationBytes ≤
        ((runtime.tensor y.region).allocationBase : Int) ∨
      (runtime.tensor y.region).allocationBase+(runtime.tensor y.region).allocationBytes ≤
        ((runtime.tensor x.region).allocationBase : Int) at fresh
    rcases fresh with before|after
    · exact Or.inr (Or.inl (by omega))
    · exact Or.inr (Or.inr (by omega))

/-- A finite-store realization of the proven complete source-memory execution.
This is a mathematical normal form; the Python wrapper adds no copyback. -/
def commit {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (values : Memory α) (m : M) : M :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.stores api (key c seed runtime)
    (fun x => values x.region x.program x.time x.lane) (cells c seed) m

theorem commit_read {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (values : Memory α) (m : M) (x : Cell)
    (write : Source.writes c seed x.region x.program x.time x.lane) :
    api.read (commit api c seed runtime values m) (key c seed runtime x)=values x.region x.program x.time x.lane := by
  have member := (cells_mem c seed x).mpr write
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.readback api _ _ _ m x member
  · intro y hy equal
    have same : y=x := by
      by_contra different
      exact Key.disjoint_ne (keys_disjoint c seed runtime accepted y x hy member different) equal
    rw [same]
  · intro y hy different
    exact keys_disjoint c seed runtime accepted x y member hy (by intro same; subst y; exact different rfl)

def Outside (c : Shape) (seed : Bool) (runtime : Runtime c seed) (owner address : Nat) : Prop :=
  ∀ x,x∈cells c seed → ¬(key c seed runtime x).covers owner address

theorem commit_frame {M α β : Type} (api : API M α β) (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (values : Memory α) (m : M) (owner address : Nat) (outside : Outside c seed runtime owner address) :
    api.observe (commit api c seed runtime values m) owner address=api.observe m owner address :=
  VeriTile.Bench.Repairs.SwigluTriton.Memory.bytes_frame api _ _ _ m owner address outside

#axiomsClean cells_mem
#axiomsClean keys_disjoint
#axiomsClean commit_read
#axiomsClean commit_frame
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Binding
