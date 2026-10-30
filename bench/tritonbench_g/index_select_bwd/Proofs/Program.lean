import bench.tritonbench_g.index_select_bwd.Proofs.Source
import bench.tritonbench_g.index_select_bwd.Proofs.Logical

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Program
open VeriTile.Triton FP.Structural
open Logical
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def destinationRegion : RegionName := ⟨"index_bwd_direct_destination"⟩
def incomingRegion : RegionName := ⟨"index_bwd_direct_incoming"⟩
def layout (C k : Nat) : Launch :=
  ⟨destinationRegion,incomingRegion,min 512 (C-k*512),1,512⟩

def initial {A : Type} [Inhabited A] {R N C : Nat} (index : Fin N → Fin R)
    (h : Coord N C → A) (i : Fin N) (k : Nat) (g : Coord R C → A) : State A where
  mem := fun r o => if bound : k*512+o<C then
    if r=incomingRegion then Cell.mk .real (h (i,⟨k*512+o,bound⟩))
    else Cell.mk .real (g (index i,⟨k*512+o,bound⟩))
    else Cell.mk .real default
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

/-- One actual same-call typed source execution. The returned kernel memory
supplies active readback; its complete exterior frame is Source.frame. -/
def chunk {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N) (k : Nat)
    (g : Coord R C → A) : Coord R C → A :=
  let result := (Source.execution M (layout C k) (initial index h i k g)).val
  fun q => if q.1=index i ∧ q.2.val/512=k then
    (result.mem destinationRegion (q.2.val%512)).read .real
  else g q

theorem chunk_cell {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N) (k : Nat)
    (g : Coord R C → A) (q : Coord R C) :
    chunk M index h i k g q = if q.1=index i ∧ q.2.val/512=k then
      Source.scalar M (g q) (h (i,q.2)) else g q := by
  by_cases selected : q.1=index i ∧ q.2.val/512=k
  · have remainder : q.2.val%512<512 := Nat.mod_lt _ (by decide)
    have divide := Nat.div_add_mod q.2.val 512
    have bound := q.2.isLt
    have actual : k*512+q.2.val%512=q.2.val := by omega
    have active : q.2.val%512<min 512 (C-k*512) := by omega
    let lane : TileIndex [512] := (⟨q.2.val%512,remainder⟩,PUnit.unit)
    have value := Source.readback M (layout C k) (initial index h i k g) rfl lane active
    simp only [chunk,selected,if_pos]
    change ((Source.execution M (layout C k) (initial index h i k g)).val.mem
      (layout C k).destination lane.1.val).read .real=_
    rw [value]
    have coordinate : (⟨k*512+q.2.val%512,by omega⟩ : Fin C)=q.2 := Fin.ext actual
    have outputPin : (initial index h i k g).mem (layout C k).destination lane.1.val = Cell.mk .real (g q) := by
      dsimp only [initial,layout,lane]
      rw [dif_pos (show k*512+q.2.val%512<C by omega)]
      rw [if_neg (by decide : destinationRegion≠incomingRegion)]
      exact congrArg (fun coordinate => Cell.mk .real (g coordinate)) (Prod.ext selected.1.symm coordinate)
    have inputPin : (initial index h i k g).mem (layout C k).incoming lane.1.val = Cell.mk .real (h (i,q.2)) := by
      dsimp only [initial,layout,lane]
      rw [dif_pos (show k*512+q.2.val%512<C by omega),if_pos rfl]
      exact congrArg (fun coordinate => Cell.mk .real (h coordinate)) (Prod.ext rfl coordinate)
    rw [outputPin,inputPin]
    rfl
  · simp [chunk,selected]

def oneRow {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N)
    (g : Coord R C → A) : Coord R C → A :=
  (List.range ((C+511)/512)).foldl (fun state k => chunk M index h i k state) g

theorem chunk_fold_cell {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N) (ks : List Nat)
    (g : Coord R C → A) (q : Coord R C) :
    (ks.foldl (fun state k => chunk M index h i k state) g) q =
      ks.foldl (fun acc k => if q.1=index i ∧ q.2.val/512=k then
        Source.scalar M acc (h (i,q.2)) else acc) (g q) := by
  induction ks generalizing g with
  | nil => rfl
  | cons k rest ih => simp only [List.foldl_cons,ih,chunk_cell]

theorem oneRow_correct {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (i : Fin N)
    (g : Coord R C → A) : oneRow M index h i g=row (Source.scalar M) index h i g := by
  funext q
  unfold oneRow
  rw [chunk_fold_cell]
  have bound := q.2.isLt
  have occurs : q.2.val/512∈List.range ((C+511)/512) := by simp only [List.mem_range];omega
  by_cases same : q.1=index i
  · simp only [same,true_and]
    have swap : (fun (acc : A) (k : Nat) => if q.2.val/512=k then Source.scalar M acc (h (i,q.2)) else acc)=
        (fun acc k => if k=q.2.val/512 then Source.scalar M acc (h (i,q.2)) else acc) := by
      funext acc k; simp only [eq_comm]
    rw [swap,fold_once _ _ _ List.nodup_range]
    simp [occurs,row,same.symm]
  · simp [same,row,Ne.symm same]

/-- The complete Python launch order, including repeated indices: index i is
outermost; all 512-column slices of i finish before i+1 is submitted. -/
def run {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (g : Coord R C → A) : Coord R C → A :=
  (List.finRange N).foldl (fun state i => oneRow M index h i state) g

theorem run_correct {A : Type} [Inhabited A] {R N C : Nat} (M : Algebra A)
    (index : Fin N → Fin R) (h : Coord N C → A) (g : Coord R C → A) :
    run M index h g=rows (Source.scalar M) index h (List.finRange N) g := by
  unfold run rows
  simp only [oneRow_correct]

#axiomsClean chunk_cell
#axiomsClean oneRow_correct
#axiomsClean run_correct
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Program
