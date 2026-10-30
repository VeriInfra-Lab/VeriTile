import bench.tritonbench_g.kldiv_triton.Proofs.ForwardIO
import bench.tritonbench_g.kldiv_triton.Proofs.BackwardLoop
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.RawGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount rounds)
noncomputable section
set_option maxHeartbeats 5000000

abbrev Coord (rows cols : Nat) := Fin rows × Fin cols
def address {rows cols : Nat} (stride : Nat) (x : Coord rows cols) := x.1.val*stride+x.2.val

def forwardValue {α : Type} [Inhabited α] (m : Tile.Model α) (s : State α)
    (sy st N B : Nat) (flag : Bool) (reduction r c : Nat) : α :=
  if reduction=0 then Tile.loss m flag
    (m.alg.fp32Load (Tile.input s "y_ptr" (r*sy) c))
    (m.alg.fp32Load (Tile.input s "gt_ptr" (r*st) c))
  else Tile.chunkSum m flag N B c (Tile.input s "y_ptr" (r*sy)) (Tile.input s "gt_ptr" (r*st))
def backwardValue {α : Type} [Inhabited α] (m : Tile.Model α) (s : State α)
    (st : Nat) (flag : Bool) (r c : Nat) : α :=
  Tile.gradient m flag (m.alg.fp32Load (Tile.input s "target_ptr" (r*st) c))

def forwardRun {α : Type} [Inhabited α] (m : Tile.Model α) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) :=
  GridExecution.sequence (fun pid s => FP.Structural.exec m.alg
    (forwardSource sy st so N B flag reduction) (GridExecution.atProgram s pid))
def backwardRun {α : Type} [Inhabited α] (m : Tile.Model α) (so st N B : Nat) (flag : Bool) :=
  GridExecution.sequence (fun pid s => FP.Structural.exec m.alg
    (backwardSource so st N B flag) (GridExecution.atProgram s pid))

private theorem whole {α : Type} (rows cols stride : Nat) (output : RegionName)
    (unique : Function.Injective (address (rows:=rows) (cols:=cols) stride))
    (before : State α) (value : Coord rows cols → Cell α)
    (one : Nat → State α → Option (State α))
    (single : ∀ pid,pid<rows → ∀ current,GridExecution.Outside output before current →
      ∃ final,one pid current=some final ∧ GridExecution.Post output (address stride) value
        (fun x : Coord rows cols => x.1.val=pid) current final)
    (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,GridExecution.sequence one schedule before=some final ∧
      GridExecution.Post output (address stride) value (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ := GridExecution.compose output (address stride) unique value
    (fun pid (x : Coord rows cols) => x.1.val=pid) one before rows single schedule
    (fun pid hp => List.mem_range.mp (order.mem_iff.mp hp)) before (fun _ _ _ => rfl)
  refine ⟨final,ran,?_,?_⟩
  · intro x _
    exact values x ⟨x.1.val,order.mem_iff.mpr (List.mem_range.mpr x.1.isLt),rfl⟩
  · intro r o outside
    exact frame r o (outside.imp id (fun h x _ => h x trivial))

theorem forward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (unique : Function.Injective (address (rows:=rows) (cols:=outputCount N B reduction) so))
    (before : State α) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,forwardRun m sy st so N B flag reduction schedule before=some final ∧
      GridExecution.Post "loss_ptr" (address (rows:=rows) (cols:=outputCount N B reduction) so)
        (fun x => .mk .real (forwardValue m before sy st N B flag reduction x.1.val x.2.val))
        (fun _ => True) before final := by
  apply whole rows (outputCount N B reduction) so "loss_ptr" unique before _ _ _ schedule order
  intro pid hp current outside
  let s := GridExecution.atProgram current pid
  let result := Tile.sourceRun m s sy st so N B flag reduction positive
  refine ⟨result.val,result.property,?_,?_⟩
  · intro x owns
    have inputs (region : RegionName) (hr : region≠"loss_ptr") (base : Nat) :
        Tile.input s region base=Tile.input before region base := by
      funext c
      exact congrArg (fun cell : Cell α => cell.read .real) (outside region (base+c) hr)
    by_cases mode : reduction=0
    · subst reduction
      have active : x.2.val<N := by simpa [outputCount] using x.2.isLt
      have value := Tile.source_element m s sy st so N B flag positive x.2.val active
      rw [show s.pids 0=pid from rfl,inputs "y_ptr" (by decide) (pid*sy),
        inputs "gt_ptr" (by decide) (pid*st)] at value
      simpa only [result,address,owns,forwardValue,if_pos rfl] using value
    · have active : x.2.val<rounds N B := by simpa [outputCount,mode] using x.2.isLt
      have value := Tile.source_chunk m s sy st so N B flag reduction positive mode x.2.val active
      change result.val.mem "loss_ptr" (x.1.val*so+x.2.val)=_
      rw [owns]
      rw [show s.pids 0=pid from rfl,inputs "y_ptr" (by decide) (pid*sy),
        inputs "gt_ptr" (by decide) (pid*st)] at value
      simpa only [result,owns,forwardValue,if_neg mode] using value
  · intro r o miss
    apply Tile.source_frame m s sy st so N B flag reduction positive r o
    intro hit
    have hit' : r="loss_ptr" ∧ pid*so≤o ∧ o<pid*so+outputCount N B reduction := by
      simpa only [Tile.RowFootprint,s,GridExecution.atProgram,ite_true] using hit
    rcases miss with different | missed
    · exact different hit'.1
    · let x : Coord rows (outputCount N B reduction) := (⟨pid,hp⟩,⟨o-pid*so,by omega⟩)
      exact missed x rfl (by dsimp [address,x];omega)

theorem backward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (address (rows:=rows) (cols:=N) so))
    (before : State α) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,backwardRun m so st N B flag schedule before=some final ∧
      GridExecution.Post "input_ptr" (address (rows:=rows) (cols:=N) so)
        (fun x => .mk .real (backwardValue m before st flag x.1.val x.2.val))
        (fun _ => True) before final := by
  apply whole rows N so "input_ptr" unique before _ _ _ schedule order
  intro pid hp current outside
  let s := GridExecution.atProgram current pid
  let result := Backward.sourceRun m s so st N B flag positive
  refine ⟨result.val,result.property,?_,?_⟩
  · intro x owns
    have value := Backward.source_cell m s so st N B flag positive x.2.val x.2.isLt
    change result.val.mem "input_ptr" (x.1.val*so+x.2.val)=_
    rw [owns]
    have read := congrArg (fun cell : Cell α => cell.read .real)
      (outside "target_ptr" (pid*st+x.2.val) (by decide))
    simpa only [result,s,GridExecution.atProgram,ite_true,backwardValue,Tile.input,read,owns] using value
  · intro r o miss
    apply Backward.source_frame m s so st N B flag positive r o
    by_cases diff : r≠"input_ptr"
    · exact Or.inl diff
    · have missed := miss.resolve_left diff
      have out : o<pid*so ∨ pid*so+N≤o := by
        by_contra h
        let x : Coord rows N := (⟨pid,hp⟩,⟨o-pid*so,by omega⟩)
        exact missed x rfl (by dsimp [address,x];omega)
      exact Or.inr out

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.RawGrid
