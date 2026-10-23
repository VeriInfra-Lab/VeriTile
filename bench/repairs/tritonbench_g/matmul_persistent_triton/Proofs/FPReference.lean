import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.HostFrame
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.DotPlans

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
open Host
noncomputable section
set_option maxHeartbeats 5000000

/-- Reference starts from caller A/B coordinates and iterates all K blocks.
It contains no PID, launch result, final memory or numerical oracle. -/
def referenceA (args : Arguments) (entry : Memory Expr) (rowBlock iteration : Nat)
    (row : Fin 128) (lane : Fin args.A.dtype.blockK) : Expr :=
  let k := iteration*args.A.dtype.blockK+lane.val
  if k<args.A.columns then
    entry (args.A.address (clamp args.A.rows (rowBlock*128+row.val)) (clamp args.A.columns k))
  else .zero

def referenceB (args : Arguments) (entry : Memory Expr) (columnBlock iteration : Nat)
    (lane : Fin args.A.dtype.blockK) (column : Fin 256) : Expr :=
  let k := iteration*args.A.dtype.blockK+lane.val
  if k<args.A.columns then
    entry (args.B.address (clamp args.A.columns k) (clamp args.B.columns (columnBlock*256+column.val)))
  else .zero

def referenceAcc (args : Arguments) (entry : Memory Expr) (rowBlock columnBlock : Nat) : Nat → Source.Acc Expr
  | 0 => fun _ _ => .zero
  | count+1 => Expr.dot args.A.dtype.blockK (referenceA args entry rowBlock count)
      (referenceB args entry columnBlock count) (referenceAcc args entry rowBlock columnBlock count)

def reference (args : Arguments) (entry : Memory Expr) (row column : Nat) : Expr :=
  if args.A.columns=0 then .zero
  else .cast args.A.dtype
    (referenceAcc args entry (row/128) (column/256) (args.problem 1).kTiles
      ⟨row%128,Nat.mod_lt _ (by decide)⟩ ⟨column%256,Nat.mod_lt _ (by decide)⟩)

theorem accumulator_reference (args : Arguments) (sms : Nat) (output : Tensor)
    (entry : Memory Expr) (tile count : Nat) :
    Source.accumulator arithmetic (args.problem sms) ⟨args.A,args.B,output⟩ entry tile count=
      referenceAcc args entry ((args.problem sms).tileRow tile) ((args.problem sms).tileColumn tile) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [Source.accumulator,referenceAcc,ih]
    rfl

theorem Completion.output_reference {σ : Type} {runtime : Runtime σ} {live : σ → Nat → Prop}
    {args : Arguments} {initial final : World Expr σ} {output : Tensor}
    (e : Host.Completion arithmetic runtime live args initial final output)
    (row column : Nat) (hr : row<args.A.rows) (hc : column<args.B.columns) :
    final.memory (output.address row column)=reference args initial.memory row column := by
  cases e.branch with
  | empty empty same =>
    change args.A.rows=0 ∨ args.B.columns=0 at empty
    omega
  | zero nonempty zeroK native executed same =>
    rw [same]
    have active : outputCell (args.problem e.sms) ⟨args.A,args.B,output⟩ (output.address row column) :=
      ⟨row,hr,column,hc,rfl⟩
    change args.A.columns=0 at zeroK
    simp only [zeroMemory,if_pos active,reference,zeroK,if_pos rfl]
    rfl
  | launch nonempty positiveK native executed same =>
    rw [same]
    obtain ⟨tile,ht,lane,ha,hrow,hcolumn⟩ := Source.output_covered (args.problem e.sms) e.smsPositive row column hr hc
    have address : Source.outputAddress (args.problem e.sms) ⟨args.A,args.B,output⟩ tile lane=output.address row column := by
      rw [Source.output_address _ _ tile lane ha,hrow,hcolumn]
    change Source.run arithmetic (args.problem e.sms) ⟨args.A,args.B,output⟩ initial.memory (output.address row column)=_
    rw [← address,Source.source_output arithmetic _ _ e.layout e.smsPositive initial.memory tile ht lane ha]
    change args.A.columns≠0 at positiveK
    change Expr.cast args.A.dtype (Source.accumulator arithmetic (args.problem e.sms) _ initial.memory tile
      (args.problem e.sms).kTiles lane.1 lane.2)=_
    rw [accumulator_reference,reference,if_neg positiveK]
    have rowBlock : (args.problem e.sms).tileRow tile=row/128 := by
      have he := congrArg (fun x => x/128) hrow
      simpa only [Source.row_recover] using he
    have columnBlock : (args.problem e.sms).tileColumn tile=column/256 := by
      have he := congrArg (fun x => x/256) hcolumn
      simpa only [Source.column_recover] using he
    have rowLane : lane.1.val=row%128 := by
      have he := congrArg (fun x => x%128) hrow
      simpa [Problem.row,Nat.add_mod,Nat.mod_eq_of_lt lane.1.isLt] using he
    have columnLane : lane.2.val=column%256 := by
      have he := congrArg (fun x => x%256) hcolumn
      simpa [Problem.column,Nat.add_mod,Nat.mod_eq_of_lt lane.2.isLt] using he
    rw [rowBlock,columnBlock]
    congr 2
    · exact Fin.ext rowLane
    · exact Fin.ext columnLane

theorem reference_computed (args : Arguments) (entry : Memory Expr) (row column : Nat) (word : Word) :
    reference args entry row column≠.raw word := by
  unfold reference
  split_ifs <;> intro equal <;> cases equal

#axiomsClean Completion.output_reference
#specNonCircular reference avoiding [Host.run,Source.run,Plan,Plan.graph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
