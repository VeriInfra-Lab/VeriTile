import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.HostFrame
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.DotPlans

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
open Host
noncomputable section
set_option maxHeartbeats 5000000

/-- Reference starts from caller A/B coordinates and iterates all K blocks.
It contains no PID, launch result, final memory or numerical oracle. -/
def referenceA (args : Arguments) (entry : Memory Expr) (rowBlock iteration : Nat)
    (row : Fin 128) (lane : Fin args.A.dtype.blockK) : Expr :=
  let k := iteration*args.A.dtype.blockK+lane.val
  if k<args.A.columns then
    entry (args.A.address ((rowBlock*128+row.val)%args.A.rows) (clamp args.A.columns k))
  else .zero

def referenceB (args : Arguments) (entry : Memory Expr) (columnBlock iteration : Nat)
    (lane : Fin args.A.dtype.blockK) (column : Fin 256) : Expr :=
  let k := iteration*args.A.dtype.blockK+lane.val
  if k<args.A.columns then
    entry (args.B.address (clamp args.A.columns k) ((columnBlock*256+column.val)%args.B.columns))
  else .zero

def referenceAcc (args : Arguments) (entry : Memory Expr) (rowBlock columnBlock : Nat) : Nat → Source.Acc Expr
  | 0 => fun _ _ => .zero
  | count+1 => Expr.dot args.A.dtype.blockK (referenceA args entry rowBlock count)
      (referenceB args entry columnBlock count) (referenceAcc args entry rowBlock columnBlock count)

def reference (args : Arguments) (entry : Memory Expr) (row column : Nat) : Expr :=
  if args.A.columns=0 then .zero
  else .cast (arithmetic.activation args.problem.enabled
    (referenceAcc args entry (row/128) (column/256) args.problem.kTiles
      ⟨row%128,Nat.mod_lt _ (by decide)⟩ ⟨column%256,Nat.mod_lt _ (by decide)⟩))

theorem accumulator_reference (args : Arguments) (output : Tensor)
    (entry : Memory Expr) (tile count : Nat) :
    Source.accumulator arithmetic args.problem ⟨args.A,args.B,output⟩ entry tile count=
      referenceAcc args entry (args.problem.tileRow tile) (args.problem.tileColumn tile) count := by
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
    have active : outputCell args.problem ⟨args.A,args.B,output⟩ (output.address row column) :=
      ⟨row,hr,column,hc,rfl⟩
    change args.A.columns=0 at zeroK
    simp only [zeroMemory,if_pos active,reference,zeroK,if_pos rfl]
    rfl
  | launch nonempty positiveK trials selected executed =>
    have memory := (sequence_memory arithmetic runtime args output (trials+1) _ final executed).1
    rw [memory,Source.replay_positive arithmetic args.problem _ e.layout trials initial.memory]
    obtain ⟨tile,ht,lane,ha,hrow,hcolumn⟩ := Source.output_covered args.problem row column hr hc
    have address : Source.outputAddress args.problem ⟨args.A,args.B,output⟩ tile lane=output.address row column := by
      rw [Source.output_address _ _ tile lane ha,hrow,hcolumn]
    change Source.run arithmetic args.problem ⟨args.A,args.B,output⟩ initial.memory (output.address row column)=_
    rw [← address,Source.source_output arithmetic _ _ e.layout initial.memory tile ht lane ha]
    change args.A.columns≠0 at positiveK
    change Expr.cast (arithmetic.activation args.problem.enabled (Source.accumulator arithmetic args.problem _ initial.memory tile
      args.problem.kTiles lane.1 lane.2))=_
    rw [accumulator_reference,reference,if_neg positiveK]
    have rowBlock : args.problem.tileRow tile=row/128 := by
      have he := congrArg (fun x => x/128) hrow
      simpa only [Source.row_recover] using he
    have columnBlock : args.problem.tileColumn tile=column/256 := by
      have he := congrArg (fun x => x/256) hcolumn
      simpa only [Source.column_recover] using he
    have rowLane : lane.1.val=row%128 := by
      have he := congrArg (fun x => x%128) hrow
      simpa [Problem.row,Nat.add_mod,Nat.mod_eq_of_lt lane.1.isLt] using he
    have columnLane : lane.2.val=column%256 := by
      have he := congrArg (fun x => x%256) hcolumn
      simpa [Problem.column,Nat.add_mod,Nat.mod_eq_of_lt lane.2.isLt] using he
    rw [rowBlock,columnBlock]
    have rowFin : lane.1=⟨row%128,Nat.mod_lt _ (by decide)⟩ := Fin.ext rowLane
    have columnFin : lane.2=⟨column%256,Nat.mod_lt _ (by decide)⟩ := Fin.ext columnLane
    rw [rowFin,columnFin]

theorem reference_computed (args : Arguments) (entry : Memory Expr) (row column : Nat) (word : Word) :
    reference args entry row column≠.raw word := by
  unfold reference
  split_ifs <;> intro equal <;> cases equal

#axiomsClean Completion.output_reference
#specNonCircular reference avoiding [Host.run,Source.run,Plan,Plan.graph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
