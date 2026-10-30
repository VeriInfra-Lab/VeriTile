import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOEvents

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOTrace
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def foldValues (ops : Ops I F H) (c : Config) (values : Fin c.S→H)
    (splits : List (Fin c.S)) (seed : Sum F H) : Sum F H :=
  splits.foldl (fun old z=>.inr (if c.S=1 then values z else ops.hadd (Quote.asH ops old) (values z))) seed

theorem fold_inr (ops : Ops I F H) (c : Config) (values : Fin c.S→H)
    (splits : List (Fin c.S)) (seed : H) :
    foldValues ops c values splits (.inr seed)=.inr (cellFold ops c values splits seed) := by
  induction splits generalizing seed with
  | nil=>rfl
  | cons z rest ih=>
    simp only [foldValues,cellFold,List.foldl_cons,Quote.asH]
    exact ih _

theorem fold_nonempty (ops : Ops I F H) (c : Config) (values : Fin c.S→H)
    (splits : List (Fin c.S)) (nonempty : splits≠[]) (seed : Sum F H) :
    foldValues ops c values splits seed=.inr (cellFold ops c values splits (Quote.asH ops seed)) := by
  cases splits with
  | nil=>exact False.elim (nonempty rfl)
  | cons z rest=>
    simp only [foldValues,cellFold,List.foldl_cons]
    exact fold_inr ops c values rest _

theorem trace (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (selected : Selection s c) (plans : q.plans=Quote.selectedPlan s c selected)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) (base : State (Sum F H))
    (cells : (IO.programIO q s c hasBias divisor schedule).InputCells vals base)
    (programs : (IO.programIO q s c hasBias divisor schedule).Programs vals base)
    (initial : (IO.programIO q s c hasBias divisor schedule).RawOutputs)
    (x : MatrixCell s) (splits : List (Fin c.S)) (before : Sum F H) :
    (IO.programIO q s c hasBias divisor schedule).SourceCellTrace vals base initial
      ⟨0,(x.1,x.2,PUnit.unit)⟩ (splits.map (fun z=>(IOPrograms.owner s c x,z))) before
        (foldValues q.ops c
          (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x)
          splits before) := by
  induction splits generalizing before with
  | nil=>
    exact IntegerDotPerAddressProgramIO.SourceCellTrace.nil
      (io:=IO.programIO q s c hasBias divisor schedule) (vals:=vals) (base:=base)
      (initial:=initial) (key:=⟨0,(x.1,x.2,PUnit.unit)⟩) before
  | cons z rest ih=>
    let io:=IO.programIO q s c hasBias divisor schedule
    let key : io.OutputKey:=⟨0,(x.1,x.2,PUnit.unit)⟩
    let event : IOPrograms.Program s c:=(IOPrograms.owner s c x,z)
    let next : Sum F H:=.inr (if c.S=1 then
      Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x z else
      q.ops.hadd (Quote.asH q.ops before)
        (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x z))
    change io.SourceCellTrace vals base initial key (event::rest.map (fun z=>(IOPrograms.owner s c x,z))) before
      (foldValues q.ops c
        (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) x) rest next)
    apply IntegerDotPerAddressProgramIO.SourceCellTrace.cons_of_programs io vals base initial key event _
      before next _ programs (IOGeometry.program_mem s c event)
    · intro final ran
      have bound:=IODataBinding.from_input_cells s c hasBias vals base cells
      have value:=IOInvocation.cell q s c hasBias divisor event vals _ final
        (IODataBinding.patch s c hasBias vals base bound (io.toMemoryIO.updateOutput initial key before))
        selected plans ran x
      rw [if_pos rfl,IOEvents.patch_cell,MemoryIO.updateOutput_same] at value
      change final.mem "C" (TileShape.linearIndex [s.M,s.N] (x.1,x.2,PUnit.unit))=Cell.mk .fp16 next
      simpa [TileShape.linearIndex,next,Quote.asH,Cell.read_same] using value
    · exact ih next

/-- Only finite independent instruction/split arithmetic appears in this
helper target. Public readers specialize it to Real or native formulas. -/
def result (q : Quote.IntegerModel I F H) (s : Shape) (c : Config) (hasBias : Bool)
    (divisor : F) (schedule : Schedule s c) (selected : Selection s c)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) :
    (IO.programIO q s c hasBias divisor schedule).Outputs :=
  Fin.cases (fun j=>cellFold q.ops c
    (Mathematical.splitValue q.ops s c selected divisor hasBias (IOLayout.data q.ops s vals) (j.1,j.2.1))
    (projection (j.1,j.2.1) schedule.events) q.ops.hzero) (fun i=>Fin.elim0 i)

theorem initial_zero (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (divisor : F) (schedule : Schedule s c)
    (vals : (IO.programIO q s c hasBias divisor schedule).Inputs) (base : State (Sum F H))
    (cells : (IO.programIO q s c hasBias divisor schedule).InputCells vals base)
    (initial : (IO.programIO q s c hasBias divisor schedule).RawOutputs)
    (output : (IO.programIO q s c hasBias divisor schedule).OutputCells vals initial base)
    (zero : IOLayout.domain q.ops s c vals) (split : c.S≠1) (x : MatrixCell s) :
    initial ⟨0,(x.1,x.2,PUnit.unit)⟩=.inr q.ops.hzero := by
  have input:=IOInputReads.output s c hasBias vals base cells x.1 x.2 split
  have out:=output ⟨0,(x.1,x.2,PUnit.unit)⟩ trivial
  change base.mem "C" (TileShape.linearIndex [s.M,s.N] (x.1,x.2,PUnit.unit))=
    Cell.mk .fp16 (initial ⟨0,(x.1,x.2,PUnit.unit)⟩) at out
  have address : TileShape.linearIndex [s.M,s.N] (x.1,x.2,PUnit.unit)=x.1.val*s.N+x.2.val := by
    simp [TileShape.linearIndex]
  rw [address,input,zero split x.1 x.2] at out
  exact (congrArg (fun cell=>cell.read .fp16) out).symm

#axiomsClean fold_inr
#axiomsClean fold_nonempty
#axiomsClean trace
#axiomsClean initial_zero
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOTrace
