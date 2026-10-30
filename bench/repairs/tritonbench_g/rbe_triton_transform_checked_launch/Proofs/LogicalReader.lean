import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.IO
namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
/-- Independent dense rotation on [batch,row,column], including odd tails. -/
def result (r : Request) (xs : LogicalView.Tensor r) (b : Fin r.batches)
    (m : Fin r.rows) (c : Fin r.columns) : ℝ :=
  let n := c.val/2*2
  let u := LogicalView.column r xs b m n
  let v := LogicalView.column r xs b m (n+1)
  let a := ((r.position : ℝ)+(m.val : ℝ))/Real.rpow 10000 ((n%128 : Nat)/(128 : ℝ))
  if decide (c.val%2=1) then u*Real.sin a+v*Real.cos a else u*Real.cos a-v*Real.sin a

theorem coordinates_at (r : Request) (b : Fin r.batches) (m : Fin r.rows) (c : Fin r.columns) :
    coordinates r ((LogicalView.indices r) ((b,m),c)).val = (b.val,m.val,c.val) := by
  have hc : 0<r.columns := Nat.zero_lt_of_lt c.isLt
  have hr : 0<r.rows := Nat.zero_lt_of_lt m.isLt
  have val : ((LogicalView.indices r) ((b,m),c)).val=r.offset b.val m.val c.val := by
    change c.val+r.columns*(m.val+r.rows*b.val)=(b.val*r.rows+m.val)*r.columns+c.val
    ring
  rw [val]
  change ((r.offset b.val m.val c.val)/(r.rows*r.columns),
    ((r.offset b.val m.val c.val)/r.columns)%r.rows,
    (r.offset b.val m.val c.val)%r.columns)=(b.val,m.val,c.val)
  have divc : (r.offset b.val m.val c.val)/r.columns=b.val*r.rows+m.val := by
    unfold Request.offset
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hc,Nat.div_eq_of_lt c.isLt]
    simp
  have divr : (b.val*r.rows+m.val)/r.rows=b.val := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hr,Nat.div_eq_of_lt m.isLt]
    simp
  rw [Nat.mul_comm r.rows r.columns,←Nat.div_div_eq_div_mul,divc,divr]
  simp [Request.offset,Nat.add_mod,Nat.mod_eq_of_lt c.isLt,Nat.mod_eq_of_lt m.isLt]

theorem input_at (r : Request) (X : RegionName)
    (xs : ChannelValues ℝ [⟨.fp32,[r.size],X⟩])
    (b : Fin r.batches) (m : Fin r.rows) (c : Fin r.columns) :
    logicalInput (α:=ℝ) r (xs 0) (r.offset b.val m.val c.val)=(LogicalView.tensorEquiv r X xs) b m c := by
  have h := offset_lt r b.val m.val c.val b.isLt m.isLt c.isLt
  rw [logicalInput, dif_pos h]
  congr 2
  apply Fin.ext
  change (b.val*r.rows+m.val)*r.columns+c.val=c.val+r.columns*(m.val+r.rows*b.val)
  ring

theorem judgment_iff (r : Request) (X Y : RegionName) (qs : List Runtime.Program) :
    (LogicalView.rotationIO r X Y qs).Implements (result r) ↔
    (ReaderBefore.realIO r X Y qs).Implements (fun xs => packOutput r Y (ReaderBefore.result r (xs ⟨0,by change 0<1; decide⟩))) := by
  unfold LogicalProgramIO.Implements LogicalProgramIO.lift
  apply Iff.of_eq
  congr 1
  funext xs o j
  have ho : o=⟨0,by change 0<1; decide⟩ := Fin.eq_zero o
  subst o
  obtain ⟨coords,hcoords⟩ := (LogicalView.indices r).surjective j.1
  rcases coords with ⟨⟨b,m⟩,c⟩
  have hj : j=((LogicalView.indices r) ((b,m),c),()) := by rw [hcoords]; rfl
  subst j
  change result r (LogicalView.tensorEquiv r X xs)
    ((LogicalView.indices r).symm ((LogicalView.indices r) ((b,m),c))).1.1
    ((LogicalView.indices r).symm ((LogicalView.indices r) ((b,m),c))).1.2
    ((LogicalView.indices r).symm ((LogicalView.indices r) ((b,m),c))).2 =
    ReaderBefore.result r (xs ⟨0,by change 0<1; decide⟩) ((LogicalView.indices r) ((b,m),c),())
  rw [Equiv.symm_apply_apply]
  simp only [ReaderBefore.result,coordinates_at]
  have hn : c.val/2*2<r.columns := lt_of_le_of_lt (Nat.div_mul_le_self _ 2) c.isLt
  dsimp only [result]
  rw [LogicalView.column,dif_pos hn,←input_at]
  by_cases hv : c.val/2*2+1<r.columns
  · rw [LogicalView.column,dif_pos hv,←input_at,if_pos hv]
    rfl
  · rw [LogicalView.column,dif_neg hv,if_neg hv]
    rfl
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.LogicalReader
