import bench.tritonbench_g.triton_attention.Proofs.GridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def backwardProgram (c : Shape) (i : Coord c) : Program := (i.1.val,0)
def backwardLane (c : Shape) (scale : ℝ) (i : Coord c) : BLane (backwardParameters c scale) :=
  (⟨i.2.1.val%128,Nat.mod_lt _ (by decide)⟩,i.2.2,())

theorem backwardProgram_legal (c : Shape) (i : Coord c) : Legal (backwardGrid c) (backwardProgram c i) :=
  ⟨i.1.isLt,by change 0<1;decide⟩

theorem backwardOffset_eq (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α) (b : Nat)
    (i : BLane (backwardParameters c scale)) :
    backwardOffset (backwardParameters c scale) (entry (backwardGrid c) pid mem) b i=
      (pid.1*c.rows+b*128+i.1.val)*c.dim+i.2.1.val := by
  rw [backwardOffset,backwardHead_eq]
  rfl

theorem backward_target_offset (c : Shape) (scale : ℝ) (mem : Memory α) (i : Coord c) :
    backwardOffset (backwardParameters c scale) (entry (backwardGrid c) (backwardProgram c i) mem)
      (i.2.1.val/128) (backwardLane c scale i)=offset c i := by
  rw [backwardOffset_eq]
  simp [backwardProgram,backwardLane,offset,Nat.add_assoc,Nat.div_add_mod']

theorem backward_other_offset (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (backwardGrid c) pid) (i : Coord c) (hne : pid≠backwardProgram c i)
    (b : Nat) (hb : b<c.blocks) (lane : BLane (backwardParameters c scale)) :
    offset c i≠backwardOffset (backwardParameters c scale) (entry (backwardGrid c) pid mem) b lane := by
  intro he
  have hrow : i.1.val*c.rows+i.2.1.val=pid.1*c.rows+b*128+lane.1.val := by
    have hc:=congrArg (· / c.dim) he
    rw [backwardOffset_eq] at hc
    simp only [offset] at hc
    have divCol (r : Nat) (d : Fin c.dim) : (r*c.dim+d.val)/c.dim=r := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.dim_pos,Nat.div_eq_of_lt d.isLt,Nat.zero_add]
    simpa only [divCol] using hc
  have hd : (i.1.val*c.rows+i.2.1.val)/c.rows=i.1.val := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.rows_pos,Nat.div_eq_of_lt i.2.1.isLt,Nat.zero_add]
  have hp0:=hp.1
  have hp1:=hp.2
  have ha:=(row_decode c (b,pid.1) ⟨hb,hp0⟩ lane.1).1
  change (pid.1*c.rows+b*128+lane.1.val)/c.rows=pid.1 at ha
  apply hne
  apply Prod.ext
  · change pid.1=i.1.val
    rw [←ha,←hd,hrow]
  · change pid.2=0
    change pid.2<1 at hp1
    omega

theorem backwardStep_other (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (backwardGrid c) pid) (i : Coord c) (hne : pid≠backwardProgram c i) (r : RegionName) :
    backwardStep P c scale pid mem r (offset c i)=mem r (offset c i) := by
  apply backwardFrame
  intro hw
  rcases hw with ⟨_,b,hb,lane,he⟩|⟨_,b,hb,lane,_,he⟩ <;>
    exact backward_other_offset c scale pid mem hp i hne b hb lane he

theorem backwardStep_readonly (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (r : RegionName) (o : Nat) (hr : r≠"DQ" ∧ r≠"DK" ∧ r≠"DV") :
    backwardStep P c scale pid mem r o=mem r o := by
  apply backwardFrame
  intro hw
  rcases hw with ⟨he,_⟩|⟨he,_⟩
  · exact hr.1 he
  · rcases he with he|he
    · exact hr.2.1 he
    · exact hr.2.2 he

#axiomsClean backward_target_offset
#axiomsClean backwardStep_other
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
