import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridCoverage
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def forwardProgram (c : Shape) (i : Coord c) : Program := (i.2.1.val/128,i.1.val)
def forwardLane (c : Shape) (scale : ℝ) (i : Coord c) : FQuery (forwardParameters c scale) :=
  (⟨i.2.1.val%128,Nat.mod_lt _ (by decide)⟩,i.2.2,())

theorem forwardProgram_legal (c : Shape) (i : Coord c) : Legal (forwardGrid c) (forwardProgram c i) := by
  exact ⟨(grid_coverage c i.2.1).1,i.1.isLt⟩

theorem forwardOffset_eq (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (i : FQuery (forwardParameters c scale)) :
    forwardOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) i=
      (pid.2*c.rows+pid.1*128+i.1.val)*c.dim+i.2.1.val := by
  simp [forwardOffset,forwardHeadRow,forwardQueryRow,forwardParameters,entry,ids,Nat.add_assoc,c.dim_pos]

theorem forwardRow_eq (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (i : FRow (forwardParameters c scale)) :
    forwardRowOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) i=
      pid.2*c.rows+pid.1*128+i.1.val := by
  simp [forwardRowOffset,forwardQueryRow,forwardParameters,entry,ids,Nat.add_assoc]

theorem forward_target_offset (c : Shape) (scale : ℝ) (mem : Memory α) (i : Coord c) :
    forwardOffset (forwardParameters c scale) (entry (forwardGrid c) (forwardProgram c i) mem)
      (forwardLane c scale i)=offset c i := by
  rw [forwardOffset_eq]
  simp [forwardProgram,forwardLane,offset,Nat.add_assoc,Nat.div_add_mod']

theorem forward_target_row (c : Shape) (scale : ℝ) (mem : Memory α) (i : Coord c) :
    forwardRowOffset (forwardParameters c scale) (entry (forwardGrid c) (forwardProgram c i) mem)
      ((forwardLane c scale i).1,())=rowOffset c i.1 i.2.1 := by
  rw [forwardRow_eq]
  simp [forwardProgram,forwardLane,rowOffset,Nat.add_assoc,Nat.div_add_mod']

theorem row_decode (c : Shape) (pid : Program) (hp : Legal (forwardGrid c) pid)
    (lane : Fin 128) :
    (pid.2*c.rows+pid.1*128+lane.val)/c.rows=pid.2 ∧
      ((pid.2*c.rows+pid.1*128+lane.val)%c.rows)/128=pid.1 := by
  have hr : pid.1*128+lane.val<c.rows := by
    have hh:=hp.1
    change pid.1<c.blocks at hh
    change _<c.blocks*128
    omega
  have hdiv : (pid.2*c.rows+(pid.1*128+lane.val))/c.rows=pid.2 := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.rows_pos,Nat.div_eq_of_lt hr,Nat.zero_add]
  have hmod : (pid.2*c.rows+(pid.1*128+lane.val))%c.rows=pid.1*128+lane.val := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hr]
  constructor
  · simpa only [Nat.add_assoc] using hdiv
  · rw [Nat.add_assoc,hmod,Nat.add_comm,Nat.add_mul_div_right _ _ (by decide : 0<128),
      Nat.div_eq_of_lt lane.isLt,Nat.zero_add]

theorem forward_other_row (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : Coord c) (hne : pid≠forwardProgram c i)
    (lane : FRow (forwardParameters c scale)) :
    rowOffset c i.1 i.2.1≠forwardRowOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) lane := by
  intro he
  apply hne
  rw [forwardRow_eq] at he
  have ht:=forward_target_row c scale mem i
  rw [forwardRow_eq] at ht
  have hrows : pid.2*c.rows+pid.1*128+lane.1.val=
      (forwardProgram c i).2*c.rows+(forwardProgram c i).1*128+(forwardLane c scale i).1.val := he.symm.trans ht.symm
  have a:=row_decode c pid hp lane.1
  have b:=row_decode c (forwardProgram c i) (forwardProgram_legal c i) (forwardLane c scale i).1
  apply Prod.ext
  · rw [←a.2,←b.2,hrows]
  · rw [←a.1,←b.1,hrows]

theorem forward_other_offset (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : Coord c) (hne : pid≠forwardProgram c i)
    (lane : FQuery (forwardParameters c scale)) :
    offset c i≠forwardOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) lane := by
  intro he
  have hrow : rowOffset c i.1 i.2.1=forwardRowOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) (lane.1,()) := by
    have hc:=congrArg (· / c.dim) he
    rw [forwardOffset_eq] at hc
    simp only [offset] at hc
    have divCol (r : Nat) (d : Fin c.dim) : (r*c.dim+d.val)/c.dim=r := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.dim_pos,Nat.div_eq_of_lt d.isLt,Nat.zero_add]
    simp only [divCol] at hc
    simpa only [forwardRow_eq,rowOffset] using hc
  exact forward_other_row c scale pid mem hp i hne (lane.1,()) hrow

#axiomsClean forward_target_offset
#axiomsClean forward_other_offset
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
