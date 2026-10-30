import bench.tritonbench_g.triton_attention.Proofs.GridAddresses
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def preProgram (c : Shape) (i : Coord c) : Program := (rowOffset c i.1 i.2.1/128,0)
def preLane (c : Shape) (i : Coord c) : PreLane (preParameters c) :=
  (⟨rowOffset c i.1 i.2.1%128,Nat.mod_lt _ (by decide)⟩,i.2.2,())

theorem preProgram_legal (c : Shape) (i : Coord c) : Legal (preprocessGrid c) (preProgram c i) := by
  have hr:=rowOffset_lt c i.1 i.2.1
  have hn : c.groups*c.rows=(c.blocks*c.groups)*128 := by unfold Shape.rows;ring
  rw [hn] at hr
  constructor
  · change rowOffset c i.1 i.2.1/128<c.blocks*c.groups
    omega
  · change 0<1;decide

theorem preOffset_eq (c : Shape) (pid : Program) (mem : Memory α) (i : PreLane (preParameters c)) :
    preOffset (preParameters c) (entry (preprocessGrid c) pid mem) i=(pid.1*128+i.1.val)*c.dim+i.2.1.val := by
  rfl

theorem preRow_eq (c : Shape) (pid : Program) (mem : Memory α) (i : PreRow (preParameters c)) :
    preRow (preParameters c) (entry (preprocessGrid c) pid mem) i=pid.1*128+i.1.val := by
  rfl

theorem pre_target_offset (c : Shape) (mem : Memory α) (i : Coord c) :
    preOffset (preParameters c) (entry (preprocessGrid c) (preProgram c i) mem) (preLane c i)=offset c i := by
  rw [preOffset_eq]
  simp [preProgram,preLane,Nat.div_add_mod',rowOffset,offset]

theorem pre_target_row (c : Shape) (mem : Memory α) (i : Coord c) :
    preRow (preParameters c) (entry (preprocessGrid c) (preProgram c i) mem) ((preLane c i).1,())=rowOffset c i.1 i.2.1 := by
  rw [preRow_eq]
  simp [preProgram,preLane,Nat.div_add_mod']

theorem pre_other_row (c : Shape) (pid : Program) (mem : Memory α) (hp : Legal (preprocessGrid c) pid)
    (i : Coord c) (hne : pid≠preProgram c i) (lane : PreRow (preParameters c)) :
    rowOffset c i.1 i.2.1≠preRow (preParameters c) (entry (preprocessGrid c) pid mem) lane := by
  intro he
  rw [preRow_eq] at he
  apply hne
  apply Prod.ext
  · change pid.1=rowOffset c i.1 i.2.1/128
    have hl : lane.1.val<128 := lane.1.isLt
    rw [he,Nat.add_comm,Nat.add_mul_div_right _ _ (by decide : 0<128),Nat.div_eq_of_lt hl,Nat.zero_add]
  · have hh:=hp.2
    change pid.2<1 at hh
    change pid.2=0
    omega

theorem pre_other_offset (c : Shape) (pid : Program) (mem : Memory α) (hp : Legal (preprocessGrid c) pid)
    (i : Coord c) (hne : pid≠preProgram c i) (lane : PreLane (preParameters c)) :
    offset c i≠preOffset (preParameters c) (entry (preprocessGrid c) pid mem) lane := by
  intro he
  have hrow : rowOffset c i.1 i.2.1=preRow (preParameters c) (entry (preprocessGrid c) pid mem) (lane.1,()) := by
    have hc:=congrArg (· / c.dim) he
    rw [preOffset_eq] at hc
    simp only [offset] at hc
    have divCol (r : Nat) (d : Fin c.dim) : (r*c.dim+d.val)/c.dim=r := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.dim_pos,Nat.div_eq_of_lt d.isLt,Nat.zero_add]
    simp only [divCol] at hc
    exact hc
  exact pre_other_row c pid mem hp i hne (lane.1,()) hrow

theorem preOffset_injective (c : Shape) (pid : Program) (mem : Memory α) :
    Function.Injective (preOffset (preParameters c) (entry (preprocessGrid c) pid mem)) := by
  intro i j he
  rw [preOffset_eq,preOffset_eq] at he
  have hi : i.2.1.val<c.dim := i.2.1.isLt
  have hj : j.2.1.val<c.dim := j.2.1.isLt
  have hcol : i.2.1=j.2.1 := by
    apply Fin.ext
    have hh:=congrArg (· % c.dim) he
    simpa only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hi,Nat.mod_eq_of_lt hj] using hh
  have hrow : i.1=j.1 := by
    apply Fin.ext
    have hh:=congrArg Fin.val hcol
    have hd:=c.dim_pos
    nlinarith
  exact Prod.ext hrow (Prod.ext hcol (Subsingleton.elim _ _))

theorem preprocessStep_otherNewDO (P : Primitives α) (c : Shape) (pid : Program) (mem : Memory α)
    (hp : Legal (preprocessGrid c) pid) (i : Coord c) (hne : pid≠preProgram c i) :
    preprocessStep P c pid mem "NewDO" (offset c i)=mem "NewDO" (offset c i) := by
  unfold preprocessStep preWrite
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl (by simp [preParameters]))]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inr (by intro lane hl;exact pre_other_offset c pid mem hp i hne lane)) _

theorem preprocessStep_otherDelta (P : Primitives α) (c : Shape) (pid : Program) (mem : Memory α)
    (hp : Legal (preprocessGrid c) pid) (i : Coord c) (hne : pid≠preProgram c i) :
    preprocessStep P c pid mem "Delta" (rowOffset c i.1 i.2.1)=mem "Delta" (rowOffset c i.1 i.2.1) := by
  unfold preprocessStep preWrite
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inr (by intro lane hl;exact pre_other_row c pid mem hp i hne lane))]
  exact State.scatter_frame _ _ _ _ _ _ (Or.inl (by simp [preParameters])) _

#axiomsClean pre_target_offset
#axiomsClean pre_other_offset
#axiomsClean preprocessStep_otherNewDO
#axiomsClean preprocessStep_otherDelta
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
