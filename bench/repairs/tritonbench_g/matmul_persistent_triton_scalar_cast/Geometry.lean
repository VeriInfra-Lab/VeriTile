import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Targets

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 7000000

abbrev LogicalIndex (p : Parameters) := Fin p.problem.M × Fin p.problem.N

def logicalAddress (p : Parameters) (x : LogicalIndex p) : Nat := p.cm*x.1.val+p.cn*x.2.val

def Owns (p : Parameters) (tile : Nat) (x : LogicalIndex p) : Prop :=
  ∃ i : CIndex,p.problem.row tile i.1.val=x.1.val ∧ p.problem.column tile i.2.1.val=x.2.val

def index (p : Parameters) (tile : Nat) (i : CIndex) (h : active p tile i) : LogicalIndex p :=
  (⟨p.problem.row tile i.1.val,h.1⟩,⟨p.problem.column tile i.2.1.val,h.2⟩)

theorem lane_owns (p : Parameters) (tile : Nat) (i : CIndex) (h : active p tile i) :
    Owns p tile (index p tile i h) := ⟨i,rfl,rfl⟩

theorem address_index (p : Parameters) (tile : Nat) (i : CIndex) (h : active p tile i) :
    cOffset p tile i=logicalAddress p (index p tile i h) := by
  simp [cOffset,logicalAddress,index,clamp,h.1,h.2]

theorem owned_active (p : Parameters) (tile : Nat) (x : LogicalIndex p) (i : CIndex)
    (hr : p.problem.row tile i.1.val=x.1.val) (hc : p.problem.column tile i.2.1.val=x.2.val) :
    active p tile i := ⟨hr ▸ x.1.isLt,hc ▸ x.2.isLt⟩

theorem owned_address (p : Parameters) (tile : Nat) (x : LogicalIndex p) (i : CIndex)
    (hr : p.problem.row tile i.1.val=x.1.val) (hc : p.problem.column tile i.2.1.val=x.2.val) :
    cOffset p tile i=logicalAddress p x := by
  simp [cOffset,logicalAddress,hr,hc,clamp,x.1.isLt,x.2.isLt]

theorem logical_injective (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) : Function.Injective (logicalAddress p) := by
  intro x y equal
  have address : b.C.address x.1.val x.2.val=b.C.address y.1.val y.2.val := by
    have he := congrArg (fun n => (b.C.storage,b.C.offset+n)) equal
    simpa [logicalAddress,Tensor.address,strides.cm,strides.cn,Nat.mul_comm,Nat.add_assoc] using he
  have same := Source.output_injective p.problem b layout _ _ _ _ x.2.isLt y.2.isLt address
  exact Prod.ext (Fin.ext same.1) (Fin.ext same.2)

theorem lane_injective (p : Parameters) (b : Buffers) (layout : Layout p.problem b)
    (strides : Strides p b) (tile : Nat) (i j : CIndex)
    (hi : active p tile i) (hj : active p tile j) (equal : cOffset p tile i=cOffset p tile j) : i=j := by
  rw [address_index p tile i hi,address_index p tile j hj] at equal
  have same := logical_injective p b layout strides equal
  have hr := congrArg (fun x : LogicalIndex p => x.1.val) same
  have hc := congrArg (fun x : LogicalIndex p => x.2.val) same
  change p.problem.tileRow tile*128+i.1.val=p.problem.tileRow tile*128+j.1.val at hr
  change p.problem.tileColumn tile*256+i.2.1.val=p.problem.tileColumn tile*256+j.2.1.val at hc
  apply Prod.ext (Fin.ext (by omega))
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

theorem covers (p : Parameters) (sms : 0<p.problem.SMS) (x : LogicalIndex p) :
    ∃ pid<p.problem.grid,∃ ordinal<p.problem.work pid,Owns p (p.problem.tile pid ordinal) x := by
  obtain ⟨tile,ht,lane,ha,hr,hc⟩ := Source.output_covered p.problem sms x.1.val x.2.val x.1.isLt x.2.isLt
  obtain ⟨pid,hp,ordinal,ho,rfl⟩ := (Source.tile_mem p.problem tile).mp ht
  exact ⟨pid,hp,ordinal,ho,(lane.1,lane.2,PUnit.unit),hr,hc⟩

variable {α : Type} [Inhabited α]

theorem value_reference (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (s : State α) (entry : Memory α) (strides : Strides p b) (inputs : Inputs p b s entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero) (tile : Nat) (i : CIndex) :
    P.cast p.problem.profile (accumulation P p s tile p.problem.kTiles i.1 i.2.1)=
      Targets.result P p.problem b entry (p.problem.row tile i.1.val) (p.problem.column tile i.2.1.val) := by
  rw [accumulation_source P p b s entry strides inputs zero]
  exact Targets.source_result P p.problem b entry tile (i.1,i.2.1)

#axiomsClean logical_injective
#axiomsClean lane_injective
#axiomsClean covers
#axiomsClean value_reference
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
