import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Geometry
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.InputContract
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.CheckedMemory

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
open VeriTile.Bench.Repairs.RmsMatmulRbe
open VeriTile.Bench.Repairs.RmsMatmulRbe.Contract hiding Inputs InputContract checkInputs
open VeriTile.Bench.Repairs.RmsMatmulRbe.Checked (Ordinary flat3 flat3_lt flat3_injective)
open VeriTile.Bench.Repairs.RmsRbeMatmul.Contract (Inputs InputContract effectiveScalars rmsTypes)
noncomputable section
set_option maxHeartbeats 5000000

def shape (i : Inputs) : Shape :=
  ⟨dim i.x 0,dim i.x 1,dim i.weight 0,dim i.x 2,i.scalars.heads.toNat,i.scalars.dim.toNat⟩
def dense3 (t : RmsMatmulRbe.Contract.Tensor) (M N : Nat) : View3 := ⟨t.base,t.offset,M*N,N,1⟩
def dense2 (t : RmsMatmulRbe.Contract.Tensor) (K : Nat) : View2 := ⟨t.base,t.offset,K,1⟩
def dense1 (t : RmsMatmulRbe.Contract.Tensor) : View1 := ⟨t.base,t.offset,1⟩
theorem dense3_address (t : RmsMatmulRbe.Contract.Tensor) (M N b m n : Nat) :
    (dense3 t M N).address b m n=(t.base,t.offset+flat3 M N b m n) := by
  simp only [dense3,View3.address,flat3]
  congr 1
  ring
structure Facts (i : Inputs) : Prop where
  valid : (shape i).Valid
  xShape : i.x.metadata.shape=[(shape i).B,(shape i).M,(shape i).K]
  weightShape : i.weight.metadata.shape=[(shape i).N,(shape i).K]
  rmsShape : i.rms.metadata.shape=[(shape i).K]
  xAdmitted : TensorAdmitted i.x [fp32]
  weightAdmitted : TensorAdmitted i.weight weightTypes
  rmsAdmitted : TensorAdmitted i.rms rmsTypes
  weightDevice : i.weight.metadata.device=i.x.metadata.device
  rmsDevice : i.rms.metadata.device=i.x.metadata.device
  rotaryBool : i.rotaryBool=Bool.true
  scalars : ScalarAdmitted (effectiveScalars i) (shape i).B (shape i).M i.rotary
  readFormat : SameReadFormat i.reads
theorem facts (i : Inputs) (h : InputContract i) : Facts i := by
  rcases h with ⟨hx,hw,hr,hdw,hdr,hbool,hxr,hwr,hp,hK,hWK,hR,hN,hmix⟩
  refine ⟨?_,RmsMatmulRbe.Checked.shape3 i.x hxr,?_,hR,hx,hw,hr,hdw,hdr,hbool,hp,hmix⟩
  · refine ⟨hK,?_,?_,?_,hN⟩ <;> dsimp [shape]
    · have := hp.2.2.1; change 0 < i.scalars.heads at this; omega
    · have := hp.2.2.2.1; change 0 < i.scalars.dim at this; omega
    · have he := hp.2.2.2.2.1
      have hn := hp.2.2.2.1
      change i.scalars.dim%2=0 at he
      change 0 < i.scalars.dim at hn
      omega
  · rw [RmsMatmulRbe.Checked.shape2 i.weight hwr,hWK]; rfl

theorem Facts.xCount (f : Facts i) : i.x.metadata.numel=(shape i).B*(shape i).M*(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.xShape,Nat.mul_assoc]
theorem Facts.wCount (f : Facts i) : i.weight.metadata.numel=(shape i).N*(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.weightShape]
theorem Facts.rCount (f : Facts i) : i.rms.metadata.numel=(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,f.rmsShape]

/-- Ordinary original Tensor.to(dtype), torch.empty and view metadata.
No value of a reduction, matmul, rotary or returned output appears here. -/
structure Allocation (i : Inputs) where
  (rms out : RmsMatmulRbe.Contract.Tensor)
  ordinary : ∀ t∈i.reads,Ordinary t
  rmsOrdinary : Ordinary rms
  outOrdinary : Ordinary out
  rmsShape : rms.metadata.shape=[(shape i).K]
  rmsDType : rms.metadata.dtype=i.weight.metadata.dtype
  rmsDevice : rms.metadata.device=i.x.metadata.device
  rmsAdmitted : TensorAdmitted rms weightTypes
  same : i.rms.metadata.dtype=i.weight.metadata.dtype → rms=i.rms
  castFresh : i.rms.metadata.dtype≠i.weight.metadata.dtype → ∀ t∈i.reads,disjoint rms t
  outShape : out.metadata.shape=[(shape i).B,(shape i).M,(shape i).N]
  outDType : out.metadata.dtype=i.weight.metadata.dtype
  outAdmitted : TensorAdmitted out weightTypes
  outDevice : out.metadata.device=i.x.metadata.device
  outOffset : out.offset=0
  outAligned : 0<out.metadata.numel → out.metadata.dataPtr.toNat%width out.metadata.dtype=0
  fresh : ∀ t∈i.reads++[rms],disjoint out t
def buffers (i : Inputs) (a : Allocation i) : Buffers :=
  let s := shape i
  ⟨dense3 i.x s.M s.K,dense1 a.rms,dense2 i.weight s.K,dense3 a.out s.M s.N,
    decide (i.weight.metadata.dtype=int8),i.rotary⟩
theorem Allocation.outCount (a : Allocation i) : a.out.metadata.numel=(shape i).B*(shape i).M*(shape i).N := by
  simp [VeriTile.Torch.TensorMetadata.numel,a.outShape,Nat.mul_assoc]
theorem Allocation.rmsCount (a : Allocation i) : a.rms.metadata.numel=(shape i).K := by
  simp [VeriTile.Torch.TensorMetadata.numel,a.rmsShape]

theorem buffers_separate (i : Inputs) (h : InputContract i) (a : Allocation i)
    (active : 0<(shape i).B ∧ 0<(shape i).M) : (buffers i a).Separate := by
  have f := facts i h
  have hn : 0<(shape i).N := by rw [f.valid.2.2.2.2]; exact Nat.mul_pos f.valid.2.1 f.valid.2.2.1
  have xp : 0 < i.x.metadata.numel := by rw [f.xCount]; exact Nat.mul_pos (Nat.mul_pos active.1 active.2) f.valid.1
  have wp : 0 < i.weight.metadata.numel := by rw [f.wCount]; exact Nat.mul_pos hn f.valid.1
  have rp : 0<a.rms.metadata.numel := by rw [a.rmsCount]; exact f.valid.1
  have op : 0<a.out.metadata.numel := by rw [a.outCount]; exact Nat.mul_pos (Nat.mul_pos active.1 active.2) hn
  have ob := a.outOrdinary.positiveBytes op
  have ne (t) (ht : t∈i.reads++[a.rms]) (ot : Ordinary t) (positive : 0<t.metadata.numel) : t.base≠a.out.base :=
    RmsMatmulRbe.Checked.disjoint_base_ne _ _ (RmsMatmulRbe.Checked.disjoint_symm _ _ (a.fresh t ht))
      (ot.positiveBytes positive) ob
  exact ⟨ne i.x (by simp [RmsRbeMatmul.Contract.Inputs.reads]) (a.ordinary i.x (by simp [RmsRbeMatmul.Contract.Inputs.reads])) xp,
    ne a.rms (by simp) a.rmsOrdinary rp,
    ne i.weight (by simp [RmsRbeMatmul.Contract.Inputs.reads]) (a.ordinary i.weight (by simp [RmsRbeMatmul.Contract.Inputs.reads])) wp⟩
theorem output_injective (i : Inputs) (a : Allocation i) :
    Function.Injective (outputAddress (buffers i a) : Cell (shape i) → Address) := by
  intro x y h
  have hj := congrArg Prod.snd h
  change ((dense3 a.out _ _).address _ _ _).2=((dense3 a.out _ _).address _ _ _).2 at hj
  rw [dense3_address,dense3_address] at hj
  exact flat3_injective (Nat.add_left_cancel hj)

def cells (s : Shape) : List (Cell s) :=
  (List.finRange s.B).product ((List.finRange s.M).product (List.finRange s.N))
def schedule (s : Shape) : Schedule s where
  events := cells s
  complete := by rintro ⟨b,m,n⟩; simp [cells]
  unique := (List.nodup_finRange s.B).product ((List.nodup_finRange s.M).product (List.nodup_finRange s.N))
def lanes (s : Shape) : List (Lane s) := (cells s).map owner
theorem lanes_cells (s : Shape) : (lanes s).map Lane.cell=(schedule s).events := by
  simp [lanes,List.map_map,Function.comp_def,owner_cell,schedule]
theorem lanes_cover (l : Lane s) : l∈lanes s := by
  rw [←lane_owner l]
  exact List.mem_map.mpr ⟨l.cell,(schedule s).complete l.cell,rfl⟩
theorem lanes_unique (s : Shape) : (lanes s).Nodup := by
  have h := (schedule s).unique
  rw [←lanes_cells s] at h
  exact List.Nodup.of_map _ h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.Checked
