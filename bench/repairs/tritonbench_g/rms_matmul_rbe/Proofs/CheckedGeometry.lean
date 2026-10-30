import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Geometry
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.InputContract
import Mathlib.Data.List.Nodup
import Mathlib.Data.List.ProdSigma

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
noncomputable section
set_option maxHeartbeats 5000000
open Contract

def shape (i : Inputs) : Shape :=
  ⟨dim i.x 0,dim i.x 1,dim i.qw 0,dim i.x 2,i.scalars.heads.toNat,i.scalars.dim.toNat⟩
def dense3 (t : Contract.Tensor) (M N : Nat) : View3 := ⟨t.base,t.offset,M*N,N,1⟩
def dense2 (t : Contract.Tensor) (K : Nat) : View2 := ⟨t.base,t.offset,K,1⟩
def dense1 (t : Contract.Tensor) : View1 := ⟨t.base,t.offset,1⟩
def weight (i : Inputs) (phase : Fin 3) : Contract.Tensor :=
  if phase.val=0 then i.qw else if phase.val=1 then i.kw else i.vw
def output (i : Inputs) (q : Contract.Tensor) (phase : Fin 3) : Contract.Tensor :=
  if phase.val=0 then q else if phase.val=1 then i.k else i.v
def buffers (i : Inputs) (q : Contract.Tensor) : Buffers :=
  let s := shape i
  ⟨dense3 i.x s.M s.K,dense1 i.rms,fun phase => dense2 (weight i phase) s.K,
    fun phase => dense3 (output i q phase) s.M s.N,decide (i.qw.metadata.dtype=int8)⟩

theorem shape3 (t : Contract.Tensor) (h : t.metadata.shape.length=3) :
    t.metadata.shape=[dim t 0,dim t 1,dim t 2] := by
  rcases hshape : t.metadata.shape with _|⟨a,_|⟨b,_|⟨c,tail⟩⟩⟩ <;> simp_all [dim]
theorem shape2 (t : Contract.Tensor) (h : t.metadata.shape.length=2) :
    t.metadata.shape=[dim t 0,dim t 1] := by
  rcases hshape : t.metadata.shape with _|⟨a,_|⟨b,tail⟩⟩ <;> simp_all [dim]

structure Facts (i : Inputs) : Prop where
  valid : (shape i).Valid
  xShape : i.x.metadata.shape=[(shape i).B,(shape i).M,(shape i).K]
  weightShape : ∀ phase,(weight i phase).metadata.shape=[(shape i).N,(shape i).K]
  rmsShape : i.rms.metadata.shape=[(shape i).K]
  kCount : i.k.metadata.numel=(shape i).B*(shape i).M*(shape i).N
  vCount : i.v.metadata.numel=(shape i).B*(shape i).M*(shape i).N
  xAdmitted : TensorAdmitted i.x [fp32]
  others : ∀ t∈i.others,TensorAdmitted t weightTypes
  device : ∀ t∈i.others,t.metadata.device=i.x.metadata.device
  format : ∀ t∈i.sameFormat,t.metadata.dtype=i.qw.metadata.dtype
  scalars : ScalarAdmitted i.scalars (shape i).B (shape i).M Bool.true
  readFormat : SameReadFormat i.reads
  kvDisjoint : disjoint i.k i.v
  outputsDisjoint : ∀ t∈i.reads,disjoint i.k t ∧ disjoint i.v t

theorem facts (i : Inputs) (h : InputContract i) : Facts i := by
  rcases h with ⟨hx,ho,hd,hf,hxr,hwr,hkw,hvw,hp,hK,hWK,hR,hN,hkc,hvc,hmix,hkv,houts⟩
  have hws : i.qw.metadata.shape=[dim i.qw 0,dim i.x 2] := by rw [shape2 i.qw hwr,hWK]
  refine ⟨?_,shape3 i.x hxr,?_,hR,hkc,hvc,hx,ho,hd,hf,hp,hmix,hkv,houts⟩
  · refine ⟨hK,?_,?_,?_,hN⟩ <;> dsimp [shape]
    · have := hp.2.2.1; omega
    · have := hp.2.2.2.1; omega
    · have he := hp.2.2.2.2.1
      have hn := hp.2.2.2.1
      omega
  · intro phase
    fin_cases phase <;> simp [weight,hkw,hvw,hws,shape]

def flat3 (M N b m n : Nat) := (b*M+m)*N+n
theorem flat3_lt (B M N b m n : Nat) (hb : b<B) (hm : m<M) (hn : n<N) :
    flat3 M N b m n<B*M*N := by
  have hr : b*M+m<B*M := by nlinarith
  unfold flat3
  nlinarith
theorem mul_add_div_right (a b c : Nat) (hc : c<b) : (a*b+c)/b=a := by
  rw [Nat.mul_comm a b,Nat.mul_add_div (by omega),Nat.div_eq_of_lt hc,Nat.add_zero]
theorem flat3_injective {B M N : Nat} :
    Function.Injective (fun x : Fin B × Fin M × Fin N => flat3 M N x.1 x.2.1 x.2.2) := by
  intro x y h
  have hN : 0<N := by have := x.2.2.isLt; omega
  have hM : 0<M := by have := x.2.1.isLt; omega
  have hn := congrArg (fun v => v%N) h
  simp only [flat3,Nat.mul_add_mod',
    Nat.mod_eq_of_lt x.2.2.isLt,Nat.mod_eq_of_lt y.2.2.isLt] at hn
  have hp := congrArg (fun v => v/N) h
  simp only [flat3,mul_add_div_right _ _ _ x.2.2.isLt,mul_add_div_right _ _ _ y.2.2.isLt] at hp
  have hm := congrArg (fun v => v%M) hp
  simp only [Nat.mul_add_mod',
    Nat.mod_eq_of_lt x.2.1.isLt,Nat.mod_eq_of_lt y.2.1.isLt] at hm
  have hb := congrArg (fun v => v/M) hp
  simp only [mul_add_div_right _ _ _ x.2.1.isLt,mul_add_div_right _ _ _ y.2.1.isLt] at hb
  exact Prod.ext (Fin.ext hb) (Prod.ext (Fin.ext hm) (Fin.ext hn))

theorem dense3_address (t : Contract.Tensor) (M N b m n : Nat) :
    (dense3 t M N).address b m n=(t.base,t.offset+flat3 M N b m n) := by
  simp only [dense3,View3.address,flat3]
  congr 1
  ring

/-- The checked source grid has exactly these active cells. owner_cell,
lane_owner and lane_grid derive the mask/filter coverage and uniqueness. -/
def cells (s : Shape) : List (Cell s) :=
  (List.finRange 3).product ((List.finRange s.B).product ((List.finRange s.M).product (List.finRange s.N)))
def schedule (s : Shape) : Schedule s where
  events := cells s
  complete := by rintro ⟨p,b,m,n⟩; simp [cells]
  unique := (List.nodup_finRange 3).product ((List.nodup_finRange s.B).product
    ((List.nodup_finRange s.M).product (List.nodup_finRange s.N)))
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
end VeriTile.Bench.Repairs.RmsMatmulRbe.Checked
