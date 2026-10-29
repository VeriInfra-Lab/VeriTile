import Mathlib
import bench.repairs.tritonbench_g.dequantize_matmul.Contract
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Machine
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Direct

theorem outputContract_iff (a b scale out : TensorInfo) : OutputContract a b scale out ↔
    out.metadata.layout=.strided ∧ out.metadata.shape=[rows a,columns b] ∧ out.metadata.dtype=.floating .fp32 ∧
    out.metadata.device=a.metadata.device ∧ out.metadata.isContiguous=Bool.true ∧ out.metadata.isNeg=Bool.false ∧
    (out.storage≠a.storage ∧ out.storage≠b.storage ∧ out.storage≠scale.storage) ∧ Aligned out := by
  constructor
  · intro h;exact ⟨h.layout,h.shape,h.dtype,h.device,h.contiguous,h.noNeg,h.separated,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h⟩;exact ⟨a,b,c,d,e,f,g,h⟩

theorem checkOutput_iff (a b scale out : TensorInfo) : checkOutput a b scale out=.ok () ↔ OutputContract a b scale out := by
  rw [outputContract_iff]
  simp [checkOutput,Aligned,ite_eq_iff,and_assoc]

theorem contract_iff (a b scale : TensorInfo) (out : Option TensorInfo) : InputContract a b scale out ↔
    (a.metadata.shape.length=2 ∧ b.metadata.shape.length=2 ∧ scale.metadata.shape.length=1) ∧
    (a.metadata.layout=.strided ∧ b.metadata.layout=.strided ∧ scale.metadata.layout=.strided) ∧
    (a.metadata.dtype=.floating .fp32 ∧ b.metadata.dtype=.other "int8" ∧ scale.metadata.dtype=.floating .fp32) ∧
    (a.metadata.device.kind="cuda" ∧ a.metadata.device=b.metadata.device ∧ a.metadata.device=scale.metadata.device) ∧
    (a.metadata.isContiguous=Bool.true ∧ b.metadata.isContiguous=Bool.true ∧ scale.metadata.isContiguous=Bool.true) ∧
    (a.metadata.isNeg=Bool.false ∧ b.metadata.isNeg=Bool.false ∧ scale.metadata.isNeg=Bool.false) ∧
    (a.storage≠b.storage ∧ scale.storage≠b.storage) ∧
    (columns a=rows b ∧ scale.metadata.shape=[columns b]) ∧
    (a.metadata.numel<2^61 ∧ b.metadata.numel<2^61 ∧ rows a*columns b<2^61) ∧
    (columns a=0 ∨ columns b=0 ∨ (columns a<2^31 ∧ columns b≤65535*32)) ∧
    (Aligned a ∧ Aligned scale) ∧ (∀ provided,out=some provided → OutputContract a b scale provided) := by
  constructor
  · intro h;exact ⟨h.rank,h.layout,h.dtype,h.device,h.contiguous,h.noNeg,h.inputAlias,h.shape,h.extent,h.launch,h.alignment,h.output⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩

theorem checkTensors_iff (a b scale : TensorInfo) (out : Option TensorInfo) :
    checkTensors a b scale out=.ok () ↔ InputContract a b scale out := by
  rw [contract_iff]
  cases out <;> simp [checkTensors,checkOutput_iff,Aligned,VeriTile.Torch.TensorMetadata.isCuda,
    Except.bind,Except.pure,bind,pure,throw,ite_eq_iff,and_assoc] <;> tauto

theorem checkInputs_iff (a b scale : Argument) (out : OutputArgument) :
    checkInputs a b scale out=.ok () ↔ AdmittedArguments a b scale out := by
  cases a <;> cases b <;> cases scale <;> cases out <;> simp [checkInputs,AdmittedArguments,checkTensors_iff]

def shape (a b : TensorInfo) : Shape := ⟨rows a,columns a,columns b⟩
def canonical2 (r c : Nat) : Dense2 r c := ⟨c,1,fun _ _=>Or.inr rfl,fun _ _=>Or.inr rfl⟩
def canonical1 (n : Nat) : Dense1 n := ⟨1,Or.inr rfl⟩

theorem shape_matrix (x : TensorInfo) (rank : x.metadata.shape.length=2) : x.metadata.shape=[rows x,columns x] := by
  rcases x with ⟨⟨dtype,layout,device,shape,contig,neg,pointer⟩,storage⟩
  rcases shape with _|⟨r,tail⟩
  · simp at rank
  rcases tail with _|⟨c,tail⟩
  · simp at rank
  cases tail with
  | nil=>rfl
  | cons _ _=>simp at rank

theorem numel_matrix (x : TensorInfo) (rank : x.metadata.shape.length=2) : x.metadata.numel=rows x*columns x := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [shape_matrix x rank]
  simp

theorem configs_bounds : ∀ c∈launchConfigs,32≤c.blockK ∧ c.blockK≤256 ∧ 32≤c.blockN ∧ c.blockN≤256 := by decide

theorem padded_bound (n b : Nat) (positive : 0 < b) (bound : b≤256) : padded n b≤n+255 := by
  have h := Nat.div_mul_le_self (n+b-1) b
  change (n+b-1)/b*b ≤ n+255
  omega

theorem grid_bound (n b limit : Nat) (positive : 0 < b) (bound : n≤limit*b) : gridCount n b≤limit := by
  have h : n+b-1 < (limit+1)*b := by rw [Nat.add_mul,Nat.one_mul];omega
  have hh := (Nat.div_lt_iff_lt_mul positive).mpr h
  change (n+b-1)/b ≤ limit
  omega

/-- Guard-derived, including masked tails, for every one of the 17 configs.
Canonical strides are the literal Python launch arguments. -/
theorem native_admitted (a b scale : TensorInfo) (out : Option TensorInfo) (h : InputContract a b scale out)
    (positiveK : 0 < columns a) (positiveN : 0 < columns b)
    (config : LaunchConfig) (member : config∈launchConfigs) :
    IndexABI (shape a b) (config.tile member) (canonical2 (columns a) (columns b))
      (canonical2 (columns a) (columns b)) (canonical1 (columns b)) := by
  obtain ⟨loK,hiK,loN,hiN⟩:=configs_bounds config member
  have bounds : columns a<2^31 ∧ columns b≤65535*32 := by rcases h.launch with z|z|hh <;> omega
  have pk := padded_bound (columns a) config.blockK (by omega) hiK
  have pn := padded_bound (columns b) config.blockN (by omega) hiN
  have product : (columns a+255)*(columns b)+(columns b+255) < 2^61 := by
    have hb := Nat.mul_le_mul (show columns a+255 ≤ 2^31+255 by omega)
      (show columns b ≤ 65535*32 from bounds.2)
    norm_num at hb ⊢
    omega
  have formed : padded (columns a) config.blockK*columns b+padded (columns b) config.blockN < 2^61 := by
    have hm := Nat.mul_le_mul_right (columns b) pk
    omega
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_⟩
  · change columns a<2^63 ∧ columns b<2^63;omega
  · change columns b<2^63 ∧ 1<2^63 ∧ columns b<2^63 ∧ 1<2^63 ∧ 1<2^63;omega
  · apply grid_bound _ _ _ (by exact (launchConfigs_positive config member).1)
    change columns a ≤ (2^31-1)*config.blockK
    norm_num at bounds ⊢
    nlinarith [bounds.1]
  · apply grid_bound _ _ _ (by exact (launchConfigs_positive config member).2)
    change columns b ≤ 65535*config.blockN
    nlinarith [bounds.2]
  · change padded (columns a) config.blockK<2^63 ∧ padded (columns b) config.blockN<2^63;omega
  · change padded (columns a) config.blockK*columns b+padded (columns b) config.blockN*1<2^63
    omega
  · change 4*(padded (columns a) config.blockK*columns b+padded (columns b) config.blockN*1)<2^63
    omega
  · change 4*(padded (columns b) config.blockN*1)<2^63
    omega

#axiomsClean checkInputs_iff
#axiomsClean native_admitted
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeMatmul.Direct
