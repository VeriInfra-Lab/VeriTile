import bench.repairs.tritonbench_g.matmul_dequantize_int4.Contract

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.TensorBinding
open VeriTile.Triton Checked
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Task-local Tensor ABI for the explicit half/int32 public API. Metadata is
shared Torch metadata. These laws describe actual stride/storage observations,
not kernel values, execution, numerical results, or allocation contents. -/
structure Tensor (observation : Observation) (view : View) where
  physicalIndex : Nat → Nat → Nat
  coordinates : ∀ r n,r<view.rows → n<view.cols →
    physicalIndex r n=observation.storageOffset+view.addr r n
  nonempty : 0<view.rows*view.cols → 0<observation.storageBytes
  capacity : ∀ r n,r<view.rows → n<view.cols →
    physicalIndex r n<observation.storageOffset+view.capacity

structure Tensors (p : Metadata) where
  a : Tensor p.a p.call.A
  q : Tensor p.q p.call.Q
  s : Tensor p.s p.call.S
  z : Tensor p.z p.call.Z
  output : Tensor p.output p.call.C
  storageIdentity : ∀ x∈observations p,∀ y∈observations p,x.storageObject=y.storageObject →
    x.storageBase=y.storageBase ∧ x.storageBytes=y.storageBytes

theorem allocation_names (p : Metadata) (t : Tensors p) (h : InputContract p) :
    (observations p).Pairwise (fun x y => x.storageObject≠y.storageObject) := by
  have valid := (admitted p h).valid
  have k8 : 0<p.call.K/8 := Nat.div_pos
    (Nat.le_of_dvd valid.k (dvd_trans (by decide : 8∣128) valid.k128)) (by decide)
  have n8 : 0<p.call.N/8 := Nat.div_pos
    (Nat.le_of_dvd valid.n (dvd_trans (by decide : 8∣256) valid.n256)) (by decide)
  have groups : 0<p.call.K/p.call.group := Nat.div_pos (Nat.le_of_dvd valid.k valid.groups) valid.g
  have live : ∀ x∈observations p,0<x.storageBytes := by
    intro x hx
    simp only [observations,List.mem_cons,List.not_mem_nil,or_false] at hx
    rcases hx with rfl|rfl|rfl|rfl|rfl
    · exact t.a.nonempty (by rw [valid.aShape.1,valid.aShape.2]; exact Nat.mul_pos valid.m valid.k)
    · exact t.q.nonempty (by rw [valid.qShape.1,valid.qShape.2]; exact Nat.mul_pos k8 valid.n)
    · exact t.s.nonempty (by rw [valid.sShape.1,valid.sShape.2]; exact Nat.mul_pos groups valid.n)
    · exact t.z.nonempty (by rw [valid.zShape.1,valid.zShape.2]; exact Nat.mul_pos groups n8)
    · exact t.output.nonempty (by rw [valid.cShape.1,valid.cShape.2]; exact Nat.mul_pos valid.m valid.n)
  have checkedMetadata := h.1
  simp only [metadataChecks,decide_eq_true_eq] at checkedMetadata
  have apart := checkedMetadata.2.2.2.2.2.2.2
  refine List.Pairwise.imp_of_mem (fun {x y} hx hy separation same => ?_) apart
  obtain ⟨base,bytes⟩ := t.storageIdentity x hx y hy same
  have positive := live x hx
  unfold Checked.apart at separation
  rw [←base,←bytes] at separation
  omega

abbrev Heap := Nat → Nat → Physical.Byte

def observation (p : Metadata) (r : RegionName) : Observation :=
  if r="A" then p.a else if r="Q" then p.q else if r="S" then p.s
  else if r="Z" then p.z else p.output

def width (r : RegionName) : Nat := if r="Q" ∨ r="Z" then 4 else 2
/-- Reindex the same raw allocations at the actual data_ptr origin. -/
def memory (p : Metadata) (heap : Heap) : Physical.Bytes := fun r b =>
  let t := observation p r
  heap t.storageObject (width r*t.storageOffset+b)

def origin (p : Metadata) := 2*p.output.storageOffset
/-- Interpret source stores on their actual output allocation. This is address
renaming, not a runtime gather, reference computation, or copyback. -/
def physical (p : Metadata) (heap : Heap) (m : Physical.Bytes) : Heap := fun r b =>
  if r=p.output.storageObject ∧ origin p≤b then m "C" (b-origin p) else heap r b

theorem physical_initial (p : Metadata) (heap : Heap) : physical p heap (memory p heap)=heap := by
  funext r b
  by_cases h : r=p.output.storageObject ∧ origin p≤b
  · rcases h with ⟨rfl,hb⟩
    simp only [physical,le_refl,hb,and_self,ite_true,memory,observation,width]
    simp only [show ("C":RegionName)≠"A" by decide,show ("C":RegionName)≠"Q" by decide,
      show ("C":RegionName)≠"S" by decide,show ("C":RegionName)≠"Z" by decide,
      false_or,ite_false]
    congr 1
    change origin p+(b-origin p)=b
    omega
  · simp only [physical,h,ite_false]

def store (p : Metadata) (heap : Heap) (a : Nat) (w : VeriTile.Triton.FP.Value .fp32) : Heap := fun r b =>
  if r=p.output.storageObject ∧ origin p≤b ∧ (b-origin p)/2=a then
    Physical.encode16 w ⟨(b-origin p)%2,Nat.mod_lt _ (by decide)⟩ else heap r b

theorem store_address (p : Metadata) (heap : Heap) (m : Physical.Bytes) (a : Nat)
    (w : VeriTile.Triton.FP.Value .fp32) :
    physical p heap (Physical.store16 m a w)=store p (physical p heap m) a w := by
  funext r b
  by_cases hr : r=p.output.storageObject ∧ origin p≤b
  · by_cases ha : (b-origin p)/2=a <;> simp [physical,Physical.store16,store,hr,hr.1,hr.2,ha]
  · simp only [physical,hr,ite_false,store]
    have hn : ¬(r=p.output.storageObject ∧ origin p≤b ∧ (b-origin p)/2=a) := fun h => hr ⟨h.1,h.2.1⟩
    simp only [hn,ite_false]

structure Result (p : Metadata) where
  memory : Heap
  output : Tensor p.output p.call.C
  returned : Option (Tensor p.output p.call.C)
  allocated : Bool

def invoke (p : Metadata) (t : Tensors p) (actual : Native.Instructions) (i : Fin 17) (heap : Heap) :
    Except Error (Option (Result p)) := do
  checkInputs p
  pure ((Physical.source actual p.call i (memory p heap)).map fun m =>
    ⟨physical p heap m,t.output,if p.call.supplied then none else some t.output,!p.call.supplied⟩)

def Outside (p : Metadata) (i : Fin 17) (r b : Nat) : Prop :=
  r≠p.output.storageObject ∨ b<origin p ∨
    ∀ x : Grid.Index (params p.call i),(b-origin p)/2≠Geometry.address (Grid.geometry (params p.call i)) x

def Guarantee (p : Metadata) (t : Tensors p) (mma : Opcode → ISA.MMA16) (i : Fin 17)
    (heap : Heap) (result : Result p) : Prop :=
  result.output=t.output ∧
  result.returned=(if p.call.supplied then none else some t.output) ∧
  result.allocated=(!p.call.supplied) ∧
  InputContract p ∧ Bounds.Signed32 p.call ∧
  (observations p).Pairwise (fun x y => x.storageObject≠y.storageObject) ∧
  (∀ x : Grid.Index (params p.call i),∀ byte : Fin 2,
    result.memory p.output.storageObject
      (origin p+2*Geometry.address (Grid.geometry (params p.call i)) x+byte.val)=
      Physical.encode16 (Physical.target mma p.call i (memory p heap) x) byte) ∧
  (∀ r b,Outside p i r b → result.memory r b=heap r b)

theorem whole_tensor (p : Metadata) (t : Tensors p) (h : InputContract p)
    (actual : Native.Instructions) (mma : Opcode → ISA.MMA16) (ports : Native.Obligations actual mma)
    (i : Fin 17) (heap : Heap) :
    ∃ result,invoke p t actual i heap=.ok (some result) ∧ Guarantee p t mma i heap result := by
  obtain ⟨m,he,values,frame⟩ := Physical.whole_bytes actual mma ports p.call (admitted p h).valid i (memory p heap)
  refine ⟨⟨physical p heap m,t.output,if p.call.supplied then none else some t.output,!p.call.supplied⟩,
    ?_,rfl,rfl,rfl,h,(admitted p h).indices,allocation_names p t h,?_,?_⟩
  · simp [invoke,(checkInputs_iff p).mpr h,he]
  · intro x byte
    have hb : origin p≤origin p+2*Geometry.address (Grid.geometry (params p.call i)) x+byte.val := by omega
    simp only [physical,hb,and_self,ite_true]
    rw [show origin p+2*Geometry.address (Grid.geometry (params p.call i)) x+byte.val-origin p=
      2*Geometry.address (Grid.geometry (params p.call i)) x+byte.val by omega]
    exact values x byte
  · intro r b outside
    by_cases hwrite : r=p.output.storageObject ∧ origin p≤b
    · have ho : ∀ x : Grid.Index (params p.call i),(b-origin p)/2≠Geometry.address (Grid.geometry (params p.call i)) x := by
        rcases outside with hr|hb|ho
        · exact False.elim (hr hwrite.1)
        · omega
        · exact ho
      have hf := frame "C" (b-origin p) (Or.inr ho)
      have hi := congrFun (congrFun (physical_initial p heap) r) b
      simp only [physical,hwrite,ite_true] at hi ⊢
      exact hf.trans hi
    · simp only [physical,hwrite,ite_false]

#specNonCircular Guarantee avoiding [invoke,Physical.source,Native.grid,Grid.grid]
#axiomsClean allocation_names
#axiomsClean physical_initial
#axiomsClean store_address
#axiomsClean whole_tensor
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.TensorBinding
