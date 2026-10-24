import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RawLifting
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeInt4
open Checked TensorBinding
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-! Complete raw-kernel Tensor lifting using only Call.Valid. The sibling host
metadata/index contract is deliberately not a premise. -/

theorem output_coordinates (p : Metadata) (output : TensorBinding.Tensor p.output p.call.C)
    (valid : p.call.Valid) (i : Fin 17) (x : Grid.Index (params p.call i)) :
    output.physicalIndex x.1.val x.2.val=p.output.storageOffset+
      Geometry.address (Grid.geometry (params p.call i)) x := by
  exact output.coordinates _ _ (by rw [valid.cShape.1]; exact x.1.isLt)
    (by rw [valid.cShape.2]; exact x.2.isLt)

theorem packed_entry (p : Metadata) (raw : Heap) (heap : RealTensor.RealHeap) :
    KernelReal.Packed (RealTensor.entry p raw heap) := by
  intro region hr offset
  exact ⟨Physical.decode32 (fun byte => memory p raw region (4*offset+byte.val)),
    by rcases hr with rfl|rfl <;> simp only [RealTensor.entry,eq_self,or_true,true_or,ite_true]⟩

def RealOutside (p : Metadata) (i : Fin 17) (r a : Nat) : Prop :=
  r≠p.output.storageObject ∨ a<p.output.storageOffset ∨
    ∀ x : Grid.Index (params p.call i),
      a-p.output.storageOffset≠Geometry.address (Grid.geometry (params p.call i)) x

/-- Independent matrix product on the actual Tensor coordinates, plus every
caller allocation and every hole in the output allocation. -/
def RealPost (p : Metadata) (output : TensorBinding.Tensor p.output p.call.C) (i : Fin 17) (raw : Heap)
    (initial final : RealTensor.RealHeap) : Prop :=
  (∀ x : Grid.Index (params p.call i),
    final p.output.storageObject (output.physicalIndex x.1.val x.2.val)=
      Real.mathematical p.call (RealHost.input p.call (RealTensor.entry p raw initial))
        x.1.val x.2.val) ∧
  (∀ r a,RealOutside p i r a → final r a=initial r a)

theorem real (p : Metadata) (output : TensorBinding.Tensor p.output p.call.C) (valid : p.call.Valid)
    (i : Fin 17) (raw : Heap) (heap : RealTensor.RealHeap) :
    ∃ state,KernelSource.run KernelReal.scalar KernelReal.dot p.call i
        (RealTensor.entry p raw heap)=some state ∧
      RealPost p output i raw heap (RealTensor.physical p heap state) := by
  obtain ⟨state,executed,values,frame⟩ := KernelReal.whole_grid p.call valid i
    (RealTensor.entry p raw heap) (packed_entry p raw heap)
  refine ⟨state,executed,?_,?_⟩
  · intro x
    rw [output_coordinates p output valid i x]
    have hv := congrArg (fun c : Cell ℝ => c.read .fp16) (values x trivial)
    simpa [RealTensor.physical,Cell.read_same] using hv
  · intro r a outside
    by_cases hw : r=p.output.storageObject ∧ p.output.storageOffset≤a
    · have ho : ∀ x : Grid.Index (params p.call i),
          a-p.output.storageOffset≠Geometry.address (Grid.geometry (params p.call i)) x := by
        rcases outside with hr|ha|ho
        · exact False.elim (hr hw.1)
        · omega
        · exact ho
      have hf := congrArg (fun c : Cell ℝ => c.read .fp16)
        (frame "C" (a-p.output.storageOffset) (Or.inr (fun x _ => ho x)))
      simp only [RealTensor.physical,hw,ite_true]
      change (state.mem "C" (a-p.output.storageOffset)).read .fp16=_
      dsimp only at hf
      rw [hf]
      simp only [RealTensor.entry,observation,show ("C":RegionName)≠"A" by decide,
        show ("C":RegionName)≠"Q" by decide,show ("C":RegionName)≠"S" by decide,
        show ("C":RegionName)≠"Z" by decide,false_or,ite_false,Cell.read_same]
      congr 1
      omega
    · simp only [RealTensor.physical,hw,ite_false]

def halfRead (heap : Heap) (storage index : Nat) :=
  Physical.decode16 (fun byte => heap storage (2*index+byte.val))

/-- Ordered fixed-K16 native target and exact little-endian half bytes at the
actual Tensor output coordinates. The target never invokes the source. -/
def NativePost (p : Metadata) (output : TensorBinding.Tensor p.output p.call.C) (mma : Opcode → ISA.MMA16)
    (i : Fin 17) (initial final : Heap) : Prop :=
  (∀ x : Grid.Index (params p.call i),∀ byte : Fin 2,
    final p.output.storageObject (2*output.physicalIndex x.1.val x.2.val+byte.val)=
      Physical.encode16 (Physical.target mma p.call i (memory p initial) x) byte) ∧
  (∀ x : Grid.Index (params p.call i),
    halfRead final p.output.storageObject (output.physicalIndex x.1.val x.2.val)=
      Physical.decode16 (Physical.encode16 (Physical.target mma p.call i (memory p initial) x))) ∧
  (∀ r b,TensorBinding.Outside p i r b → final r b=initial r b)

theorem native (p : Metadata) (output : TensorBinding.Tensor p.output p.call.C) (valid : p.call.Valid)
    (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (i : Fin 17) (heap : Heap) :
    ∃ state,Native.grid actual p.call i (Physical.load (memory p heap))=some state ∧
      NativePost p output mma i heap
        (physical p heap (Physical.commit (memory p heap) state)) := by
  obtain ⟨m,executed,values,frame⟩ := Physical.whole_bytes actual mma ports p.call
    valid i (memory p heap)
  unfold Physical.source at executed
  cases he : Native.grid actual p.call i (Physical.load (memory p heap)) with
  | none => simp [he] at executed
  | some state =>
    simp only [he,Option.map_some,Option.some.injEq] at executed
    subst m
    have bytes : ∀ x : Grid.Index (params p.call i),∀ byte : Fin 2,
        physical p heap (Physical.commit (memory p heap) state) p.output.storageObject
          (2*output.physicalIndex x.1.val x.2.val+byte.val)=
        Physical.encode16 (Physical.target mma p.call i (memory p heap) x) byte := by
      intro x byte
      rw [output_coordinates p output valid i x]
      have hb : origin p≤2*(p.output.storageOffset+
          Geometry.address (Grid.geometry (params p.call i)) x)+byte.val := by
        unfold origin
        omega
      simp only [physical,hb,and_self,ite_true]
      rw [show 2*(p.output.storageOffset+Geometry.address (Grid.geometry (params p.call i)) x)+
        byte.val-origin p=2*Geometry.address (Grid.geometry (params p.call i)) x+byte.val by
          unfold origin; omega]
      exact values x byte
    refine ⟨state,rfl,bytes,?_,?_⟩
    · intro x
      unfold halfRead
      congr 1
      funext byte
      exact bytes x byte
    · intro r b outside
      by_cases hw : r=p.output.storageObject ∧ origin p≤b
      · have ho : ∀ x : Grid.Index (params p.call i),
            (b-origin p)/2≠Geometry.address (Grid.geometry (params p.call i)) x := by
          rcases outside with hr|hb|ho
          · exact False.elim (hr hw.1)
          · omega
          · exact ho
        have hf := frame "C" (b-origin p) (Or.inr ho)
        have hi := congrFun (congrFun (physical_initial p heap) r) b
        simp only [physical,hw,ite_true] at hi ⊢
        exact hf.trans hi
      · simp only [physical,hw,ite_false]


#specNonCircular RealPost avoiding [KernelSource.run,KernelSource.one,RealHost.grid,Real.value]
#specNonCircular NativePost avoiding [Native.grid,Native.exec,Physical.source,Grid.grid]
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RawLifting
