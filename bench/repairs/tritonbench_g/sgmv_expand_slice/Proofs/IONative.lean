import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PublicIO
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural NativeBackend
noncomputable section
set_option maxHeartbeats 4000000

/-- FP16 channel words use the existing 16-bit encoding in the shared 32-bit
interpreter payload. This is representation typing, not a numerical law or a
finite-operand restriction. Every FP16 and FP32 bit pattern is admitted. -/
def nativeDomain (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues Raw (inputChannels p shape segments)) : Prop :=
  metadataDomain p shape segments xs ∧
  ∀ a : Fin shape.adapters, ∀ c : Fin shape.columns, ∀ k : Fin shape.inner,
    pack .half (unpack .half (xs ⟨1,by simp⟩ (a,c,k,PUnit.unit))) =
      xs ⟨1,by simp⟩ (a,c,k,PUnit.unit)

def nativeInputs (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues Raw (inputChannels p shape segments)) : Native.Inputs (profile abi) where
  x := fun r k=>unpack (scalar abi.input) (inputValues xs r k)
  weight := fun a c k=>unpack (weight abi.weight).format (weightValues xs a c k)

def nativeFormula (ops : Native.Primitives) (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (segments : List Segment) (xs : ChannelValues Raw (inputChannels p shape segments)) :
    Nat → Nat → Raw :=
  fun r c=>pack (scalar abi.output) (Native.result ops (profile abi) shape segments
    (nativeInputs abi p shape segments xs)
    (fun row col=>unpack (scalar abi.output) (entryValues xs row col)) r c)

theorem native_implements (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (hi : abi.input=.single) (hw : abi.weight=.half) (ho : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    (io (SourceBody.install (algebra rest ops (profile abi)) (dot ops (weight abi.weight)))
      abi p shape segments).ImplementsOn (nativeDomain p shape segments)
      (fun xs=>resultValues p shape (nativeFormula ops abi p shape segments xs)) := by
  apply implements (algebra rest ops (profile abi)) (dot ops (weight abi.weight))
    abi hw ho p shape segments separation fits inner columns slice capacity layout _ (fun _ h=>h.1)
  intro xs hd initial binding
  have memory := binding_inputs _ abi hi hw p shape segments xs initial binding inner columns
  have memory' : SourceMemory.Inputs abi p shape initial
      (inputWords abi (nativeInputs abi p shape segments xs))
      (weightWords abi (nativeInputs abi p shape segments xs)) := by
    refine ⟨memory.inner,memory.columns,?_,?_⟩
    · intro r k hr hk
      rw [memory.x r k hr hk]
      simp [inputWords,nativeInputs,hi,scalar,pack,unpack,Native.Format.bits]
    · intro a c k ha hc hk
      rw [memory.weight a c k ha hc hk]
      change weightValues xs a c k = pack (weight abi.weight).format
        (unpack (weight abi.weight).format (weightValues xs a c k))
      rw [hw]
      have canonical := hd.2 ⟨a,ha⟩ ⟨c,hc⟩ ⟨k,hk⟩
      simpa [weightWords,nativeInputs,weightValues,matrixAt,ha,hc,hk,hw,weight,
        Native.WeightFormat.format] using canonical.symm
  have output := binding_output _ abi ho p shape segments xs initial binding
  have output' : SourceSuffix.OutputMemory abi p shape initial
      (fun r c=>pack (scalar abi.output) (unpack (scalar abi.output) (entryValues xs r c))) := by
    intro r c hr hc
    rw [output r c hr hc]
    simp [ho,scalar,pack,unpack,Native.Format.bits]
  obtain ⟨final,executed,result,frame⟩ := NativeBackend.source_result rest ops zeros abi p shape segments
    (nativeInputs abi p shape segments xs) (fun r c=>unpack (scalar abi.output) (entryValues xs r c))
    initial memory' output' (binding_metadata _ abi p shape segments xs hd.1 initial binding)
    separation fits slice capacity layout
  exact ⟨final,executed,result⟩

#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
