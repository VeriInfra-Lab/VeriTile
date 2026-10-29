import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostCoordinates

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.ReturnedMemory
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory RawStorage HostStorage
noncomputable section
set_option maxHeartbeats 8000000

def historyRaw (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) :=
  ScalarNative.eval I (((HostData.innerResult r before scratch).mem "h" i).read .real)
def finalRaw (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) :=
  ScalarNative.eval I (((HostData.innerResult r before scratch).mem "ht" i).read .fp32)
def prefixRaw (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) :=
  ScalarNative.eval I (((HostData.prefixResult r BS before scratch).mem "o" i).read .fp32)

def innerGrid (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (scratch : State NativeGraph.Expr) : Memory :=
  let history:=stores p.memory p.history.byteBase (historyCapacity r) (historyRaw I r before scratch)
  if r.final.isSome then stores history p.final.byteBase (finalCapacity r) (finalRaw I r before scratch) else history

/-- This copy occurs only after every source batch has completed. Values
remain snapshots even when the original final target aliases a read input. -/
def copyFinal (r : Admission.Inner) (memory : Memory) (values : Nat→Word) : Memory :=
  match r.final with
  | none=>memory
  | some tensor=>stores memory tensor.byteBase (finalCount r) values

def innerBytes (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr) : Memory :=
  stores (copyFinal r (innerGrid I r before p scratch) (HostData.finalWord I r before scratch))
    finish.place.tensor.byteBase (innerCount r) (HostData.historyWord I r before scratch)

def emptyBytes (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory) : Memory :=
  copyFinal r before (HostData.emptyFinal I r before)

def prefixGrid (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) (scratch : State NativeGraph.Expr) : Memory :=
  stores p.memory p.output.byteBase (prefixCapacity r) (prefixRaw I r BS before scratch)
def prefixBytes (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) (finish : PrefixFinish r before p) (scratch : State NativeGraph.Expr) : Memory :=
  stores (prefixGrid I r BS before p scratch) finish.place.tensor.byteBase (prefixCount r)
    (HostData.prefixWord I r BS before scratch)

theorem inner_read (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr)
    (i : Nat) (hi : i < innerCount r) :
    read (innerBytes I r before p finish scratch) (finish.place.tensor.byteBase+4*i)=
      HostData.historyWord I r before scratch i := stores_read _ _ _ _ i hi

theorem final_read (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr)
    (tensor : Admission.Tensor) (someFinal : r.final=some tensor) (i : Nat) (hi : i<finalCount r)
    (shape : tensor.shape=[r.B,r.H,r.K,r.V]) :
    read (innerBytes I r before p finish scratch) (tensor.byteBase+4*i)=HostData.finalWord I r before scratch i := by
  unfold innerBytes
  rw [stores_read_frame]
  · simp only [copyFinal,someFinal]
    exact stores_read _ _ _ _ i hi
  · apply disjoint_cell (countB:=finalCount r) _ hi
    have fresh:=finish.place.fresh tensor (by simp [Admission.Inner.tensors,someFinal])
    simpa [shape,innerCount,finalCount,Nat.mul_assoc] using fresh

theorem empty_read (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (tensor : Admission.Tensor) (someFinal : r.final=some tensor) (i : Nat) (hi : i<finalCount r) :
    read (emptyBytes I r before) (tensor.byteBase+4*i)=HostData.emptyFinal I r before i := by
  simp only [emptyBytes,copyFinal,someFinal]
  exact stores_read _ _ _ _ i hi

theorem prefix_read (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) (finish : PrefixFinish r before p) (scratch : State NativeGraph.Expr)
    (i : Nat) (hi : i<prefixCount r) :
    read (prefixBytes I r BS before p finish scratch) (finish.place.tensor.byteBase+4*i)=
      HostData.prefixWord I r BS before scratch i := stores_read _ _ _ _ i hi

theorem copy_frame (r : Admission.Inner) (memory : Memory) (values : Nat→Word) (address : Nat)
    (outside : ∀ tensor,r.final=some tensor → Outside tensor.byteBase (finalCount r) address) :
    copyFinal r memory values address=memory address := by
  cases h : r.final with
  | none=>simp [copyFinal,h]
  | some tensor=>simp only [copyFinal,h];exact stores_frame _ _ _ _ _ (outside tensor h)

theorem inner_caller_frame (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr)
    (tensor : Admission.Tensor) (member : tensor∈r.tensors) (j : Nat) (hj : j<4*tensor.shape.prod)
    (outside : ∀ output,r.final=some output → Outside output.byteBase (finalCount r) (tensor.byteBase+j)) :
    innerBytes I r before p finish scratch (tensor.byteBase+j)=before (tensor.byteBase+j) := by
  unfold innerBytes
  rw [stores_frame _ _ _ _ _ (disjoint_byte (by
    simpa [innerCount,Nat.mul_assoc] using finish.place.fresh tensor member) hj),copy_frame _ _ _ _ outside]
  unfold innerGrid
  split
  · rename_i enabled
    rw [stores_frame _ _ _ _ _ (disjoint_byte (p.finalFresh enabled tensor (by simp [member])) hj)]
    rw [stores_frame _ _ _ _ _ (disjoint_byte (p.historyFresh tensor (by simp [member])) hj)]
    exact p.callerFrame tensor member j hj
  · rw [stores_frame _ _ _ _ _ (disjoint_byte (p.historyFresh tensor (by simp [member])) hj)]
    exact p.callerFrame tensor member j hj

theorem prefix_caller_frame (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) (finish : PrefixFinish r before p) (scratch : State NativeGraph.Expr)
    (j : Nat) (hj : j<4*r.input.shape.prod) :
    prefixBytes I r BS before p finish scratch (r.input.byteBase+j)=before (r.input.byteBase+j) := by
  unfold prefixBytes prefixGrid
  rw [stores_frame _ _ _ _ _ (disjoint_byte (by
    simpa [prefixCount,Nat.mul_assoc] using finish.place.fresh r.input (by simp)) hj)]
  rw [stores_frame _ _ _ _ _ (disjoint_byte (p.fresh r.input (by simp)) hj)]
  exact p.callerFrame j hj

theorem final_source_read (I : ScalarNative.Primitives) (r : Admission.Inner) (checked : r.check=Bool.true)
    (before : Memory) (p : InnerPrepared r before) (scratch : State NativeGraph.Expr)
    (enabled : r.final.isSome=Bool.true) (i : Nat) (hi : i<finalCount r) :
    read (innerGrid I r before p scratch) (p.final.byteBase+4*HostData.finalOffset r i)=
      HostData.finalWord I r before scratch i := by
  simp only [innerGrid,enabled,ite_true]
  exact stores_read _ _ _ _ _ (HostCoordinates.final_bound r checked i hi)

theorem prefix_source_read (I : ScalarNative.Primitives) (r : Admission.Prefix) (checked : r.check=Bool.true)
    (BS : Nat) (before : Memory) (p : PrefixPrepared r before) (scratch : State NativeGraph.Expr)
    (i : Nat) (hi : i<prefixCount r) :
    read (prefixGrid I r BS before p scratch) (p.output.byteBase+4*HostData.prefixOffset r i)=
      HostData.prefixWord I r BS before scratch i :=
  stores_read _ _ _ _ _ (HostCoordinates.prefix_bound r checked i hi)

theorem history_source_read (I : ScalarNative.Primitives) (r : Admission.Inner) (checked : r.check=Bool.true)
    (before : Memory) (p : InnerPrepared r before) (scratch : State NativeGraph.Expr)
    (i : Nat) (hi : i < innerCount r) :
    read (innerGrid I r before p scratch) (p.history.byteBase+4*HostData.historyOffset r i)=
      HostData.historyWord I r before scratch i := by
  have count : p.history.shape.prod=historyCapacity r := by
    rw [p.historyValid.2.2.2.2.2.2.2.2]
    simp [historyCapacity,Admission.Inner.config,Nat.mul_assoc]
  have bound:=HostCoordinates.history_bound r checked i hi
  unfold innerGrid
  split
  · rename_i enabled
    rw [stores_read_frame]
    · exact stores_read _ _ _ _ _ bound
    · apply disjoint_cell (countB:=historyCapacity r) _ bound
      rw [←count]
      exact p.finalFresh enabled p.history (by simp)
  · exact stores_read _ _ _ _ _ bound

theorem inner_frame (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (p : InnerPrepared r before) (finish : InnerFinish r before p) (scratch : State NativeGraph.Expr) (address : Nat)
    (padding : ∀ t∈innerCopies r p.key p.value p.gate p.seed,Outside t.byteBase t.shape.prod address)
    (history : Outside p.history.byteBase (historyCapacity r) address)
    (final : r.final.isSome=Bool.true → Outside p.final.byteBase (finalCapacity r) address)
    (copy : ∀ t,r.final=some t → Outside t.byteBase (finalCount r) address)
    (output : Outside finish.place.tensor.byteBase (innerCount r) address) :
    innerBytes I r before p finish scratch address=before address := by
  unfold innerBytes
  rw [stores_frame _ _ _ _ _ output,copy_frame _ _ _ _ copy]
  unfold innerGrid
  split
  · rename_i enabled
    rw [stores_frame _ _ _ _ _ (final enabled),stores_frame _ _ _ _ _ history]
    exact p.frame address padding
  · rw [stores_frame _ _ _ _ _ history]
    exact p.frame address padding

theorem prefix_frame (I : ScalarNative.Primitives) (r : Admission.Prefix) (BS : Nat) (before : Memory)
    (p : PrefixPrepared r before) (finish : PrefixFinish r before p) (scratch : State NativeGraph.Expr) (address : Nat)
    (padding : ∀ t∈copyWrites r.input p.input,Outside t.byteBase t.shape.prod address)
    (kernel : Outside p.output.byteBase (prefixCapacity r) address)
    (output : Outside finish.place.tensor.byteBase (prefixCount r) address) :
    prefixBytes I r BS before p finish scratch address=before address := by
  unfold prefixBytes prefixGrid
  rw [stores_frame _ _ _ _ _ output,stores_frame _ _ _ _ _ kernel]
  exact p.frame address padding

#axiomsClean inner_read
#axiomsClean final_read
#axiomsClean empty_read
#axiomsClean prefix_read
#axiomsClean inner_caller_frame
#axiomsClean prefix_caller_frame
#axiomsClean final_source_read
#axiomsClean prefix_source_read
#axiomsClean history_source_read
#axiomsClean inner_frame
#axiomsClean prefix_frame
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.ReturnedMemory
