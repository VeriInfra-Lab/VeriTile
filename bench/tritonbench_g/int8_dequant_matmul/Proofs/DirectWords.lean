import bench.tritonbench_g.int8_dequant_matmul.Proofs.DirectPointers

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
noncomputable section
set_option maxHeartbeats 1500000

/-- Actual data_ptr arithmetic names the same original physical input cells
used by the logical word target, including nonzero storage offsets. -/
structure OperandPointers (base : Nat → Nat) (r : Request) : Prop where
  a : ∀ i,i<r.a.info.metadata.numel → nativePointer r.a.info.metadata.dataPtr 1 i=base r.a.info.storage+r.a.physicalIndex i
  b : ∀ i,i<r.b.info.metadata.numel → nativePointer r.b.info.metadata.dataPtr 1 i=base r.b.info.storage+r.b.physicalIndex i
  x : ∀ i,i<r.x.info.metadata.numel → nativePointer r.x.info.metadata.dataPtr 4 i=base r.x.info.storage+4*r.x.physicalIndex i
  w : ∀ i,i<r.w.info.metadata.numel → nativePointer r.w.info.metadata.dataPtr 4 i=base r.w.info.storage+4*r.w.physicalIndex i
  bias : ∀ t,r.bias=some t → ∀ i,i<t.info.metadata.numel →
    nativePointer t.info.metadata.dataPtr 4 i=base t.info.storage+4*t.physicalIndex i

theorem operand_pointers (base : Nat → Nat) (r : Request) (laws : RequestLaws base r) : OperandPointers base r := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro i hi;simpa using original_pointer 1 base r.a laws.a (by decide) r.contract.contiguous.1 i hi
  · intro i hi;simpa using original_pointer 1 base r.b laws.b (by decide) r.contract.contiguous.2.1 i hi
  · exact original_pointer 4 base r.x laws.x (by decide) r.contract.contiguous.2.2.1
  · exact original_pointer 4 base r.w laws.w (by decide) r.contract.contiguous.2.2.2
  · intro t eq i hi
    exact original_pointer 4 base t (laws.bias t eq) (by decide) (r.contract.biasContract t.info (by simp [eq])).direct.1 i hi

/-- Logical input words from the original TensorViews, before allocation or
execution. This target has no source buffer or returned-memory parameter. -/
structure LogicalWords (F : Type) where
  A : Nat → Nat → BitVec 8
  B : Nat → Nat → BitVec 8
  x : Nat → F
  w : Nat → F
  bias : Option (Nat → F)

def originalWords (a : Ops I F H) (r : Request) (m : Memory F H) : LogicalWords F where
  A := fun row k=>read8 m (logicalAddress r.a (row*(requestShape r).K+k))
  B := fun k n=>read8 m (logicalAddress r.b (k*(requestShape r).N+n))
  x := fun row=>read32 a m (logicalAddress r.x row)
  w := fun n=>read32 a m (logicalAddress r.w n)
  bias := r.bias.map (fun t n=>read32 a m (logicalAddress t n))

/-- Independent ordered instruction reference on logical operands. The exact
selected fixed-K32/scalar plan, old accumulator and fusion are retained. -/
def logicalAccumulation (a : Ops I F H) (s : Shape) (c : Config)
    (selected : Selection s c) (input : LogicalWords F) (row n z : Nat) : Nat → I
  | 0=>a.izero
  | t+1=>(selected.plan row n z t).eval a
      (fun l=>if sourceK c t z l.val<s.K then input.A row (sourceK c t z l.val) else 0)
      (fun l=>if sourceK c t z l.val<s.K then input.B (sourceK c t z l.val) n else 0)
      (logicalAccumulation a s c selected input row n z t)

def logicalPartial (a : Ops I F H) (s : Shape) (c : Config)
    (selected : Selection s c) (divisor : F) (input : LogicalWords F) (row n z : Nat) : H :=
  let acc:=logicalAccumulation a s c selected input row n z (steps s c)
  let value:=a.narrow (a.mul (input.w n) (a.mul (input.x row) (a.mul (a.toFloat acc) divisor)))
  match input.bias with
  | none=>value
  | some bias=>if z=0 then a.hadd value (a.narrow (bias n)) else value

def logicalWordReference (a : Ops I F H) (s : Shape) (c : Config)
    (selected : Selection s c) (divisor : F) (q : Schedule s c)
    (input : LogicalWords F) (cell : MatrixCell s) : H :=
  cellFold a c (fun z=>logicalPartial a s c selected divisor input cell.1 cell.2 z)
    (projection cell q.events) a.hzero

theorem accumulation_original (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection (requestShape r) c)
    (m : Memory F H) (out : Nat) (cell : MatrixCell (requestShape r)) (z t : Nat) :
    accumulation a (requestShape r) c (requestBuffers r out) selected m cell.1 cell.2 z t=
      logicalAccumulation a (requestShape r) c selected (originalWords a r m) cell.1 cell.2 z t := by
  have ha : ∀ t,operandA (requestShape r) c (requestBuffers r out) m cell.1 t z=
      (fun l=>if sourceK c t z l.val<(requestShape r).K then
        (originalWords a r m).A cell.1 (sourceK c t z l.val) else 0) := by
    intro t;funext l
    unfold operandA
    split_ifs with hk
    · have h:=(original_operands base r laws m out cell.1 cell.2 _ cell.1.isLt cell.2.isLt hk).1
      simp only [originalWords,read8,h]
    · rfl
  have hb : ∀ t,operandB (requestShape r) c (requestBuffers r out) m cell.2 t z=
      (fun l=>if sourceK c t z l.val<(requestShape r).K then
        (originalWords a r m).B (sourceK c t z l.val) cell.2 else 0) := by
    intro t;funext l
    unfold operandB
    split_ifs with hk
    · have h:=(original_operands base r laws m out cell.1 cell.2 _ cell.1.isLt cell.2.isLt hk).2.1
      simp only [originalWords,read8,h]
    · rfl
  induction t with
  | zero=>rfl
  | succ t ih=>simp only [accumulation,logicalAccumulation,ha,hb,ih]

theorem partial_original (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection (requestShape r) c) (divisor : F)
    (m : Memory F H) (out : Nat) (cell : MatrixCell (requestShape r)) (z : Nat) :
    partialValue a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) m cell.1 cell.2 z=
      logicalPartial a (requestShape r) c selected divisor (originalWords a r m) cell.1 cell.2 z := by
  have hx : cell.1.val<r.x.info.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.1,List.prod_cons,List.prod_nil,Nat.mul_one] using cell.1.isLt
  have hw : cell.2.val<r.w.info.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,r.contract.shape.2.2,List.prod_cons,List.prod_nil,Nat.mul_one] using cell.2.isLt
  have sx:=original_read 4 base r.x laws.x r.contract.contiguous.2.2.1 m cell.1.val hx
  have sw:=original_read 4 base r.w laws.w r.contract.contiguous.2.2.2 m cell.2.val hw
  unfold partialValue logicalPartial
  rw [accumulation_original base r laws]
  cases eq : r.bias with
  | none=>simp [originalWords,hasBias,eq,requestBuffers,buffers,read32,sx,sw]
  | some t=>
    have tc:=r.contract.biasContract t.info (by simp [eq])
    have nt : cell.2.val<t.info.metadata.numel := by
      simpa only [VeriTile.Torch.TensorMetadata.numel,tc.descriptor.2.2.1,List.prod_cons,List.prod_nil,Nat.mul_one] using cell.2.isLt
    have sb:=original_read 4 base t (laws.bias t eq) tc.direct.1 m cell.2.val nt
    simp [originalWords,hasBias,eq,requestBuffers,buffers,biasOffset,read32,sx,sw,sb]

theorem word_original (base : Nat → Nat) (r : Request) (laws : RequestLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection (requestShape r) c) (divisor : F)
    (q : Schedule (requestShape r) c) (m : Memory F H) (out : Nat) (cell : MatrixCell (requestShape r)) :
    wordReference a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) q m cell=
      logicalWordReference a (requestShape r) c selected divisor q (originalWords a r m) cell := by
  unfold wordReference logicalWordReference
  have h : (fun z : Fin c.S=>partialValue a (requestShape r) c (requestBuffers r out) selected divisor (hasBias r) m cell.1 cell.2 z)=
      (fun z : Fin c.S=>logicalPartial a (requestShape r) c selected divisor (originalWords a r m) cell.1 cell.2 z) := by
    funext z;exact partial_original base r laws a c selected divisor m out cell z
  rw [h]

#axiomsClean accumulation_original
#axiomsClean operand_pointers
#axiomsClean partial_original
#axiomsClean word_original
#specNonCircular logicalWordReference avoiding [run,wrapper,invoke,partialValue,wordReference]
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
