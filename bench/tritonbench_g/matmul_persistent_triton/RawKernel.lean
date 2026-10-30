import bench.tritonbench_g.matmul_persistent_triton.Execution

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section

/-- Kernel allocation identity, offset, capacity and native pointer only.
Shapes, storage precision and strides come from the source parameters. -/
structure Allocation where
  storage : Nat
  offset : Nat
  capacityBytes : Nat
  base : BitVec 64
  pointer : BitVec 64

structure Buffers where
  a : Allocation
  b : Allocation
  c : Allocation

/-- Numeric source-address representability and backing bounds. -/
structure Bounds (a : Allocation) (rows columns rowStride columnStride : Nat) (dtype : Storage) : Prop where
  rowsBound : rows<2^63
  columnsBound : columns<2^63
  rowStrideBound : rowStride<2^63
  columnStrideBound : columnStride<2^63
  offsetBound : a.offset<2^63
  span : if rows=0 ∨ columns=0 then True else
    dtype.bytes*(a.offset+(rows-1)*rowStride+(columns-1)*columnStride+1)≤a.capacityBytes
  capacity : a.capacityBytes<2^63
  physical : a.base.toNat+a.capacityBytes≤2^64
  pointer : if rows=0 ∨ columns=0 then True else a.pointer.toNat=a.base.toNat+dtype.bytes*a.offset

structure Layout (p : Core.Parameters) (b : Buffers) : Prop where
  aBounds : Bounds b.a p.problem.M p.problem.K p.am p.ak p.problem.profile
  bBounds : Bounds b.b p.problem.K p.problem.N p.bk p.bn p.problem.profile
  cBounds : Bounds b.c p.problem.M p.problem.N p.cm p.cn p.problem.profile
  outputStrides : p.cm=max p.problem.N 1 ∧ p.cn=1
  aSeparate : b.c.storage≠b.a.storage
  bSeparate : b.c.storage≠b.b.storage

/-- Internal adapter for the established execution proof. API-only fields
are fixed constants, and do not occur in the raw interface. -/
def Allocation.descriptor (a : Allocation) (rows columns rs cs : Nat) (dtype : Storage) : Tensor :=
  ⟨0,a.storage,a.offset,rows,columns,rs,cs,dtype,0,Bool.false,Bool.false,a.capacityBytes,a.base,a.pointer⟩

def Buffers.descriptors (b : Buffers) (p : Core.Parameters) :
    VeriTile.Bench.Repairs.MatmulPersistentTriton.Buffers :=
  ⟨b.a.descriptor p.problem.M p.problem.K p.am p.ak p.problem.profile,
   b.b.descriptor p.problem.K p.problem.N p.bk p.bn p.problem.profile,
   b.c.descriptor p.problem.M p.problem.N p.cm p.cn p.problem.profile⟩

theorem Bounds.descriptor (a : Allocation) (rows columns rs cs : Nat) (dtype : Storage)
    (h : Bounds a rows columns rs cs dtype) : (a.descriptor rows columns rs cs dtype).Bounded := by
  refine ⟨h.rowsBound,h.columnsBound,h.rowStrideBound,h.columnStrideBound,h.offsetBound,?_,
    h.capacity,h.physical,h.pointer⟩
  by_cases empty : rows=0 ∨ columns=0
  · simpa [Tensor.span,Allocation.descriptor,empty] using h.span
  · simpa [Tensor.span,Allocation.descriptor,empty] using h.span

theorem Layout.descriptors (p : Core.Parameters) (b : Buffers) (h : Layout p b) :
    VeriTile.Bench.Repairs.MatmulPersistentTriton.Layout p.problem (b.descriptors p) :=
  ⟨⟨rfl,rfl⟩,⟨rfl,rfl⟩,⟨rfl,rfl⟩,rfl,rfl,rfl,
    h.aBounds.descriptor _ _ _ _ _ _,h.bBounds.descriptor _ _ _ _ _ _,h.cBounds.descriptor _ _ _ _ _ _,
    h.outputStrides,h.aSeparate,h.bSeparate⟩

theorem strides (p : Core.Parameters) (b : Buffers) : Core.Strides p (b.descriptors p) :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

structure Inputs [Inhabited α] (p : Core.Parameters) (b : Buffers) (s : State α) (entry : Memory α) : Prop where
  a : ∀ i,Core.readCell p.problem.profile (s.mem "A" i)=entry (b.a.storage,b.a.offset+i)
  b : ∀ i,Core.readCell p.problem.profile (s.mem "B" i)=entry (b.b.storage,b.b.offset+i)

theorem Inputs.descriptors [Inhabited α] (p : Core.Parameters) (b : Buffers) (s : State α)
    (entry : Memory α) (h : Inputs p b s entry) : Core.Inputs p (b.descriptors p) s entry := ⟨h.a,h.b⟩

def realInputs (p : Core.Parameters) (b : Buffers) (entry : Memory ℝ) : RealMath.Inputs :=
  ⟨fun row k => entry (b.a.storage,b.a.offset+row*p.am+k*p.ak),
   fun k column => entry (b.b.storage,b.b.offset+k*p.bk+column*p.bn)⟩

def reference (P : Source.Arithmetic α α) (p : Core.Parameters) (b : Buffers)
    (entry : Memory α) (row column : Nat) : α :=
  Targets.result P p.problem (b.descriptors p) entry row column

theorem real_reference (p : Core.Parameters) (b : Buffers) (entry : Memory ℝ)
    (sms : 0<p.problem.SMS) (row column : Nat) (hr : row<p.problem.M) (hc : column<p.problem.N) :
    reference Real.arithmetic p b entry row column=RealMath.formula p.problem.K (realInputs p b entry) row column := by
  obtain ⟨pid,hp,ordinal,ho,i,ri,ci⟩ := Core.covers p sms (⟨row,hr⟩,⟨column,hc⟩)
  have active : Source.active p.problem (p.problem.tile pid ordinal) (i.1,i.2.1) := ⟨ri ▸ hr,ci ▸ hc⟩
  simpa only [ri,ci] using Targets.real p.problem (b.descriptors p) entry (p.problem.tile pid ordinal) (i.1,i.2.1) active

#axiomsClean Layout.descriptors
#axiomsClean real_reference
#specNonCircular reference avoiding [Core.executeGrid,Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.RawKernel
