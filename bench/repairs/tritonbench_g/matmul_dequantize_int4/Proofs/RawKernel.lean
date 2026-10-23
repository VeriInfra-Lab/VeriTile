import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelRealGrid
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantReal

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.RawKernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only kernel dimensions and raw strided memory views. -/
structure Layout where
  M : Nat
  N : Nat
  K : Nat
  group : Nat
  A : View
  Q : View
  S : View
  Z : View
  C : View

structure Layout.Valid (c : Layout) : Prop where
  m : 0<c.M
  n : 0<c.N
  k : 0<c.K
  g : 0<c.group
  n256 : 256∣c.N
  k128 : 128∣c.K
  g128 : 128∣c.group
  groups : c.group∣c.K
  aShape : c.A.rows=c.M ∧ c.A.cols=c.K
  qShape : c.Q.rows=c.K/8 ∧ c.Q.cols=c.N
  sShape : c.S.rows=c.K/c.group ∧ c.S.cols=c.N
  zShape : c.Z.rows=c.K/c.group ∧ c.Z.cols=c.N/8
  cShape : c.C.rows=c.M ∧ c.C.cols=c.N
  aContiguous : c.A.sr=c.K ∧ c.A.sc=1
  aValid : c.A.Valid
  qValid : c.Q.Valid
  sValid : c.S.Valid
  zValid : c.Z.Valid
  cValid : c.C.Valid

def Layout.call (c : Layout) : Call :=
  ⟨c.M,c.N,c.K,c.group,c.A,c.Q,c.S,c.Z,c.C,Bool.true⟩

def Layout.ofCall (c : Call) : Layout :=
  ⟨c.M,c.N,c.K,c.group,c.A,c.Q,c.S,c.Z,c.C⟩

theorem Layout.valid (c : Layout) (h : c.Valid) : c.call.Valid :=
  ⟨h.m,h.n,h.k,h.g,h.n256,h.k128,h.g128,h.groups,h.aShape,h.qShape,h.sShape,h.zShape,h.cShape,h.aContiguous,h.aValid,h.qValid,h.sValid,h.zValid,h.cValid,by simp [Layout.call]⟩

theorem Layout.ofCall_valid (c : Call) (h : c.Valid) : (Layout.ofCall c).Valid :=
  ⟨h.m,h.n,h.k,h.g,h.n256,h.k128,h.g128,h.groups,h.aShape,h.qShape,h.sShape,h.zShape,h.cShape,h.aContiguous,h.aValid,h.qValid,h.sValid,h.zValid,h.cValid⟩

/-- The wrapper's supplied-output flag does not occur in source execution. -/
theorem source_ofCall (c : Call) (i : Fin 17) (initial : State ℝ) :
    KernelSource.run KernelReal.scalar KernelReal.dot (Layout.ofCall c).call i initial=
      KernelSource.run KernelReal.scalar KernelReal.dot c i initial := rfl

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.RawKernel
