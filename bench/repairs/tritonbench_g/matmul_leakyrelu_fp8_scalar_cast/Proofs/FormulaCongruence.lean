import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Target
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.ReadProjection

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FormulaCongruence
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution ReadProjection
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Internal storage labels commute with the existing geometry adapter. -/
def virtualBuffers (b : KernelBuffers.Buffers) : KernelBuffers.Buffers :=
  ⟨{b.A with storage:=0},{b.B with storage:=1},{b.C with storage:=2}⟩

theorem virtual_views (b : KernelBuffers.Buffers) (g : KernelBuffers.Geometry) :
    (virtualBuffers b).views g=views (b.views g) := rfl

theorem a_value (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M ops g b).InputBinding vals s)
    (i k : Nat) (hi : i<g.M) (hk : k<g.K) :
    decode (b.views g) s ((views (b.views g)).A.address i k)=IO.matrixA g vals i k := by
  rw [show (views (b.views g)).A.address i k=(0,(b.views g).A.offset+(i*b.A.rowStride+k*b.A.columnStride)) by
    simp [Tensor.address,views,KernelBuffers.Buffers.views,KernelBuffers.Buffer.view,Nat.add_assoc]]
  rw [decode_a]
  have bound : (s.mem "A" (i*b.A.rowStride+k*b.A.columnStride)).read .real=
      vals 0 (⟨i,hi⟩,⟨k,hk⟩,()) := h (0 : Fin 2) (⟨i,hi⟩,⟨k,hk⟩,()) trivial
  simpa [IO.matrixA,hi,hk] using bound

theorem b_value (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M ops g b).InputBinding vals s)
    (k j : Nat) (hk : k<g.K) (hj : j<g.N) :
    decode (b.views g) s ((views (b.views g)).B.address k j)=IO.matrixB g vals k j := by
  rw [show (views (b.views g)).B.address k j=(1,(b.views g).B.offset+(k*b.B.rowStride+j*b.B.columnStride)) by
    simp [Tensor.address,views,KernelBuffers.Buffers.views,KernelBuffers.Buffer.view,Nat.add_assoc]]
  rw [decode_b]
  have bound : (s.mem "B" (k*b.B.rowStride+j*b.B.columnStride)).read .real=
      vals 1 (⟨k,hk⟩,⟨j,hj⟩,()) := h (1 : Fin 2) (⟨k,hk⟩,⟨j,hj⟩,()) trivial
  simpa [IO.matrixB,hk,hj] using bound

theorem native_acc (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M ops g b).InputBinding vals s)
    (i j : Nat) (hi : i<g.M) (hj : j<g.N) (count : Nat) :
    referenceAcc ops ((virtualBuffers b).arguments g) (decode (b.views g) s) i j count=
      IO.accumulated ops g.K (IO.matrixA g vals) (IO.matrixB g vals) i j count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    have ha : (fun k : Fin 64 => if count*64+k.val<g.K then
        decode (b.views g) s ((views (b.views g)).A.address i (count*64+k.val)) else ops.zero)=
        (fun k : Fin 64 => if count*64+k.val<g.K then IO.matrixA g vals i (count*64+k.val) else ops.zero) := by
      funext k
      split
      · exact a_value M ops g b vals s h i _ hi (by assumption)
      · rfl
    have hb : (fun k : Fin 64 => if count*64+k.val<g.K then
        decode (b.views g) s ((views (b.views g)).B.address (count*64+k.val) j) else ops.zero)=
        (fun k : Fin 64 => if count*64+k.val<g.K then IO.matrixB g vals (count*64+k.val) j else ops.zero) := by
      funext k
      split
      · exact b_value M ops g b vals s h _ j (by assumption) hj
      · rfl
    change block ops
      (fun k => if count*64+k.val<g.K then decode (b.views g) s ((views (b.views g)).A.address i (count*64+k.val)) else ops.zero)
      (fun k => if count*64+k.val<g.K then decode (b.views g) s ((views (b.views g)).B.address (count*64+k.val) j) else ops.zero)
      (referenceAcc ops ((virtualBuffers b).arguments g) (decode (b.views g) s) i j count)=_
    rw [ih,ha,hb]
    rfl

theorem native (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues α (IO.inputs g)) (s : State α)
    (h : (IO.matmulIO M ops g b).InputBinding vals s)
    (i j : Nat) (hi : i<g.M) (hj : j<g.N) :
    reference ops ((virtualBuffers b).arguments g) (decode (b.views g) s) i j=
      IO.native ops g (IO.matrixA g vals) (IO.matrixB g vals) i j := by
  simp only [reference,KernelBuffers.arguments_problem,IO.native]
  rw [native_acc M ops g b vals s h i j hi hj]
  rfl

theorem mathematical (M : Algebra ℝ) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (vals : ChannelValues ℝ (IO.inputs g)) (s : State ℝ)
    (h : (IO.matmulIO M realOps g b).InputBinding vals s)
    (i j : Nat) (hi : i<g.M) (hj : j<g.N) :
    Real.formula g.problem (views (b.views g)) (decode (b.views g) s) i j=
      Target.mathematical g (IO.matrixA g vals) (IO.matrixB g vals) i j := by
  apply congrArg (Real.activate g.enabled)
  apply Finset.sum_congr rfl
  intro k hk
  have hk' : k<g.K := Finset.mem_range.mp hk
  rw [show (Real.inputs (views (b.views g)) (decode (b.views g) s)).a i k=IO.matrixA g vals i k from
      a_value M realOps g b vals s h i k hi hk',
    show (Real.inputs (views (b.views g)) (decode (b.views g) s)).b k j=IO.matrixB g vals k j from
      b_value M realOps g b vals s h k j hk' hj]

#axiomsClean native
#axiomsClean mathematical
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FormulaCongruence
