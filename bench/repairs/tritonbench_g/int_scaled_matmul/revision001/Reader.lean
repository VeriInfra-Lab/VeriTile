import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Kernels
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Correct
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.FPEquiv
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.HostBounds
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Reader
open VeriTile.Triton FP.Structural Host Kernels
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem read_int_matmul_correct (rest : Algebra Word) (x : Arguments) (initial : State Word)
    (h : Valid x) (_ha : Int8Storage x.a initial) (_hb : Int8Storage x.b initial) :
    ∃ final, Host.int_matmul_kernel rest x initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val) := by
  exact VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.int_matmul_correct rest x initial h _ha _hb

#axiomsClean read_int_matmul_correct

theorem read_int_scaled_matmul_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (initial : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (_ha : Int8Storage x.a initial) (_hb : Int8Storage x.b initial) (_hs : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) initial) :
    ∃ final, Host.int_scaled_matmul_kernel rest cfg u x scale initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => ScaledWord.logicalResult cfg (scaledParameters x scale) initial i.1.val i.2.val) := by
  exact VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.int_scaled_matmul_correct rest cfg u x scale initial h hs _ha _hb _hs hd

#axiomsClean read_int_scaled_matmul_correct

theorem read_scaled_result_conversion (cfg : FP.Config) (x : Arguments) (scale : Tensor) (s : State Word)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) s) (i : Index x) :
    ∃ q, (ScaledWord.logicalScaled cfg (scaledParameters x scale) s i.1.val i.2.val).toRat?=some q ∧
      FP.IntegerConversion.TowardZero q (ScaledWord.logicalResult cfg (scaledParameters x scale) s i.1.val i.2.val) ∧
      FP.IntegerConversion.SignedRange 32 (ScaledWord.logicalResult cfg (scaledParameters x scale) s i.1.val i.2.val) := by
  exact VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.scaled_result_conversion cfg x scale s hd i

#axiomsClean read_scaled_result_conversion

theorem read_zero_inner_unscaled (x : Arguments) (s : State Word) (hK : x.a.columns=0) (r c : Nat) :
    BlockWord.logicalDot (blockParameters x) s r c=0 := by
  exact VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.zero_inner_unscaled x s hK r c

#axiomsClean read_zero_inner_unscaled

theorem read_launch_cover (x : Arguments) (scale : Tensor) (h : Valid x) :
    (scaledParameters x scale).inner≤(scaledParameters x scale).steps*(scaledParameters x scale).blockInner := by
  exact Host.launch_cover x scale h

#axiomsClean read_launch_cover

theorem read_launch_even (x : Arguments) (scale : Tensor) (h : Valid x) :
    (scaledParameters x scale).even=Bool.true →
    (scaledParameters x scale).inner=(scaledParameters x scale).steps*(scaledParameters x scale).blockInner := by
  exact Host.launch_even x scale h

#axiomsClean read_launch_even

theorem read_shape_rejected (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (s : State Word) (h : ¬shapeCompatible x) :
    int_matmul_kernel rest x s=none ∧ int_scaled_matmul_kernel rest cfg u x scale s=none := by
  exact Host.shape_rejected rest cfg u x scale s h

#axiomsClean read_shape_rejected

theorem read_empty_return (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (s : State Word) (hs : shapeCompatible x) (he : emptyOutput x) :
    int_matmul_kernel rest x s=some (x.c,s) ∧
    int_scaled_matmul_kernel rest cfg u x scale s=some (x.c,s) := by
  exact Host.empty_return rest cfg u x scale s hs he

#axiomsClean read_empty_return

theorem read_block_accesses (x : Arguments) (h : Valid x) (s : State Word) :
    (∀ k (i : BlockWord.AIndex (blockParameters x)),
      (BlockWord.aPointer (blockParameters x) s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
      (BlockWord.aPointer (blockParameters x) s k).address (TileShape.indexToList _ i)<x.a.capacity) ∧
    (∀ k (i : BlockWord.BIndex (blockParameters x)),
      (BlockWord.bPointer (blockParameters x) s k).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
      (BlockWord.bPointer (blockParameters x) s k).address (TileShape.indexToList _ i)<x.b.capacity) ∧
    (∀ (i : BlockWord.CIndex (blockParameters x)), BlockWord.active (blockParameters x) s i →
      BlockWord.outputOffset (blockParameters x) s i<x.c.capacity) := by
  exact Host.block_accesses x h s

#axiomsClean read_block_accesses

theorem read_scaled_accesses (x : Arguments) (scale : Tensor) (h : Valid x) (hs : ScaleValid x scale)
    (hne : ¬emptyOutput x) (s : State Word) :
    (∀ k, k<(scaledParameters x scale).steps → ∀ (i : ScaledWord.AIndex (scaledParameters x scale)),
      ((scaledParameters x scale).even=Bool.true ∨
        ScaledWord.inputInner (scaledParameters x scale) s i.2.1<
        (scaledParameters x scale).inner-k*(scaledParameters x scale).blockInner) →
      ScaledWord.aOffset (scaledParameters x scale) s k i<x.a.capacity) ∧
    (∀ k, k<(scaledParameters x scale).steps → ∀ (i : ScaledWord.BIndex (scaledParameters x scale)),
      ((scaledParameters x scale).even=Bool.true ∨
        ScaledWord.inputInner (scaledParameters x scale) s i.1<
        (scaledParameters x scale).inner-k*(scaledParameters x scale).blockInner) →
      ScaledWord.bOffset (scaledParameters x scale) s k i<x.b.capacity) ∧
    (∀ (i : ScaledWord.CIndex (scaledParameters x scale)), ScaledWord.active (scaledParameters x scale) s i →
      ScaledWord.outputOffset (scaledParameters x scale) s i<x.c.capacity) ∧
    (∀ (i : ScaledWord.CIndex (scaledParameters x scale)), ScaledWord.active (scaledParameters x scale) s i →
      ScaledWord.outputRow (scaledParameters x scale) s i*(scaledParameters x scale).scaleRowStride<scale.capacity) := by
  exact Host.scaled_accesses x scale h hs hne s

#axiomsClean read_scaled_accesses

theorem read_scaled_backend_independent (rest₁ rest₂ : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (x : Arguments) (scale : Tensor) (initial a b : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (ha : Int8Storage x.a initial) (hb : Int8Storage x.b initial) (hscale : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) initial)
    (ea : Host.int_scaled_matmul_kernel rest₁ cfg u x scale initial=some (x.c,a))
    (eb : Host.int_scaled_matmul_kernel rest₂ cfg v x scale initial=some (x.c,b)) : a.mem=b.mem := by
  exact VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.scaled_backend_independent rest₁ rest₂ cfg u v x scale initial a b h hs ha hb hscale hd ea eb

#axiomsClean read_scaled_backend_independent

theorem read_unscaled_binding (x : Host.Arguments) : unscaled x=BlockWord.source (Host.blockParameters x) := by
  exact Kernels.unscaled_binding x

#axiomsClean read_unscaled_binding

theorem read_scaled_binding (x : Host.Arguments) (scale : Host.Tensor) :
    scaled x scale=ScaledWord.source (Host.scaledParameters x scale) := by
  exact Kernels.scaled_binding x scale

#axiomsClean read_scaled_binding

theorem read_group_coverage (g : Geometry.Layout) (h : Geometry.Valid g) (x : Geometry.Cell g) :
    ∃ pid, pid<Geometry.count g ∧ Geometry.Owns g pid x := by
  exact Geometry.covers g h x

theorem read_group_disjoint (g : Geometry.Layout) (h : Geometry.Valid g) (p q : Nat)
    (hp : p<Geometry.count g) (hq : q<Geometry.count g) (hpq : p≠q)
    (i j : Geometry.Lane g) (hi : Geometry.Active g p i) (hj : Geometry.Active g q j) :
    Geometry.offset g p i≠Geometry.offset g q j := by
  exact Geometry.stores_disjoint g h p q hp hq hpq i j hi hj

#axiomsClean read_group_coverage
#axiomsClean read_group_disjoint

theorem read_native_correct (rest : Algebra Word) (native : Native.Primitives)
    (numeric : Native.PrimitiveCorrespondence native)
    (x : Arguments) (scale : Tensor) (initial : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (ha : Int8Storage x.a initial) (hb : Int8Storage x.b initial) (hscale : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined Native.sourceConfig (scaledParameters x scale) initial) :
    ∃ final, Native.call rest native x scale initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => ScaledWord.logicalResult Native.sourceConfig
        (scaledParameters x scale) initial i.1.val i.2.val) := by
  exact int_scaled_matmul_native_correct rest native numeric x scale initial h hs ha hb hscale hd

theorem read_native_call_binding (rest : Algebra Word) (native : Native.Primitives)
    (numeric : Native.PrimitiveCorrespondence native) (x : Arguments) (scale : Tensor) (initial : State Word) :
    Native.call rest native x scale initial =
      Host.int_scaled_matmul_kernel rest Native.sourceConfig (Native.fallback native) x scale initial := by
  exact Native.call_eq rest native numeric x scale initial

#axiomsClean read_native_correct
#axiomsClean read_native_call_binding
#print Native.PrimitiveCorrespondence
#print Native.sourceConfig
#print ScaledWord.LogicalDefined
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Reader
