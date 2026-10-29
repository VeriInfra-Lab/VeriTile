import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.UncachedInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.CachedInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.NativeRaw

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section

/-- The existing scalar primitive model fixes multiplication order and which
product contracts into FMA. This is a definitional graph identity, not an
experimental numerical assumption. Input node provenance is unrestricted. -/
theorem rotation_native (p : Native.Instructions) (cached : Bool) (a b c z : Native.Node) (odd : Bool) :
    rotation (Native.ops p cached) a b c z odd=nativeRotation p cached a b c z odd := by
  cases cached <;> cases odd <;> rfl

theorem u_target_native (p : Native.Instructions) (g : Geometry) :
    uTarget (Native.ops p Bool.false) g=uNativeTarget p g := by
  funext v o i
  change Fin 4 at o
  fin_cases o <;> apply rotation_native

theorem c_target_native (p : Native.Instructions) (g : Geometry) :
    cTarget (Native.ops p Bool.true) g=cNativeTarget p g := by
  funext v o i
  change Fin 4 at o
  fin_cases o <;> apply rotation_native

theorem uncached_native (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (p q : Nat) (heads : ∀ key,0<g.request.heads key) :
    (uncachedIO rest (Native.ops op Bool.false) g p q).Implements (uNativeTarget op g) := by
  rw [←u_target_native]
  exact uncached_implements rest (Native.ops op Bool.false) g p q heads

theorem cached_native (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (p q : Nat) (token : q<g.tokens) :
    (cachedIO rest (Native.ops op Bool.true) g p q).ImplementsOn (MetadataDomain g) (cNativeTarget op g) := by
  rw [←c_target_native]
  exact cached_implements rest (Native.ops op Bool.true) g p q token

#axiomsClean uncached_native
#axiomsClean cached_native
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
