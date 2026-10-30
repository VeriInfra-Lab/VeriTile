import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CumulativeFP
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000
def oldStorage : NativeSource.Storage →
    VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedSource.Storage
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

section Recurrent
variable (dot gate history : NativeSource.Storage) (seed final : FloatDType)
    (k v g h h0 ht : RegionName)
    (kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base : Nat)
    (gateK hasSeed storeFinal : Bool)
def recurrent : Program where
  kernel := NativeSource.hOriginal dot gate history seed final k v g h h0 ht
    kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal
  domain := FPEquivProofs.requirements
    (NativeSource.hOriginal dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal)
    (oldStorage dot) BK BT BV

def recurrentSumLeft : Program :=
  { recurrent dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal with
    kernel := NativeSource.hSumLeft dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal }

theorem recurrent_domain_present :
    (recurrent dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal).domain.length = 1 := by
  cases gateK <;> cases dot <;> cases gate <;> cases history <;> rfl

theorem recurrent_equiv (R : FP.ScalarArithmetic.Rules) :
    recurrent dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal ≡[R]
    recurrentSumLeft dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal := by
  cases gateK <;> cases dot <;> cases gate <;> cases history <;>
    equiv_decompose <;> fp_prove
end Recurrent

#print_fp_assumptions cumulative_equiv
#print_fp_assumptions recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
