import bench.tritonbench_g.chunk_gated_attention.Sources.CumulativeFP
import bench.tritonbench_g.chunk_gated_attention.Proofs.HSourceFP
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedProgram
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000
section Recurrent
variable (dot gate history : NativeSource.Storage) (seed final : FloatDType)
    (k v g h h0 ht : RegionName)
    (kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base : Nat)
    (gateK hasSeed storeFinal : Bool)
def parameters : HSource.Parameters :=
  RebasedProgram.parameters ⟨kh,kt,kd,vh,vt,vd,hh,hk,hv,T,K,V,BT,BK,BV,NT⟩ base gateK hasSeed storeFinal Bool.false

def regions : RealHProgram.Regions := ⟨k,v,g,h,h0,ht⟩

def recurrent : Program :=
  HSourceFP.original (parameters kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal)
    (regions k v g h h0 ht) dot.dtype gate.dtype history.dtype seed final

def recurrentSumLeft : Program :=
  HSourceFP.candidate (parameters kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal)
    (regions k v g h h0 ht) dot.dtype gate.dtype history.dtype seed final

theorem recurrent_domain_present :
    (recurrent dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal).domain.length=1 :=
  HSourceFP.domain_present _ _ _ _ _ _ _

theorem recurrent_equiv (R : FP.ScalarArithmetic.Rules) :
    recurrent dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal ≡[R]
    recurrentSumLeft dot gate history seed final k v g h h0 ht
      kh kt kd vh vt vd hh hk hv T K V BT BK BV NT base gateK hasSeed storeFinal :=
  HSourceFP.equivalent _ _ _ _ _ _ _ R
end Recurrent

#print_fp_assumptions cumulative_equiv
#print_fp_assumptions recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
