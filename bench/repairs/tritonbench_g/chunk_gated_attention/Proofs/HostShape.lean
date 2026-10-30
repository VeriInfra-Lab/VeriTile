import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.StagedLayout
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CumulativeHost

/-! The actual host ceil-division/padding shapes imply the layout contracts.
These facts impose no divisibility restriction on caller logical dimensions. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostShape
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention

def chunks (n b : Nat) := (n+b-1)/b
def padded (n b : Nat) := chunks n b*b

theorem le_padded (n b : Nat) (hb : 0<b) : n ≤ padded n b := by
  have hm := Nat.mod_lt (n+b-1) hb
  have he := Nat.mod_add_div (n+b-1) b
  rw [Nat.mul_comm b] at he
  unfold padded chunks
  omega

theorem padded_div (n b : Nat) (hb : 0<b) : padded n b/b=chunks n b := by
  simp [padded,hb.ne']

def recurrent (T K V BT BK BV : Nat) : RealHMath.Layout :=
  let NT := chunks T BT
  let PT := padded T BT
  let PK := padded K BK
  let PV := padded V BV
  ⟨PT*PK,PK,1,PT*PV,PV,1,NT*PK*PV,PV,1,PT,PK,PV,BT,BK,BV,NT⟩

theorem recurrent_canonical (T K V BT BK BV : Nat) (ht : 0<BT) (hk : 0<BK) (hv : 0<BV) :
    StagedLayout.Canonical (recurrent T K V BT BK BV) := by
  constructor <;> simp [recurrent,padded,chunks,ht.ne',hk.ne',hv.ne',ht,hk,hv]

def cumulative (T S BT BS : Nat) : RealCumProgram.Layout :=
  let PT := padded T BT
  let PS := padded S 64
  ⟨PT*PS,PS,1,PT,PS,BT,BS⟩

theorem cumulative_canonical (T S BT BS : Nat) (ht : 0<BT)
    (hs : BS=16 ∨ BS=32 ∨ BS=64) : CumulativeHost.Canonical (cumulative T S BT BS) := by
  rcases hs with rfl|rfl|rfl
  all_goals constructor <;> simp [cumulative,padded,chunks,ht.ne',ht]
  all_goals omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostShape
