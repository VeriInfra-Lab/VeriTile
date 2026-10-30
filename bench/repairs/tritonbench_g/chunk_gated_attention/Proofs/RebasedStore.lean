import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedAccess

/-! Reuse the existing complete tagged-cell store/readback/frame proof by
instantiating its parent with the actual local tile, then identify its scatter
with the global logical tile. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStore
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHTiles
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000

def localLayout (p : Layout) : Layout := {p with K := p.BK, V := p.BV}
def zeroLaunch : Launch := ⟨0,0,0⟩
def shifted (p : Layout) (l : Launch) (base sr sc : Nat) :=
  base+l.key*p.BK*sr+l.value*p.BV*sc

theorem scatter_eq (p : Layout) (l : Launch) (out : RegionName)
    (base sr sc : Nat) (f : StateValue p) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V) :
    RealHStore.scatter (localLayout p) zeroLaunch out (shifted p l base sr sc) sr sc f u =
      RealHStore.scatter p l out base sr sc f u := by
  unfold RealHStore.scatter
  apply congrArg (fun step => List.foldl step u (TileShape.allIndices [p.BK,p.BV]))
  funext st idx
  rcases idx with ⟨e,j,⟨⟩⟩
  have he : e.val < p.BK := e.isLt
  have hj : j.val < p.BV := j.isLt
  have hek : l.key*p.BK+e.val < p.K := by omega
  have hjv : l.value*p.BV+j.val < p.V := by omega
  simp only [RealHStore.active,RealHStore.address,localLayout,zeroLaunch,keyIndex,valueIndex,
    Nat.zero_mul,Nat.zero_add,he,hj,hek,hjv,and_self,if_true,shifted]
  congr 1 <;> ring

theorem store_eval (p : Layout) (l : Launch) (out : RegionName)
    (base sr sc : Nat) (f : StateValue p) (name : RegName) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f))
    (hp : u.regs .blockPtr [p.BK,p.BV] name =
      some (RebasedAccess.matrix out (shifted p l base sr sc) p.BK p.BV sr sc)) :
    stepStmt (.store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] name) [0,1])
      (.ref .real [p.BK,p.BV] "b_h") .none) u =
      some (RealHStore.scatter p l out base sr sc f u) := by
  have h := RealHStore.store_eval (localLayout p) zeroLaunch out (shifted p l base sr sc)
    sr sc f name u hh (by simpa [RealHStore.pointer,RebasedAccess.matrix,
      RealHPointers.matrixPointer,localLayout,zeroLaunch] using hp)
  rw [scatter_eq p l out base sr sc f u hk hv] at h
  exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStore
