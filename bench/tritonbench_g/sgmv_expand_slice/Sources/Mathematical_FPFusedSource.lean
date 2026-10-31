import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPContract
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
/-! Separate explicit-accumulator graph observed in the compiled one-comma
candidate. No equivalence to the original independent dot+add AST is asserted. -/

structure DotAcc where
  apply : Option ComputeDType → (left right output : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    (TileIndex (batch++[m,n]) → Word) → (TileIndex (batch++[m,n]) → Word)

def installAcc (rest : Algebra Word) (F : DotAcc) : Algebra Word :=
  {rest with
    dotAcc := fun p l r o opts {_} {_} {_} {_} a b c =>
      some (fun i => ofFloat o (F.apply p l r o opts (fun j => toFloat l (a j))
        (fun j => toFloat r (b j)) (fun j => toFloat o (c j)) i))}

def unusedDot : Dot := ⟨fun _ _ _ _ _ {_} {_} {_} {_} _ _ _ => FP.Value.qNaN .fp32⟩
def fusedBackend (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc) := backend (installAcc rest F) cfg unusedDot

def fusedUpdate [Inhabited Word] (F : DotAcc) (fmt : Storage) (p : Parameters) (s : State Word)
    (cur : Nat) (acc : Accumulator p) : Accumulator p :=
  F.apply (some .fp32) (storageType fmt) (storageType fmt) .real {} (batch := [])
    (aValues fmt p s cur) (bValues fmt p s cur) acc

def fusedAccumulated [Inhabited Word] (F : DotAcc) (fmt : Storage) (p : Parameters) (s : State Word) :=
  ParallelAttention.FPStages.foldRange (fusedUpdate F fmt p s) 0 p.inner p.blockInner (initial p)
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
