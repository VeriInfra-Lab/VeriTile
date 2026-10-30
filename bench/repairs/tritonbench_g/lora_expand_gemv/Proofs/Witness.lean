import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Geometry
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Witness
noncomputable section

def geometry : Geometry where
  B := 2
  L := 2
  N := 2
  K := 1
  xm := 1
  xk := 1
  l0 := 2
  wn := 1
  wk := 1
  ym := 2
  yn := 1
  indexStart := 0
  indexStride := 2
  indexCapacity := 3
  xCapacity := 2
  wCapacity := 4
  yCapacity := 4
  add := Bool.true

def raw (i : Nat) : Int := if i=0 then 0 else if i=2 then 1 else -1

theorem domain_nonempty : Valid geometry raw := by
  constructor
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · intro b hb
    change b<2 at hb
    interval_cases b <;> decide +kernel
  · intro b hb
    change b<2 at hb
    interval_cases b <;> decide +kernel
  · intro b k hb hk
    change b<2 at hb
    change k<1 at hk
    interval_cases b <;> interval_cases k <;> decide +kernel
  · intro l n k hl hn hk
    change l<2 at hl
    change n<2 at hn
    change k<1 at hk
    interval_cases l <;> interval_cases n <;> interval_cases k <;> decide +kernel
  · intro b n hb hn
    change b<2 at hb
    change n<2 at hn
    interval_cases b <;> interval_cases n <;> decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel

theorem stride2_distinguishes_logical_index :
    raw 1=-1 ∧ logicalIndex geometry raw 1=1 ∧ denseIndices geometry raw 1=1 := by
  decide +kernel

#axiomsClean domain_nonempty
#axiomsClean stride2_distinguishes_logical_index
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Witness
