import bench.repairs.tritonbench_g.fp4_to_bf16.IO
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPContract

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- A source load observes a bounded logical byte. A Cell.read binding is
enough; this avoids adding a stricter tagged-cell premise to shared IO. -/
theorem read_lane_table (p : Parameters) (s : State Word)
    (bytes : ∀ i : Fin p.block,inputOffset p s i.val<p.count →
      (s.mem p.input (inputOffset p s i.val)).read .nat<256)
    (j : TileIndex [p.block*2]) (live : active p s j) :
    converted (laneNibble p s j)=expectedValue p s j := by
  have parent := active_parent p s j live
  have lane : j.1.val/2<p.block := by have := j.1.isLt;omega
  let b : BitVec 8 := BitVec.ofNat 8 ((s.mem p.input (inputOffset p s (j.1.val/2))).read .nat)
  have read : (s.mem p.input (inputOffset p s (j.1.val/2))).read .nat=b.toNat := by
    dsimp [b]
    simp only [BitVec.toNat_ofNat]
    exact (Nat.mod_eq_of_lt (bytes ⟨j.1.val/2,lane⟩ parent)).symm
  have loaded : FPValue.loaded p s (j.1.val/2)=b.toNat := by
    simp only [FPValue.loaded,parent,if_true,read]
  simp only [laneNibble,loaded,expectedValue,initialNibble,read,BitVec.ofNat_toNat,BitVec.setWidth_eq]
  split_ifs
  · rw [(byte_halves b).1,converted_table]
  · rw [(byte_halves b).2,converted_table]

theorem read_execution_memory (rest : Algebra Word) (p : Parameters) (s : State Word)
    (bytes : ∀ i : Fin p.block,inputOffset p s i.val<p.count →
      (s.mem p.input (inputOffset p s i.val)).read .nat<256) :
    (execution rest p s).val.mem=(written p s).mem := by
  rw [raw_execution_memory]
  have same : rawWritten p s=written p s := by
    unfold rawWritten written
    congr 1
    funext t j
    by_cases live : active p s j
    · simp only [live,if_true,read_lane_table p s bytes j live]
    · simp only [live,if_false]
  rw [same]

def Reads (h : Input) (values : Inputs A h) (s : State Word) : Prop :=
  ∀ i : Fin h.shape.count,(s.mem h.input i.val).read .nat=byte values i

theorem logical_nibble (h : Input) (values : Inputs A h) (s : State Word)
    (reads : Reads h values s) (j : Fin (h.shape.count*2)) :
    Host.nibble h s j.val=nibble values j := by
  simp only [Host.nibble,nibble,reads ⟨j.val/2,by have := j.isLt;omega⟩]

/-- Complete logical cell map retains every raw exterior cell. It reads only
the independent logical byte channel, never an execution witness. -/
def targetMemory (h : Input) (values : Inputs A h) (s : State Word) : RegionName → Nat → Cell Word :=
  fun r offset => if live : r=h.output ∧ offset<h.shape.count*2 then
    .mk .bf16 (embed ⟨Bits.expectedBits (nibble values ⟨offset,live.2⟩)⟩) else s.mem r offset

theorem program_memory (rest : Algebra Word) (h : Input) (values : Inputs A h)
    (domain : ByteDomain h values) (s : State Word) (reads : Reads h values s)
    (pid : Nat) (r : RegionName) (offset : Nat) :
    (execution rest (Host.parameters h) (programState h pid s)).val.mem r offset=
      if live : r=h.output ∧ offset<h.shape.count*2 ∧ offset/1024=pid then
        .mk .bf16 (embed ⟨Bits.expectedBits (nibble values ⟨offset,live.2.1⟩)⟩)
      else s.mem r offset := by
  have bytes : ∀ i : Fin (Host.parameters h).block,
      inputOffset (Host.parameters h) (programState h pid s) i.val<(Host.parameters h).count →
      ((programState h pid s).mem (Host.parameters h).input
        (inputOffset (Host.parameters h) (programState h pid s) i.val)).read .nat<256 := by
    intro i live
    change (s.mem h.input (pid*512+i.val)).read .nat<256
    rw [reads ⟨pid*512+i.val,live⟩]
    exact domain _
  rw [read_execution_memory rest _ _ bytes]
  by_cases here : r=h.output ∧ offset<h.shape.count*2 ∧ offset/1024=pid
  · obtain ⟨hr,hn,hp⟩ := here
    subst r
    let j : TileIndex [(Host.parameters h).block*2] :=
      (⟨offset%1024,by dsimp [Host.parameters];omega⟩,PUnit.unit)
    have off : outputOffset (Host.parameters h) (programState h pid s) j=offset := by
      dsimp [outputOffset,Host.parameters,programState,j];omega
    have active : FPValue.active (Host.parameters h) (programState h pid s) j := by
      change outputOffset (Host.parameters h) (programState h pid s) j<h.shape.count*2
      rw [off];exact hn
    have value := readback (Host.parameters h) (programState h pid s) j active
    rw [off] at value
    change (written (Host.parameters h) (programState h pid s)).mem h.output offset=_ at value
    rw [value]
    have parent : pid*512+j.1.val/2=offset/2 := by dsimp [j];omega
    have parity : j.1.val%2=offset%2 := by dsimp [j];omega
    simp only [expectedValue,initialNibble,inputOffset,Host.parameters,programState,
      ↓reduceIte,parent,parity]
    have hb := reads ⟨offset/2,by omega⟩
    simp only [hb]
    simp only [hp,hn,and_self,dite_true,nibble]
  · have untouched : r≠(Host.parameters h).output ∨
        ∀ j : TileIndex [(Host.parameters h).block*2],
          active (Host.parameters h) (programState h pid s) j →
          offset≠outputOffset (Host.parameters h) (programState h pid s) j := by
      by_cases hr : r=h.output
      · right
        intro j live off
        have bound := j.1.isLt
        dsimp [active,outputOffset,Host.parameters,programState] at live off bound
        apply here
        exact ⟨hr,by omega,by omega⟩
      · exact Or.inl hr
    rw [frame _ _ _ _ untouched]
    simp only [here,dite_false,programState]

#axiomsClean program_memory
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
