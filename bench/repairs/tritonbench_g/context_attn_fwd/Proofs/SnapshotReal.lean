import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot
open VeriTile.Triton NativeGraph
noncomputable section

/-- Source arguments have distinct floating and integer logical role labels;
physical bases may still overlap between any read-only roles. -/
def LogicalRoles (c : Host.Call) : Prop :=
  ∀ rg,rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out] → rg∉metadataRegions c

/-- Fixed IEEE binary32-to-Real abstraction. All finite words have their usual
value, signed zeros both map to zero. Nonfinite words map to zero solely to
make the independent Real abstraction total; the native theorem retains their
original bits and makes no equality/error claim against this abstraction. -/
def real32 (word : Word) : ℝ :=
  let n : Nat := word.toNat
  let exponent : Nat := n/2^23%256
  let fraction : Nat := n%2^23
  let sign : ℝ := if n/2^31=0 then 1 else -1
  if exponent=255 then 0 else
  if exponent=0 then sign*(fraction : ℝ)*(2 : ℝ)^(-149 : Int)
  else sign*((2^23+fraction : Nat) : ℝ)*(2 : ℝ)^((exponent : Int)-150)

def realInitial (c : Host.Call) (B : Native.Bytes extent) (mem : Physical.Memory) : BlockState where
  mem := fun rg off => if rg∈metadataRegions c then
    .of .int (load32 mem (B.base rg+4*off)).toInt
    else .real (real32 (load32 mem (B.base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := c.grid.toNumPids

theorem real_metadata (c : Host.Call) (B : Native.Bytes extent) (mem : Physical.Memory)
    (hm : Metadata c B mem) : Geometry.Metadata c (realInitial c B mem) := by
  refine ⟨?_,?_,?_⟩ <;> intro b hb
  · simpa [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs,BlockState.readMemValue,BlockState.readMemTyped,
      realInitial,metadataRegions] using hm.start b hb
  · simpa [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs,BlockState.readMemValue,BlockState.readMemTyped,
      realInitial,metadataRegions] using hm.sequence b hb
  · simpa [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs,BlockState.readMemValue,BlockState.readMemTyped,
      realInitial,metadataRegions] using hm.prompt b hb

theorem real_floating (c : Host.Call) (B : Native.Bytes extent) (mem : Physical.Memory)
    (roles : LogicalRoles c) (rg : RegionName) (hr : rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out])
    (off : Nat) :
    (realInitial c B mem).readMem rg off=real32 (load32 mem (B.base rg+4*off)) := by
  simp [realInitial,BlockState.readMem,roles rg hr]

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot
