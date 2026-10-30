import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedCachedMemory

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Masked primitive loads before the pair computation. This temporary
reference retains padding during structural normalization; its values at
active stores reduce to the independent unmasked rotation below. -/
def uMaskedValue {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s data : State A) (key odd : Bool) (i : ULane p) : Cell A :=
  let zero := rest.literal (some .fp32) .real 0.0
  let a := if uActive p s key i then (data.mem (uRegion p key) (uOffset p s key Bool.false i)).read .real else zero
  let b := if uActive p s key i then (data.mem (uRegion p key) (uOffset p s key Bool.true i)).read .real else zero
  let c := if uToken p s i<p.q_total_tokens ∧ i.2.2.1.val<p.HEAD_DIM/2 then
    (s.mem p.cos (uCoefficient p s Bool.false i)).read .real else zero
  let z := if uToken p s i<p.q_total_tokens ∧ i.2.2.1.val<p.HEAD_DIM/2 then
    (s.mem p.sin (uCoefficient p s Bool.true i)).read .real else zero
  .mk .real (if odd then ops.add (ops.mul a z) (ops.mul b c)
    else ops.sub (ops.mul a c) (ops.mul b z))

def uMaskedBranch {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s data : State A) (key : Bool) (entry : State A) : State A :=
  Pair.written (uRegion p key) (uActive p s key) (uOffset p s key)
    (uMaskedValue rest ops p s data key) entry

theorem u_masked_branch {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s data : State A) (key : Bool) (entry : State A) :
    uMaskedBranch rest ops p s data key entry=uBranch ops p s data key entry := by
  have values (odd : Bool) (i : ULane p) (hi : uActive p s key i) :
      uMaskedValue rest ops p s data key odd i=uValue ops p s data key odd i := by
    simp [uMaskedValue,uValue,hi,hi.2.1,hi.2.2]
  have step (odd : Bool) :
      (fun (t : State A) (i : ULane p) => if uActive p s key i then
        t.write (uRegion p key) (uOffset p s key odd i) (uMaskedValue rest ops p s data key odd i) else t)=
      (fun (t : State A) (i : ULane p) => if uActive p s key i then
        t.write (uRegion p key) (uOffset p s key odd i) (uValue ops p s data key odd i) else t) := by
    funext t i
    by_cases hi : uActive p s key i <;> simp [hi,values odd i]
  simp only [uMaskedBranch,uBranch,Pair.written,Pair.halfScatter,step]

def uMaskedReference {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s : State A) : State A :=
  let q := uMaskedBranch rest ops p s s Bool.false s
  if s.pids 0%p.KV_GROUP_NUM=0 then uMaskedBranch rest ops p s q Bool.true q else q

theorem u_masked_reference {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s : State A) :
    uMaskedReference rest ops p s=uReference ops p s := by
  simp only [uMaskedReference,uReference,u_masked_branch]

theorem uncached_memory {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.UncachedLayout) (s : State A) :
    (uncachedExecution rest ops p s).val.mem=(uReference ops p s).mem := by
  rw [←u_masked_reference rest ops p s]
  by_cases leader : s.pids 0%p.KV_GROUP_NUM=0
  all_goals
    simp only [uncachedExecution,uMaskedReference,leader,↓reduceDIte,↓reduceIte,uMaskedBranch,
      Pair.written,Pair.halfScatter,uMaskedValue,uOffset,uCoefficient,uToken,uHead,uDim,uActive,uRegion,
      algebra,VeriTile.Bench.Repairs.RotaryEmb.Typed.algebra,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,masked_scatter_memory]
    dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.ofNat_lt]
    all_goals simp (config := { maxSteps := 3000000 }) [and_assoc,dimension_projection]
    try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
    try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      Int.toNat_natCast,Int.ofNat_lt]
    try simp (config := { maxSteps := 3000000 }) [and_assoc,dimension_projection]
    rfl

#axiomsClean uncached_memory
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Typed
