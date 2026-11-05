import bench.tritonbench_g.kv_cache_filling.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.KvCacheFilling.Correct
open VeriTile Triton FP.Structural
noncomputable section

/-- IO.lean retains every typed input/output, the schedule, policy masks and
source/destination addresses through bijective named mathematical views. -/
def cacheIO (r : Request) (j : Job) (version : Bool) (pid : Nat) :=
  LogicalIO.io QuantReal.ops r j version pid

/-- Affine scale on the complete input head, including constant rows. -/
def affineScale (levels : Nat) (lo hi : ℝ) : ℝ :=
  if hi = lo then 1 else (hi-lo)/levels

/-- Nearest nonnegative integer code represented as one wrapping byte. -/
def affineCode (levels : Nat) (lo hi x : ℝ) : BitVec 8 :=
  BitVec.ofNat 8 (Int.floor ((x-lo)/affineScale levels lo hi+1/2)).toNat

/-- Copy, one-byte affine quantization, or two half-head codes packed into
one byte. Scale/zero are the two metadata lanes of each quantized row. -/
def affineRow (policy : Policy) (w : Nat) (x : Nat → ℝ) (lo hi : ℝ) : RowResult ℝ (BitVec 8) :=
  match policy with
  | .copy => ⟨fun i => .fp32 (x i), 0, 0⟩
  | .int8 => ⟨fun i => .uint8 (affineCode 255 lo hi (x i)),
      affineScale 255 lo hi, -lo/affineScale 255 lo hi⟩
  | .int4 => ⟨fun i => .uint8 (affineCode 15 lo hi (x i)+affineCode 15 lo hi (x (i+w))*16),
      affineScale 15 lo hi, -lo/affineScale 15 lo hi⟩

/-- Extrema range over exactly dk/dv input coordinates, before block padding.
Both cache rows and both scale/zero pairs are part of the result. -/
def formula (r : Request) (x : LogicalIO.Inputs r) : LogicalIO.Outputs r :=
  let k := LogicalIO.extend x.key
  let v := LogicalIO.extend x.value
  let kr := affineRow r.policy r.ck k
    (QuantReal.low r.dk (fun i => k i.val)) (QuantReal.high r.dk (fun i => k i.val))
  let vr := affineRow r.policy r.cv v
    (QuantReal.low r.dv (fun i => v i.val)) (QuantReal.high r.dv (fun i => v i.val))
  ⟨fun i => Interface.cacheValue r.policy (kr.value i.val),
   fun i => Interface.cacheValue r.policy (vr.value i.val),
   fun i => if i.val = 0 then kr.scale else kr.zero,
   fun i => if i.val = 0 then vr.scale else vr.zero⟩

/-- Every scheduled job, both source versions and every pid: valid accesses,
successful execution, all cache/metadata outputs and exact exterior frame.
The input domain fixes only the two decoded scheduling integers. -/
specification whole_shared_real (r : Request) (d : Domain r) :
    Spec.Real (∀ j ∈ jobs r, ∀ version pid,
      (cacheIO r j version pid).ImplementsOn (LogicalIO.domain r j) (formula r)) := by
  intro j _ version pid
  exact LogicalIO.real_implements r d j version pid

#specNonCircular formula avoiding [FP.Structural.exec,
  SharedTarget.value, QuantProfile.row4, QuantProfile.row8]
#axiomsClean whole_shared_real
#stmtSurfaceSubset whole_shared_real ⊆
  [cacheIO,formula,Domain,LogicalIO.domain,LogicalIO.Inputs,LogicalIO.Outputs,Job,Request,jobs,
   Spec.Real,LogicalProgramIO.ImplementsOn]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.Correct
