import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Native
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Prefix

namespace VeriTile.Bench.Repairs.UniformSampling.Execution
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
open IndexProof
noncomputable section
open scoped Classical


def nativeMemory (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (count : Nat) (m : Nat→Store) : Nat→Store :=
  Function.update m scratch.region
    (Native.run api M scalars seed start scratch ((schedule n start).take count) (m scratch.region))

def copyMemory (api : API α Store Byte) (M : Arithmetic α) (v : View n)
    (scratch : Scratch n) (count : Nat) (m : Nat→Store) : Nat→Store :=
  VeriTile.Bench.Repairs.UniformSampling.Shared.copyPrefix api count n v.region v.address
    (fun i=>physical M v (api.read (m scratch.region) (scratchAddress scratch i))) m

def run (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (v : View n) (m : Nat→Store) : Nat→Store :=
  copyMemory api M v scratch n
    (nativeMemory api M scalars seed start scratch (schedule n start).length m)

def Outside (v : View n) (r b : Nat) : Prop :=
  r≠v.region ∨ ∀ i,i<n→b<v.address i ∨ v.address i+4≤b

theorem native_read (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (m : Nat→Store) (hs:block n∣start)
    {i : Nat} (hi:i<n) :
    api.read (nativeMemory api M scalars seed start scratch (schedule n start).length m scratch.region)
      (scratchAddress scratch i)=Native.result M scalars seed start (block n) i := by
  simp only [nativeMemory,Function.update_self,List.take_length]
  exact Native.readback api M scalars seed start scratch _ hs hi

theorem copy_full (api : API α Store Byte) (M : Arithmetic α) (v : View n)
    (scratch : Scratch n) (m : Nat→Store) :
    copyMemory api M v scratch n m=VeriTile.Bench.Repairs.UniformSampling.Shared.Copy.run api n v.region v.address
      (fun i=>physical M v (api.read (m scratch.region) (scratchAddress scratch i))) m :=
  VeriTile.Bench.Repairs.UniformSampling.Shared.copyPrefix_full _ _ _ _ _ _

theorem physical_readback (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (v : View n) (m : Nat→Store)
    (hs:block n∣start) {i : Nat} (hi:i<n) :
    api.read (run api M scalars seed start scratch v m v.region) (v.address i)=
      physical M v (Native.result M scalars seed start (block n) i) := by
  unfold run
  rw [copy_full,VeriTile.Bench.Repairs.UniformSampling.Shared.Copy.readback api n v.region v.address
    _ _ v.separate hi,native_read api M scalars seed start scratch m hs hi]

theorem logical_readback (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (v : View n) (m : Nat→Store)
    (neg:∀x,M.neg32 (M.neg32 x)=x) (hs:block n∣start) {i : Nat} (hi:i<n) :
    logical api M (run api M scalars seed start scratch v m) v i=
      Native.result M scalars seed start (block n) i := by
  unfold logical
  rw [physical_readback api M scalars seed start scratch v m hs hi]
  cases h:v.negative <;> simp [physical,h,neg]

theorem native_prefix_frame (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (count : Nat) (m : Nat→Store) (r b : Nat)
    (h:r≠scratch.region) :
    api.byte (nativeMemory api M scalars seed start scratch count m r) b=api.byte (m r) b := by
  simp [nativeMemory,Function.update_of_ne h]

theorem copy_prefix_frame (api : API α Store Byte) (M : Arithmetic α) (v : View n)
    (scratch : Scratch n) (count : Nat) (m : Nat→Store) (r b : Nat) (h:Outside v r b) :
    api.byte (copyMemory api M v scratch count m r) b=api.byte (m r) b :=
  VeriTile.Bench.Repairs.UniformSampling.Shared.copyPrefix_frame api count n v.region v.address _ m r b h

theorem byte_frame (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (seed start : Nat) (scratch : Scratch n) (v : View n) (m : Nat→Store) (r b : Nat)
    (fresh:r≠scratch.region) (outside:Outside v r b) :
    api.byte (run api M scalars seed start scratch v m r) b=api.byte (m r) b := by
  unfold run
  rw [copy_prefix_frame api M v scratch n _ r b outside,
    native_prefix_frame api M scalars seed start scratch _ m r b fresh]

#axiomsClean physical_readback
#axiomsClean logical_readback
#axiomsClean native_prefix_frame
#axiomsClean copy_prefix_frame
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Execution
