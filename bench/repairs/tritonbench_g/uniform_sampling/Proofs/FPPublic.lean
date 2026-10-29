import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Numerics
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Protocol

namespace VeriTile.Bench.Repairs.UniformSampling.FPEquiv
open VeriTile.Triton Numerics
noncomputable section

/-- Complete raw FP32 source readback, for all scalar and stored words, with no
numerical rewrite premises. Integer Philox seed/counters, all four streams,
logical writeback, output identity and generator state are explicit. -/
theorem returned (p : Primitives) (scalars : Scalars) (v : View n) (scratch : Scratch n)
    (which : Nat) (lib : Host.Runtime) (stops : Host.Stops) (s : Host.State Store) (out : Nat) (hn:0<n)
    (ok:((Host.entry api (model p) scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      out=v.handle ∧
      (∀ i,i<n→logical api (model p)
        ((Host.entry api (model p) scalars v scratch which lib stops).run s).state.memory v i=
          Native.result (model p) scalars r.seed r.start (block n) i) ∧
      ((Host.entry api (model p) scalars v scratch which lib stops).run s).state.generators=
        Function.update s.generators which g ∧
      ((Host.entry api (model p) scalars v scratch which lib stops).run s).state.device=s.device :=
  Host.returned api (model p) scalars v scratch which lib stops s out hn (negate_twice p) ok

/-- Concrete four-byte readback includes lazy-negative output encoding; no
memory-frame or encoding obligation is deferred to numerical primitives. -/
theorem output_bytes (p : Primitives) (scalars : Scalars) (v : View n) (scratch : Scratch n)
    (which : Nat) (lib : Host.Runtime) (stops : Host.Stops) (s : Host.State Store) (out : Nat) (hn:0<n)
    (ok:((Host.entry api (model p) scalars v scratch which lib stops).run s).result=.ok out) :
    ∃ r g,reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g) ∧
      ∀ i,i<n→∀ byte : Fin 4,
        ((Host.entry api (model p) scalars v scratch which lib stops).run s).state.memory v.region
          (v.address i+byte.val)=VeriTile.Bench.Repairs.UniformSampling.Shared.Bytes.encode
            (physical (model p) v (Native.result (model p) scalars r.seed r.start (block n) i)) byte := by
  obtain ⟨r,g,hr,_,hstate⟩:=Host.entry_success api (model p) scalars v scratch which lib stops s out hn ok
  have hstart: r.start=alignedStart (s.generators which).offset (block n):=
    (ReservationProof.success hr).2.2.2.2.2.1
  have hs:block n∣r.start:=by rw [hstart];exact ReservationProof.alignment _ _ (ReservationProof.block_properties n).2.1
  refine ⟨r,g,hr,?_⟩
  intro i hi byte
  rw [hstate]
  have hread:=Execution.physical_readback api (model p) scalars r.seed r.start scratch v s.memory hs hi
  have hbyte:=VeriTile.Bench.Repairs.UniformSampling.Shared.Bytes.byte_of_read
    (Execution.run api (model p) scalars r.seed r.start scratch v s.memory v.region) (v.address i) byte
  rw [hread] at hbyte
  exact hbyte

theorem real_correspondence (p : Primitives) (scalars : Scalars) (seed start B i : Nat)
    (lower upper : ℝ)
    (ports:LocalExact p scalars (Spec.word seed start B i) lower upper) :
    project32 (Native.result (model p) scalars seed start B i)=
      some (Spec.target lower upper seed start B i) :=
  local_correspondence p scalars (Spec.word seed start B i) lower upper ports

#axiomsClean returned
#axiomsClean output_bytes
#axiomsClean real_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.FPEquiv
