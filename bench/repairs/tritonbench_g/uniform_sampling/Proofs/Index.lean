import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Chunks
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Counter

namespace VeriTile.Bench.Repairs.UniformSampling.IndexProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling

abbrev Location (B : Nat) := Chunk × Nat × Fin 4 × Fin B

def globalIndex (x : Location B) : Nat := 4*x.1.done+outputIndex x.2.1 B x.2.2.1 x.2.2.2

def schedule (n start : Nat) : List (Location (block n)) :=
  (chunks n start).flatMap fun c=>
    (lanes (block n) (chunkPrograms (block n) c)).map fun pfi=>(c,pfi)

theorem lanes_mem {B programs p : Nat} {f : Fin 4} {i : Fin B} :
    (p,f,i)∈lanes B programs ↔ p<programs := by
  simp [lanes]

theorem schedule_mem {n start : Nat} {x : Location (block n)} :
    x∈schedule n start ↔ x.1∈chunks n start ∧ x.2.1<chunkPrograms (block n) x.1 := by
  rcases x with ⟨c,p,f,i⟩
  simp [schedule,lanes_mem]

theorem index_output {B : Nat} (hB:0<B) (p : Nat) (f : Fin 4) (i : Fin B) :
    Spec.index B (outputIndex p B f i)=p*B+i.val := by
  have hd:=SourceDomain.output_div hB p f i
  have hm:=SourceDomain.output_mod p B f i
  have hq:outputIndex p B f i/(4*B)=p := by
    rw [Nat.mul_comm 4 B,←Nat.div_div_eq_div_mul,hd]
    omega
  simp [Spec.index,hq,hm]

theorem field_output {B : Nat} (hB:0<B) (p : Nat) (f : Fin 4) (i : Fin B) :
    Spec.field B (outputIndex p B f i)=field f := by
  rw [Spec.field,SourceDomain.output_div hB]
  have h:(p*4+f.val)%4=f.val:=by omega
  rw [h]
  fin_cases f <;> rfl

theorem global_as_output {B : Nat} (c : Chunk) (p : Nat) (f : Fin 4) (i : Fin B)
    (hd:B∣c.done) : globalIndex (c,p,f,i)=outputIndex (c.done/B+p) B f i := by
  have h:=Nat.div_mul_cancel hd
  unfold globalIndex outputIndex SourceDomain.outputIndex
  dsimp
  nlinarith

theorem global_index {B : Nat} (hB:0<B) (c : Chunk) (p : Nat) (f : Fin 4) (i : Fin B)
    (hd:B∣c.done) :
    Spec.index B (globalIndex (c,p,f,i))=c.done+p*B+i.val ∧
    Spec.field B (globalIndex (c,p,f,i))=field f := by
  rw [global_as_output c p f i hd,index_output hB,field_output hB]
  have h:=Nat.div_mul_cancel hd
  constructor
  · nlinarith
  · rfl

theorem launch_bounds {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    x.2.1*block n+x.2.2.2.val<x.1.count ∧
    outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<4*x.1.count ∧
    (start+x.1.done)%2^32+(x.2.1*block n+x.2.2.2.val)<2^32 := by
  obtain ⟨hc,hp⟩:=schedule_mem.mp hx
  have hb:=ReservationProof.block_properties n
  have h:=ChunkProof.chunks_properties n start hs hc
  have hmul:=Nat.mul_le_mul_right (block n) (Nat.succ_le_of_lt hp)
  have he:=Nat.div_mul_cancel h.2.2.2.2
  have hi:=x.2.2.2.isLt
  have hf:=x.2.2.1.isLt
  unfold chunkPrograms at hp
  unfold chunkPrograms at hmul
  unfold outputIndex SourceDomain.outputIndex
  constructor
  · nlinarith
  constructor <;> nlinarith

theorem mask_global {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<chunkSize n x.1 ↔ globalIndex x<n := by
  have h:=launch_bounds hs hx
  unfold chunkSize globalIndex
  rw [lt_min_iff]
  omega

theorem coverage {n start i : Nat} (hi:i<n) (hs:block n∣start) :
    ∃ x,x∈schedule n start ∧ globalIndex x=i ∧
      outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<chunkSize n x.1 := by
  have hb:=ReservationProof.block_properties n
  have ht:=ChunkProof.total_covers n
  obtain ⟨c,hc,hlo,hhi⟩:=ChunkProof.cover start (counterCount n) 0 (i/4) (by omega) (by omega)
  have hm:=ChunkProof.chunks_properties n start hs hc
  have hlocal:i-4*c.done<chunkSize n c:=by unfold chunkSize;omega
  obtain ⟨pfi,hout,_⟩:=SourceDomain.active_coverage_unique hb.2.1 hlocal
  have hgrid:grid (chunkSize n c) (block n)≤c.count/block n := by
    apply (ceilDiv_le_iff_le_mul (by omega : 0<4*block n)).mpr
    have hd:=Nat.div_mul_cancel hm.2.2.2.2
    have hcsize:chunkSize n c≤4*c.count:=Nat.min_le_right _ _
    nlinarith
  let x : Location (block n):=(c,pfi.1.val,pfi.2.1,pfi.2.2)
  have hx:x∈schedule n start:=schedule_mem.mpr ⟨hc,lt_of_lt_of_le pfi.1.isLt hgrid⟩
  have he:globalIndex x=i:=by
    change 4*c.done+outputIndex pfi.1.val (block n) pfi.2.1 pfi.2.2=i
    change 4*c.done+SourceDomain.outputIndex pfi.1.val (block n) pfi.2.1 pfi.2.2=i
    rw [hout]
    omega
  exact ⟨x,hx,he,by simpa [x,outputIndex,hout] using hlocal⟩

theorem word_binding {n start seed : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    sourceWord (signed64 seed) (signed64 (start+x.1.done)) x.2.1 (block n) x.2.2.1 x.2.2.2=
      Spec.word seed start (block n) (globalIndex x) := by
  have hb:=ReservationProof.block_properties n
  have hm:=ChunkProof.chunks_properties n start hs (schedule_mem.mp hx).1
  have hi:=global_index hb.2.1 x.1 x.2.1 x.2.2.1 x.2.2.2 hm.2.2.2.1
  have hl:=launch_bounds hs hx
  unfold sourceWord ScalarSpec.word Spec.word
  rw [CounterProof.words hl.2.2,hi.1,hi.2]
  congr 2
  omega

#axiomsClean coverage
#axiomsClean launch_bounds
#axiomsClean word_binding
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.UniformSampling.IndexProof
