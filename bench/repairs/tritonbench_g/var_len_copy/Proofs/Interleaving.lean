import bench.repairs.tritonbench_g.var_len_copy.Proofs.Execution

namespace VeriTile.Bench.Repairs.VarLenCopy
noncomputable section
set_option maxHeartbeats 3000000

/-- Arbitrary interleaving of the complete native copy events. The fresh source
is immutable and all writes to one destination have the same initial source,
so cross-program/lane scheduling cannot change any physical memory cell. -/
theorem interleaving_equal (l : Layout) (r : Request) (s : Memory)
    (prep : Prepared l r s) (valid : Valid r) (xs : List Event)
    (permutation : xs.Perm (allEvents l r (List.range l.count))) :
    execute xs s=launch l s := by
  have mem := allEvents_mem l r s prep valid (List.range l.count)
    (by intro p hp;simpa [prep.counts] using hp)
  have xmem : ∀ e,e∈xs ↔ ∃ p i,p<r.count ∧ i<(r.segment p).len ∧
      e=(addr l.output ((r.segment p).dest+i),addr l.source ((r.segment p).old+i)) := by
    intro e
    rw [permutation.mem_iff,mem e]
    simp only [List.mem_range,prep.counts]
  have separate : ∀ e∈xs,∀ f∈xs,e.1≠f.2 := by
    intro e he f hf
    obtain ⟨p,i,hp,hi,rfl⟩ := (xmem e).mp he
    obtain ⟨q,j,hq,hj,rfl⟩ := (xmem f).mp hf
    have ho := (valid.outputBound p hp (by omega)).2
    have hs := (valid.sourceBound q hq (by omega)).2
    exact (prep.disjoint _ (by rw [prep.outputSize];omega)).1 _ (by rw [prep.sourceSize];omega)
  obtain ⟨values,frame⟩ := launch_correct l r s prep valid
  funext cell
  by_cases touched : ∃ x,Touched r x ∧ cell=addr l.output x
  · obtain ⟨x,⟨p,i,hp,hi,rfl⟩,rfl⟩ := touched
    have value : launch l s (addr l.output ((r.segment p).dest+i))=
        s (addr l.source ((r.segment p).old+i)) := values.1 p i hp hi
    rw [value,execute_snapshot xs s s (fun _ _=>rfl) separate]
    apply snapshot_read
    · intro e he heq
      obtain ⟨q,j,hq,hj,rfl⟩ := (xmem e).mp he
      have same : (r.segment q).dest+j=(r.segment p).dest+i := by
        change addr l.output ((r.segment q).dest+j)=addr l.output ((r.segment p).dest+i) at heq
        unfold addr at heq;omega
      obtain ⟨rfl,rfl⟩ := valid.unique q p j i hq hp hj hi same
      rfl
    · apply Or.inl
      exact ⟨_,(xmem _).mpr ⟨p,i,hp,hi,rfl⟩,rfl⟩
  · rw [frame cell touched]
    apply execute_frame
    intro e he same
    obtain ⟨p,i,hp,hi,rfl⟩ := (xmem e).mp he
    exact touched ⟨(r.segment p).dest+i,⟨p,i,hp,hi,rfl⟩,same.symm⟩

#axiomsClean interleaving_equal
end
end VeriTile.Bench.Repairs.VarLenCopy
