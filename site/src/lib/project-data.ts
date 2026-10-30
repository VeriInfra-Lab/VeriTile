import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import demoRecord from './vector-add-record.json';

// The documented dev script and CI build both run from site/.
const repositoryRoot = resolve(process.cwd(), '..');
const benchRoot = resolve(repositoryRoot, 'bench/tritonbench_g');
const ports = readdirSync(benchRoot, { withFileTypes: true })
  // Private/generated directories (for example _spec-sheets) are metadata.
  .filter((entry) => entry.isDirectory() && !entry.name.startsWith('_') && !entry.name.startsWith('.'))
  .map((entry) => {
    const files = readdirSync(resolve(benchRoot, entry.name));
    return {
      python: files.some((file) => file.endsWith('.py')),
      lean: files.some((file) => file.endsWith('.lean')),
      readme: files.includes('README.md'),
    };
  })
  .filter((entry) => entry.python || entry.lean || entry.readme);

export const corpus = {
  pairedPorts: ports.filter((port) => port.python && port.lean).length,
  scaffolds: ports.filter((port) => !port.python && !port.lean).length,
  totalDirectories: ports.length,
};

const vectorAddSource = readFileSync(resolve(repositoryRoot, 'bench/examples/VectorAdd/Correct.lean'), 'utf8')
  + '\n' + readFileSync(resolve(repositoryRoot, 'bench/examples/VectorAdd/Kernels.lean'), 'utf8');
const kernelMatch = vectorAddSource.match(/def addKernel [\s\S]*?\n  (triton \{[\s\S]*?\n\})/);
const contractMatch = vectorAddSource.match(/specification add_kernel_correctness[^\n]*\n\s*(.+) := by/);
const proofMatch = vectorAddSource.match(/(specification add_kernel_correctness[^\n]*\n[\s\S]*?)(?=\n\nsection Optimized)/);
if (!kernelMatch || !contractMatch || !proofMatch) {
  throw new Error('The VectorAdd showcase changed. Update the homepage excerpt selectors.');
}

export const vectorAddKernel = kernelMatch[1];
export const vectorAddContract = contractMatch[1];
export const vectorAddProof = proofMatch[1];

const pythonSource = readFileSync(resolve(repositoryRoot, 'bench/examples/VectorAdd/native_kernels.py'), 'utf8');
const pythonMatch = pythonSource.match(/(@triton\.jit\ndef addKernel[\s\S]*?)(?=\n\n@triton\.jit)/);
if (!pythonMatch) throw new Error('Homepage Python excerpt changed. Update its selector.');
// Both excerpts retain their actual source identifiers and operations.
function comparisonBody(lines: string[]): string {
  return lines
    .filter((line) => !line.trim().startsWith('#'))
    .map((line) => line.trim()
      .replace(/^(\w+)\s*(:?=)\s*/, (_, name: string, operator: string) => `${name.padEnd(7)} ${operator} `))
    .join('\n');
}
export const vectorAddBodies = {
  python: comparisonBody(pythonMatch[1].trimEnd().split('\n').slice(2)),
  lean: comparisonBody(vectorAddKernel.split('\n').slice(1, -1)),
};

const proofRows = readFileSync(resolve(benchRoot, 'proof_gap_manifest.tsv'), 'utf8').trim().split('\n');
const proofColumns = proofRows.shift()!.split('\t');
const declarationColumn = proofColumns.indexOf('declaration');
const fileColumn = proofColumns.indexOf('file');
if (declarationColumn < 0 || fileColumn < 0) throw new Error('Proof manifest schema changed.');
export const proofInventory = {
  statements: new Set(proofRows.map((row) => {
    const columns = row.split('\t');
    return `${columns[fileColumn]}:${columns[declarationColumn]}`;
  })).size,
};

// Fail the site build if recorded Lean results no longer match this checkout.
function leanFiles(directory: string): string[] {
  return readdirSync(resolve(repositoryRoot, directory), { withFileTypes: true }).flatMap((entry) => {
    const path = `${directory}/${entry.name}`;
    return entry.isDirectory() ? leanFiles(path) : entry.name.endsWith('.lean') ? [path] : [];
  });
}
const recordInputs = [
  'lean-toolchain', 'lakefile.toml', 'lake-manifest.json',
  'site/scripts/record-home-demo.py', 'bench/examples/VectorAdd/Correct.lean',
  'bench/examples/VectorAdd/Kernels.lean',
  'bench/examples/VectorAdd/Proofs/Real.lean',
  ...leanFiles('VeriTile'),
].sort();
const fingerprint = createHash('sha256');
for (const path of recordInputs) {
  fingerprint.update(path + '\0');
  fingerprint.update(readFileSync(resolve(repositoryRoot, path)));
  fingerprint.update('\0');
}
if (fingerprint.digest('hex') !== demoRecord.fingerprint || demoRecord.variants[0].kernel !== vectorAddKernel) {
  throw new Error('Homepage Lean records are stale. Run lake build, then python3 site/scripts/record-home-demo.py.');
}
export const vectorAddDemo = demoRecord;
export const leanToolchain = readFileSync(resolve(repositoryRoot, 'lean-toolchain'), 'utf8').trim().split(':').at(-1);
export const quickStart = `git clone https://github.com/VeriInfra-Lab/VeriTile.git
cd VeriTile
lake build
lake env lean bench/examples/VectorAdd/Correct.lean`;
