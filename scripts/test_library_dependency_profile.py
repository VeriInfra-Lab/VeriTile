"""Keep the library-only dependency view synchronized with the public package."""
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LibraryDependencyProfileTests(unittest.TestCase):
    def test_profile_retains_library_configuration_and_omits_only_examples(self):
        source = tomllib.loads((ROOT / 'lakefile.toml').read_text())
        actual = tomllib.loads((ROOT / 'library/lakefile.toml').read_text())
        expected = {**source, 'srcDir': '..', 'buildDir': '../.lake/build',
                    'manifestFile': '../lake-manifest.json'}
        expected['lean_lib'] = [row for row in source['lean_lib']
                                if row['name'] != 'TritonBenchSpecExamples']
        self.assertEqual(actual, expected)
        self.assertEqual(len(source['lean_lib']) - len(actual['lean_lib']), 1)
        for row in actual['lean_lib']:
            self.assertFalse(any(glob.startswith('bench.examples.') for glob in row.get('globs', [])))


if __name__ == '__main__':
    unittest.main()
