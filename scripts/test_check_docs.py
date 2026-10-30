"""Repository documentation checks must distinguish prose links from examples."""
from pathlib import Path
import tempfile
import unittest

from check_docs import check_links, links


class DocumentationLinks(unittest.TestCase):
    def test_code_fences_and_spans_do_not_create_links(self):
        text = '''````md
[example](absent.md)
```
[still example](absent.md)
````
`[inline](absent.md)` [real](present.md)
[source]: <source file.md>
'''
        self.assertEqual(list(links(text)), [(6, 'present.md'), (7, 'source file.md')])

    def test_encoded_paths_reference_links_and_parentheses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'source file.md').touch()
            (root / 'source(v2).md').touch()
            document = root / 'README.md'
            document.write_text('[file](source%20file.md#details) [version](source(v2).md)\n'
                                '[reference]: <source file.md>\n'
                                '[web](https://example.com/absent) [route](/VeriTile/) [anchor](#title)\n')
            self.assertEqual(check_links(root, [document]), [])

    def test_missing_files_and_links_outside_repository_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repo'
            root.mkdir()
            (root.parent / 'outside.md').touch()
            document = root / 'README.md'
            document.write_text('[missing](absent.md)\n[outside](../outside.md)\n')
            errors = check_links(root, [document])
            self.assertEqual(len(errors), 2)
            self.assertIn('README.md:1:', errors[0])
            self.assertIn('README.md:2:', errors[1])


if __name__ == '__main__':
    unittest.main()
