import tempfile
import unittest
from pathlib import Path
from main import prepare, build_parser


class ConfigurationTests(unittest.TestCase):
    def test_missing_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare(Path(tmp), Path(tmp) / 'local.yaml')

    def test_paths_and_class_order(self):
        with tempfile.TemporaryDirectory(prefix='fire data ') as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                for folder in ('images', 'labels'):
                    (root / split / folder).mkdir(parents=True)
            out = prepare(root, root / 'config' / 'local.yaml')
            text = out.read_text(encoding='utf-8')
            self.assertIn('0: smoke\n  1: fire', text)
            self.assertIn(root.as_posix(), text)

    def test_validation_does_not_default_to_test_split(self):
        args = build_parser().parse_args(['val', '--weights', 'best.pt'])
        self.assertEqual(args.split, 'val')


if __name__ == '__main__':
    unittest.main()
