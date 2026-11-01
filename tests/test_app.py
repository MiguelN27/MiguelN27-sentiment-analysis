import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src import app


class FakeClassifier:
    def __init__(self, predictions=None):
        self.predictions = predictions
        self.calls = []

    def tokenizer(self, texts, **kwargs):
        return {'input_ids': [[0] * (len(text.split()) + 2) for text in texts]}

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if self.predictions is not None:
            return self.predictions
        return [{'label': '5 stars', 'score': 0.8} for text in texts]


class SentimentTests(unittest.TestCase):
    def setUp(self):
        self.reviews = pd.DataFrame({
            'review_id': [11, 12, 13],
            'rating': [5, 4, 2],
            'review_text': ['Great food!', 'Great food!', 'Slow service.'],
        }, index=[8, 3, 6])

    def test_all_labels_and_bands(self):
        self.assertEqual([app.stars_to_sentiment(stars) for stars in range(1, 6)],
                         ['negative', 'negative', 'neutral', 'positive', 'positive'])
        reviews = pd.DataFrame({'review_id': range(5), 'rating': [3] * 5,
                                'review_text': ['Example'] * 5})
        predictions = [{'label': label, 'score': 0.5} for label in app.LABEL_TO_STARS]
        self.assertEqual(app.score_reviews(reviews, FakeClassifier(predictions)).predicted_stars.tolist(),
                         [1, 2, 3, 4, 5])

    def test_invalid_stars(self):
        for value in [0, 6, 2.5, True, '5']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.stars_to_sentiment(value)

    def test_preserves_order_index_raw_columns_and_duplicate_texts(self):
        classifier = FakeClassifier()
        result = app.score_reviews(self.reviews, classifier, batch_size=2)
        pd.testing.assert_frame_equal(result[self.reviews.columns], self.reviews)
        self.assertEqual(result.star_gap.tolist(), [0, 1, 3])
        self.assertEqual(result.is_repeated_text.tolist(), [True, True, False])
        self.assertEqual(result.model_revision.unique().tolist(), [app.MODEL_REVISION])
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(classifier.calls[0][1],
                         {'batch_size': 2, 'truncation': True, 'max_length': 512})

    def test_truncation_flag(self):
        reviews = self.reviews.copy()
        reviews.loc[8, 'review_text'] = 'word ' * 511
        self.assertEqual(app.score_reviews(reviews, FakeClassifier()).was_truncated.tolist(),
                         [True, False, False])

    def test_invalid_inputs_fail_before_classifier(self):
        cases = [self.reviews.iloc[:0], self.reviews.drop(columns='rating')]
        for column, value in [('rating', 6), ('rating', 2.5), ('rating', None),
                              ('review_text', '   '), ('review_text', None),
                              ('review_text', 123), ('review_id', None), ('review_id', 12)]:
            invalid = self.reviews.copy()
            invalid[column] = invalid[column].astype(object)
            invalid.loc[8, column] = value
            cases.append(invalid)
        for invalid in cases:
            classifier = Mock()
            with self.assertRaises(ValueError):
                app.score_reviews(invalid, classifier)
            classifier.assert_not_called()

    def test_invalid_batch_sizes(self):
        for size in [0, -1, True, 1.5]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                app.score_reviews(self.reviews, FakeClassifier(), size)

    def test_unexpected_prediction_label_or_count(self):
        for predictions in [[], [{'label': 'POSITIVE', 'score': 0.8}] * 3]:
            with self.assertRaises(ValueError):
                app.score_reviews(self.reviews, FakeClassifier(predictions))

    def test_invalid_scores(self):
        for score in [-0.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                app.score_reviews(self.reviews, FakeClassifier([{'label': '5 stars', 'score': score}] * 3))

    def test_import_does_not_load_model_or_database(self):
        process = subprocess.run(
            [sys.executable, '-c',
             "import sys; import src.app; assert 'transformers' not in sys.modules; "
             "assert 'torch' not in sys.modules; assert 'src.utils' not in sys.modules; "
             "assert 'utils' not in sys.modules"],
            cwd=app.PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_main_loads_once_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / 'reviews.csv'
            output_path = Path(directory) / 'nested/scored.csv'
            self.reviews.to_csv(input_path, index=False)
            with patch.object(app, 'load_classifier', return_value=FakeClassifier()) as loader:
                app.main(['--input', str(input_path), '--output', str(output_path), '--batch-size', '2'])
                loader.assert_called_once_with()
            result = pd.read_csv(output_path)
            self.assertEqual(len(result), 3)
            self.assertEqual(result.review_id.tolist(), [11, 12, 13])

    def test_no_partial_output_for_failed_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / 'reviews.csv'
            output_path = Path(directory) / 'scored.csv'
            self.reviews.to_csv(input_path, index=False)
            with patch.object(app, 'load_classifier', return_value=FakeClassifier([])):
                with self.assertRaises(ValueError):
                    app.main(['--input', str(input_path), '--output', str(output_path)])
            self.assertFalse(output_path.exists())

    def test_defaults_are_project_relative_and_input_is_protected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'PROJECT_ROOT', Path(directory)):
            input_path = Path(directory) / 'data/raw/reviews.csv'
            input_path.parent.mkdir(parents=True)
            self.reviews.to_csv(input_path, index=False)
            with patch.object(app, 'load_classifier', return_value=FakeClassifier()):
                app.main([])
            self.assertTrue((Path(directory) / 'data/processed/reviews_with_sentiment.csv').is_file())
            with self.assertRaises(ValueError):
                app.main(['--input', str(input_path), '--output', str(input_path)])


if __name__ == '__main__':
    unittest.main()