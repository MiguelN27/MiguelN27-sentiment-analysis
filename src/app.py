import argparse
import math
from pathlib import Path

import pandas as pd


MODEL_ID = 'nlptown/bert-base-multilingual-uncased-sentiment'
MODEL_REVISION = '8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d'
MAX_LENGTH = 512
LABEL_TO_STARS = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5}
STAR_TO_SENTIMENT = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_classifier():
	import torch
	from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

	torch.set_num_threads(4)
	tokenizer = AutoTokenizer.from_pretrained(
		MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, use_safetensors=True,
	)
	if set(model.config.id2label.values()) != set(LABEL_TO_STARS):
		raise ValueError('The pinned model has an unexpected label mapping.')
	return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=-1)


def stars_to_sentiment(stars: int) -> str:
	if isinstance(stars, bool) or stars not in STAR_TO_SENTIMENT:
		raise ValueError('Stars must be an integer between 1 and 5.')
	return STAR_TO_SENTIMENT[stars]


def validate_reviews(reviews: pd.DataFrame) -> None:
	required = ['review_id', 'rating', 'review_text']
	if not set(required).issubset(reviews.columns):
		raise ValueError(f'Required columns: {required}')
	if reviews.empty:
		raise ValueError('The input contains no reviews.')
	if reviews['review_id'].isna().any() or reviews['review_id'].duplicated().any():
		raise ValueError('Review IDs must be nonmissing and unique.')
	ratings = pd.to_numeric(reviews['rating'], errors='coerce')
	if not (ratings.between(1, 5) & ratings.mod(1).eq(0)).all():
		raise ValueError('Ratings must be integers between 1 and 5.')
	if not reviews['review_text'].map(
		lambda value: isinstance(value, str) and bool(value.strip())
	).all():
		raise ValueError('Review texts must be nonblank strings.')


def score_reviews(reviews: pd.DataFrame, classifier, batch_size: int = 16) -> pd.DataFrame:
	validate_reviews(reviews)
	if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
		raise ValueError('Batch size must be a positive integer.')
	texts = reviews['review_text'].tolist()
	tokenized = classifier.tokenizer(texts, truncation=False, add_special_tokens=True)
	token_lengths = [len(tokens) for tokens in tokenized['input_ids']]
	predictions = classifier(texts, batch_size=batch_size, truncation=True, max_length=MAX_LENGTH)
	if len(predictions) != len(reviews) or len(token_lengths) != len(reviews):
		raise ValueError('Prediction or tokenizer count does not match the input.')

	predicted_stars = []
	scores = []
	for prediction in predictions:
		if prediction.get('label') not in LABEL_TO_STARS:
			raise ValueError(f'Unexpected prediction label: {prediction.get("label")}')
		score = float(prediction['score'])
		if not math.isfinite(score) or not 0 <= score <= 1:
			raise ValueError('Prediction scores must be finite and between 0 and 1.')
		predicted_stars.append(LABEL_TO_STARS[prediction['label']])
		scores.append(score)

	enriched = reviews.copy()
	enriched['predicted_stars'] = predicted_stars
	enriched['sentiment'] = [stars_to_sentiment(stars) for stars in predicted_stars]
	enriched['sentiment_score'] = scores
	numeric_ratings = pd.to_numeric(reviews['rating']).astype(int)
	enriched['human_sentiment'] = numeric_ratings.map(STAR_TO_SENTIMENT)
	enriched['star_gap'] = enriched['predicted_stars'] - numeric_ratings
	enriched['is_repeated_text'] = reviews['review_text'].duplicated(keep=False)
	enriched['was_truncated'] = [length > MAX_LENGTH for length in token_lengths]
	enriched['model_id'] = MODEL_ID
	enriched['model_revision'] = MODEL_REVISION
	return enriched


def main(argv=None) -> None:
	parser = argparse.ArgumentParser(description='Score review text with a pinned sentiment model.')
	parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data/raw/reviews.csv')
	parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/processed/reviews_with_sentiment.csv')
	parser.add_argument('--batch-size', type=int, default=16)
	arguments = parser.parse_args(argv)
	if arguments.input.resolve() == arguments.output.resolve():
		raise ValueError('Output must not overwrite the input CSV.')
	if arguments.batch_size < 1:
		raise ValueError('Batch size must be a positive integer.')
	reviews = pd.read_csv(arguments.input)
	validate_reviews(reviews)
	classifier = load_classifier()
	enriched = score_reviews(reviews, classifier, arguments.batch_size)
	arguments.output.parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(arguments.output, index=False, encoding='utf-8')
	print(f'Wrote {len(enriched)} reviews to {arguments.output}')
	print(enriched['sentiment'].value_counts().to_string())


if __name__ == '__main__':
	main()
