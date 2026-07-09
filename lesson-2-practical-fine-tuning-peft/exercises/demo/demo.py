import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from npcpy.npc_compiler import NPC

def prepare_movie_review_data():
    # Using 20newsgroups as proxy for review data
    categories = ['rec.autos', 'rec.motorcycles']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    reviews = []
    for text, label in zip(newsgroups.data[:100], newsgroups.target[:100]):
        sentiment = 'positive' if len(text.split()) > 50 else 'negative'  # Simple heuristic
        reviews.append({'text': text[:200], 'sentiment': sentiment})
    
    return pd.DataFrame(reviews)

def movie_review_lora_demo():
    df = prepare_movie_review_data()
    
    trainer_npc = NPC(
        name='LoRA Trainer',
        primary_directive='Configure LoRA parameters for sentiment analysis fine-tuning',
        model='llama3.2',
        provider='ollama'
    )
    
    config_format = '''
    {"lora_config": {
        "r": 16,
        "lora_alpha": 32, 
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "epochs": 3
    }}
    '''
    
    prompt = f"""Configure LoRA parameters for sentiment analysis on movie reviews.
Dataset size: {len(df)} samples
Task: Binary sentiment classification

Recommend optimal LoRA hyperparameters.
Format as: {config_format}"""
    
    response = trainer_npc.get_llm_response(prompt, format='json')
    config = response['response']['lora_config']
    
    # Save training data and config
    df.to_csv('movie_reviews_training.csv', index=False)
    config_df = pd.DataFrame([config])
    config_df.to_csv('lora_config.csv', index=False)
    
    print("LoRA configuration and training data prepared")
    print(f"Config: {config}")
    return df, config

movie_data, lora_config = movie_review_lora_demo()