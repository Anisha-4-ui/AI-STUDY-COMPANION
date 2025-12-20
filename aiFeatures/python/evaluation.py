# enhanced_evaluation.py
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from evaluation_dataset import test_dataset

# Initialize semantic similarity model
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(" Loaded semantic similarity model")
except:
    semantic_model = None
    print("⚠  Semantic model not available - install sentence-transformers")

def compute_recall_at_k(retrieved_chunks, relevant_chunks, k=3):
    """Compute Recall@K for retrieval evaluation"""
    if not retrieved_chunks or not relevant_chunks:
        return 0.0
    
    # Take top-k retrieved chunks
    top_k_retrieved = retrieved_chunks[:k]
    
    # Count how many relevant chunks are in top-k
    matches = 0
    for rel_chunk in relevant_chunks:
        for ret_chunk in top_k_retrieved:
            if rel_chunk.lower() in ret_chunk.lower() or ret_chunk.lower() in rel_chunk.lower():
                matches += 1
                break
    
    recall_at_k = matches / len(relevant_chunks)
    return recall_at_k

def compute_mrr(retrieved_chunks, relevant_chunks):
    """Compute Mean Reciprocal Rank for retrieval evaluation"""
    if not retrieved_chunks or not relevant_chunks:
        return 0.0
    
    reciprocal_ranks = []
    
    for rel_chunk in relevant_chunks:
        rank = None
        for i, ret_chunk in enumerate(retrieved_chunks):
            if rel_chunk.lower() in ret_chunk.lower() or ret_chunk.lower() in rel_chunk.lower():
                rank = i + 1
                break
        
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)

def compute_semantic_similarity(generated_answer, reference_answer):
    """Compute semantic similarity using sentence embeddings"""
    if not semantic_model:
        return 0.0
    
    try:
        # Generate embeddings
        gen_embedding = semantic_model.encode([generated_answer])
        ref_embedding = semantic_model.encode([reference_answer])
        
        # Compute cosine similarity
        similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error computing semantic similarity: {e}")
        return 0.0

def compute_rouge(generated_answer, reference_answer):
    """Compute ROUGE scores for text generation evaluation"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_answer, generated_answer)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def analyze_educational_quality(generated_answer):
    """Analyze educational quality beyond ROUGE scores"""
    analysis = {}
    
    # Check for educational structure
    analysis['has_examples'] = 'example' in generated_answer.lower() or 'for instance' in generated_answer.lower()
    analysis['has_structure'] = bool(re.search(r'[*•\-]\s', generated_answer)) or bool(re.search(r'\d+\.', generated_answer))
    analysis['has_definitions'] = 'is a' in generated_answer.lower() or 'means' in generated_answer.lower()
    analysis['interactive'] = '?' in generated_answer
    analysis['comprehensive'] = len(generated_answer.split()) > 50
    
    # Count key concepts coverage
    key_concepts = ['data', 'algorithm', 'model', 'learning', 'pattern', 'prediction']
    concepts_covered = sum(1 for concept in key_concepts if concept in generated_answer.lower())
    analysis['concept_coverage'] = concepts_covered / len(key_concepts)
    
    return analysis

def find_matching_query(user_query):
    """Find matching query from test dataset using flexible matching"""
    user_query_lower = user_query.lower().strip()
    
    # Direct keyword matching
    keywords_map = {
        "machine learning": ["machine learning", "ml", "learning algorithm"],
        "neural network": ["neural", "network", "deep learning"],
        "python": ["python", "programming language"],
        "algorithm": ["algorithm", "procedure", "step"],
        "data science": ["data science", "data analysis", "analytics"]
    }
    
    # Check for keyword matches
    for query_type, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in user_query_lower:
                # Find the corresponding dataset entry
                for data in test_dataset:
                    if query_type in data['query'].lower():
                        return data
    
    # Fallback: partial string matching
    for data in test_dataset:
        query_words = set(user_query_lower.split())
        dataset_words = set(data['query'].lower().split())
        
        # If at least 30% of words match
        if len(query_words & dataset_words) >= len(query_words) * 0.3:
            return data
    
    return None

def evaluate_response(user_query, generated_answer, retrieved_chunks=None):
    """Enhanced evaluation with semantic metrics and RAG-specific metrics"""
    # Find matching ground truth
    ground_truth = find_matching_query(user_query)
    
    if not ground_truth:
        print(" EVALUATION: No ground truth found for this query")
        print(f"   Query: '{user_query}'")
        print("   Available topics: machine learning, neural networks, python, algorithms, data science")
        return
    
    print("\n" + "="*60)
    print(" ENHANCED EVALUATION RESULTS")
    print(f" Query: {user_query}")
    print(f" Matched: {ground_truth['query']}")
    print("="*60)
    
    # RAG RETRIEVAL EVALUATION
    if retrieved_chunks and ground_truth.get('relevant_chunks'):
        print(f" RETRIEVAL EVALUATION:")
        print(f"   Retrieved: {len(retrieved_chunks)} chunks")
        print(f"   Expected: {len(ground_truth['relevant_chunks'])} chunks")
        
        # Recall@K
        recall_at_3 = compute_recall_at_k(retrieved_chunks, ground_truth['relevant_chunks'], k=3)
        recall_at_5 = compute_recall_at_k(retrieved_chunks, ground_truth['relevant_chunks'], k=5)
        
        # MRR
        mrr_score = compute_mrr(retrieved_chunks, ground_truth['relevant_chunks'])
        
        print(f"   📍 Recall@3: {recall_at_3:.3f}")
        print(f"   📍 Recall@5: {recall_at_5:.3f}")
        print(f"   📍 MRR: {mrr_score:.3f}")
        
        if recall_at_3 >= 0.6:
            print("    Excellent retrieval")
        elif recall_at_3 >= 0.3:
            print("   ⚠  Good retrieval")
        else:
            print("    Poor retrieval")
    
    # TEXT GENERATION EVALUATION
    if generated_answer and ground_truth.get('reference_answer'):
        print(f"\n GENERATION EVALUATION:")
        
        # ROUGE Scores (fluency)
        rouge_scores = compute_rouge(generated_answer, ground_truth['reference_answer'])
        print(f"    ROUGE-1: {rouge_scores['rouge1']:.3f}")
        print(f"    ROUGE-2: {rouge_scores['rouge2']:.3f}")
        print(f"    ROUGE-L: {rouge_scores['rougeL']:.3f}")
        avg_rouge = sum(rouge_scores.values()) / len(rouge_scores)
        
        # Semantic Similarity (meaning)
        semantic_score = compute_semantic_similarity(generated_answer, ground_truth['reference_answer'])
        print(f"    Semantic Similarity: {semantic_score:.3f}")
        
        # Educational Quality Analysis
        edu_analysis = analyze_educational_quality(generated_answer)
        print(f"\n EDUCATIONAL QUALITY:")
        print(f"     Examples: {'✅' if edu_analysis['has_examples'] else '❌'}")
        print(f"    Structure: {'✅' if edu_analysis['has_structure'] else '❌'}")
        print(f"    Definitions: {'✅' if edu_analysis['has_definitions'] else '❌'}")
        print(f"    Interactive: {'✅' if edu_analysis['interactive'] else '❌'}")
        print(f"    Comprehensive: {'✅' if edu_analysis['comprehensive'] else '❌'}")
        print(f"    Concept Coverage: {edu_analysis['concept_coverage']:.1%}")
        
        # HYBRID SCORE (combines ROUGE + Semantic + Educational)
        edu_score = sum([
            edu_analysis['has_examples'],
            edu_analysis['has_structure'], 
            edu_analysis['has_definitions'],
            edu_analysis['interactive'],
            edu_analysis['comprehensive']
        ]) / 5.0
        
        hybrid_score = (avg_rouge * 0.3) + (semantic_score * 0.4) + (edu_score * 0.3)
        
        print(f"\n HYBRID SCORE: {hybrid_score:.3f}")
        print(f"   (ROUGE: {avg_rouge:.3f} | Semantic: {semantic_score:.3f} | Educational: {edu_score:.3f})")
        
        if hybrid_score >= 0.7:
            print("    EXCELLENT educational response!")
        elif hybrid_score >= 0.5:
            print("    GOOD educational response")
        elif hybrid_score >= 0.3:
            print("   ⚠  FAIR educational response")
        else:
            print("    POOR educational response")
    
    print("="*60 + "\n")