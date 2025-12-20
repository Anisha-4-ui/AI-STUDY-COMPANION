import google.generativeai as genai
import json
import re
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def generate_quiz_from_content(content: str, num_questions: int = 10, vector_store=None) -> List[Dict[str, Any]]:
    """
    Generate a quiz from the provided content using Gemini AI
    
    Args:
        content (str): The content to generate quiz from (can be empty if vector_store provided)
        num_questions (int): Number of questions to generate (default: 10)
        vector_store: Optional vector store to retrieve comprehensive content from
    
    Returns:
        List[Dict]: List of quiz questions with options and correct answers
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # If vector_store is provided, retrieve more comprehensive content
        if vector_store:
            try:
                # Retrieve much more content for quiz generation
                from aiFeatures.python.rag_pipeline import retrieve_answer
                quiz_queries = [
                    "Retrieve all key concepts, definitions, and important topics",
                    "Retrieve all theories, principles, and methodologies discussed",
                    "Retrieve all examples, case studies, and applications mentioned"
                ]
                
                retrieved_content = []
                for query in quiz_queries:
                    chunk = retrieve_answer(query, vector_store, k=20)  # Get 20 chunks per query
                    if chunk:
                        retrieved_content.append(chunk)
                
                if retrieved_content:
                    content = "\n\n".join(retrieved_content)
            except Exception as e:
                print(f"Warning: Could not retrieve from vector store: {e}")
        
        # Use much more content - Gemini 2.0 Flash can handle ~1M tokens
        # Approximately 4 characters per token, so use up to 500k characters
        content_to_use = content[:500000] if len(content) > 500000 else content
        
        prompt = f"""
        You are creating a quiz based STRICTLY AND ONLY on the study material provided below.
        
        CRITICAL RULES:
        1. ALL questions MUST be answerable from the content provided
        2. DO NOT use your general knowledge - only use the specific material given
        3. Questions should directly reference concepts, facts, or ideas from this content
        4. If the content is insufficient, generate fewer questions rather than making up questions
        
        Create exactly {num_questions} multiple-choice questions based on the following study material.
        Each question should test understanding of key concepts from THIS specific material.
        
        Return the response as a valid JSON array with this exact structure:
        [
            {{
                "question": "Question text based on the content",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "explanation": "Explanation referencing the content"
            }}
        ]
        
        Requirements:
        - Generate exactly {num_questions} questions
        - Each question should have exactly 4 options
        - correct_answer should be the index (0-3) of the correct option
        - Questions should cover different aspects of the content
        - Make questions challenging but fair
        - Provide clear explanations that reference the source material
        - Return ONLY the JSON array, no additional text or markdown
        
        STUDY MATERIAL:
        {content_to_use}
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response to extract JSON
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'^```\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            quiz_data = json.loads(response_text)
            
            # Validate the structure
            if not isinstance(quiz_data, list):
                raise ValueError("Quiz data should be a list")
            
            validated_quiz = []
            for i, question in enumerate(quiz_data):
                if not isinstance(question, dict):
                    continue
                
                # Ensure required fields exist
                if all(key in question for key in ['question', 'options', 'correct_answer', 'explanation']):
                    # Validate options
                    if isinstance(question['options'], list) and len(question['options']) == 4:
                        # Validate correct_answer index
                        if isinstance(question['correct_answer'], int) and 0 <= question['correct_answer'] < 4:
                            validated_quiz.append(question)
                
                # Stop if we have enough questions
                if len(validated_quiz) >= num_questions:
                    break
            
            if len(validated_quiz) == 0:
                print("Warning: No valid questions generated. Using fallback quiz.")
                return generate_fallback_quiz(num_questions)
            
            if len(validated_quiz) < num_questions:
                print(f"Warning: Only generated {len(validated_quiz)} valid questions out of {num_questions} requested")
            
            return validated_quiz[:num_questions]  # Ensure exact number
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text preview: {response_text[:500]}")
            # Return fallback quiz
            return generate_fallback_quiz(num_questions)
            
    except Exception as e:
        print(f"Error in generate_quiz_from_content: {e}")
        return generate_fallback_quiz(num_questions)

def generate_fallback_quiz(num_questions: int = 10) -> List[Dict[str, Any]]:
    """Generate a fallback quiz when AI generation fails"""
    fallback_questions = [
        {
            "question": "What is the primary purpose of artificial intelligence?",
            "options": [
                "To replace human intelligence completely",
                "To augment human capabilities and solve complex problems",
                "To create conscious machines",
                "To automate all jobs"
            ],
            "correct_answer": 1,
            "explanation": "AI is designed to augment human capabilities and help solve complex problems, not to replace humans entirely."
        },
        {
            "question": "Which of the following is a key component of machine learning?",
            "options": [
                "Manual programming of all rules",
                "Data-driven pattern recognition",
                "Human supervision at all times",
                "Pre-written decision trees"
            ],
            "correct_answer": 1,
            "explanation": "Machine learning relies on data-driven pattern recognition to learn and make predictions."
        },
        {
            "question": "What does 'RAG' stand for in AI systems?",
            "options": [
                "Rapid Application Generation",
                "Retrieval Augmented Generation",
                "Random Access Gateway",
                "Recursive Algorithm Graph"
            ],
            "correct_answer": 1,
            "explanation": "RAG stands for Retrieval Augmented Generation, which combines retrieval of relevant information with text generation."
        },
        {
            "question": "What is the main advantage of vector databases in AI applications?",
            "options": [
                "They store data in traditional rows and columns",
                "They enable semantic similarity search",
                "They are faster than all other databases",
                "They require less storage space"
            ],
            "correct_answer": 1,
            "explanation": "Vector databases excel at semantic similarity search, allowing AI systems to find contextually relevant information."
        },
        {
            "question": "In natural language processing, what is tokenization?",
            "options": [
                "Converting text to numbers",
                "Breaking text into smaller units like words or subwords",
                "Encrypting sensitive information",
                "Translating between languages"
            ],
            "correct_answer": 1,
            "explanation": "Tokenization is the process of breaking text into smaller units (tokens) that can be processed by AI models."
        },
        {
            "question": "What is the purpose of fine-tuning in machine learning?",
            "options": [
                "To make models run faster",
                "To adapt pre-trained models to specific tasks",
                "To reduce model size",
                "To increase data storage"
            ],
            "correct_answer": 1,
            "explanation": "Fine-tuning adapts pre-trained models to perform better on specific tasks by training on domain-specific data."
        },
        {
            "question": "Which technique helps prevent overfitting in neural networks?",
            "options": [
                "Adding more layers",
                "Using dropout regularization",
                "Increasing learning rate",
                "Using more training data only"
            ],
            "correct_answer": 1,
            "explanation": "Dropout regularization helps prevent overfitting by randomly setting some neurons to zero during training."
        },
        {
            "question": "What is the primary function of an embedding in AI?",
            "options": [
                "To compress file sizes",
                "To represent data as dense vectors in high-dimensional space",
                "To encrypt sensitive information",
                "To speed up database queries"
            ],
            "correct_answer": 1,
            "explanation": "Embeddings represent data (like words or documents) as dense vectors that capture semantic relationships."
        },
        {
            "question": "In transformer architecture, what is the role of attention mechanisms?",
            "options": [
                "To reduce computational cost",
                "To focus on relevant parts of the input sequence",
                "To generate random variations",
                "To compress the input data"
            ],
            "correct_answer": 1,
            "explanation": "Attention mechanisms allow the model to focus on different parts of the input sequence when processing each element."
        },
        {
            "question": "What is prompt engineering in the context of large language models?",
            "options": [
                "Writing code to train models",
                "Designing effective input prompts to get desired outputs",
                "Engineering hardware for AI systems",
                "Creating user interfaces for AI applications"
            ],
            "correct_answer": 1,
            "explanation": "Prompt engineering involves crafting effective input prompts to guide language models toward producing desired outputs."
        }
    ]
    
    return fallback_questions[:num_questions]

def calculate_quiz_score(user_answers: List[int], quiz_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate quiz score and provide detailed feedback
    
    Args:
        user_answers (List[int]): List of user's selected option indices
        quiz_data (List[Dict]): Original quiz data with correct answers
    
    Returns:
        Dict: Results containing score, feedback, and statistics
    """
    if len(user_answers) != len(quiz_data):
        return {
            'score': 0,
            'total': len(quiz_data),
            'percentage': 0.0,
            'feedback': []
        }
    
    correct_count = 0
    feedback = []
    
    for i, (user_answer, question_data) in enumerate(zip(user_answers, quiz_data)):
        correct_answer_index = question_data['correct_answer']
        is_correct = user_answer == correct_answer_index
        
        if is_correct:
            correct_count += 1
        
        feedback.append({
            'question_num': i + 1,
            'question': question_data['question'],
            'user_answer': question_data['options'][user_answer],
            'correct_answer': question_data['options'][correct_answer_index],
            'correct': is_correct,
            'explanation': question_data.get('explanation', 'No explanation available')
        })
    
    percentage = (correct_count / len(quiz_data)) * 100
    
    return {
        'score': correct_count,
        'total': len(quiz_data),
        'percentage': percentage,
        'feedback': feedback
    }

def get_performance_message(percentage: float) -> str:
    """Get a motivational message based on quiz performance"""
    if percentage >= 90:
        return "🌟 Excellent! You've mastered this material!"
    elif percentage >= 80:
        return "🎯 Great job! You have a solid understanding!"
    elif percentage >= 70:
        return "👍 Good work! You're on the right track!"
    elif percentage >= 60:
        return "📚 Not bad! Consider reviewing the material again."
    else:
        return "💪 Keep studying! Practice makes perfect!"