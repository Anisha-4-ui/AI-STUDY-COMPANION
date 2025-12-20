# evaluation_dataset.py
test_dataset = [
    {
        "query": "What is machine learning?",
        "relevant_chunks": [
            "Machine learning is a subset of artificial intelligence",
            "ML algorithms learn from data without explicit programming",
            "Types include supervised, unsupervised, and reinforcement learning"
        ],
        "reference_answer": "Machine learning is a subset of artificial intelligence that enables computer systems to learn from data without being explicitly programmed. Instead of following pre-written instructions, ML algorithms identify patterns in data and improve their performance over time. The main types include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards and penalties). Machine learning algorithms analyze data, create models, and use these models to make predictions or decisions on new data."
    },
    {
        "query": "Explain neural networks",
        "relevant_chunks": [
            "Neural networks are inspired by biological neurons",
            "They consist of interconnected nodes called neurons",
            "Deep learning uses multiple hidden layers"
        ],
        "reference_answer": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Information flows from input layer through hidden layers to output layer. Deep neural networks with multiple hidden layers enable deep learning, which can learn complex patterns in data."
    },
    {
        "query": "What is Python used for?",
        "relevant_chunks": [
            "Python is a versatile programming language",
            "Used for web development, data science, AI",
            "Popular libraries include NumPy, Pandas, TensorFlow"
        ],
        "reference_answer": "Python is a versatile, high-level programming language used for web development, data science, artificial intelligence, automation, and more. It has extensive libraries like NumPy for numerical computing, Pandas for data analysis, and TensorFlow for machine learning."
    },
    {
        "query": "Define algorithms",
        "relevant_chunks": [
            "Algorithms are step-by-step procedures",
            "They solve computational problems",
            "Examples include sorting and searching algorithms"
        ],
        "reference_answer": "Algorithms are step-by-step procedures or instructions designed to solve computational problems or perform specific tasks. They take input, process it through defined steps, and produce output. Common examples include sorting algorithms (like quicksort) and searching algorithms (like binary search)."
    },
    {
        "query": "What is data science?",
        "relevant_chunks": [
            "Data science extracts insights from data",
            "Involves statistics, programming, and domain knowledge",
            "Uses techniques like data mining and visualization"
        ],
        "reference_answer": "Data science is an interdisciplinary field that extracts insights and knowledge from structured and unstructured data. It combines statistics, programming, mathematics, and domain expertise to analyze data, identify patterns, and make data-driven decisions. Key techniques include data mining, statistical analysis, and data visualization."
    }
]