import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from io import BytesIO
import datetime
import time

# Add aiFeatures/python to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "aiFeatures/python")))
from aiFeatures.python.ai_response import generate_response_without_retrieval, generate_response_with_retrieval, ChatSessionManager
from aiFeatures.python.web_scraping import web_response
from aiFeatures.python.speech_to_text import speech_to_text
from aiFeatures.python.text_to_speech import say as text_to_speech
from aiFeatures.python.rag_pipeline import retrieve_answer, index_pdfs
from quiz_system import generate_quiz_from_content, calculate_quiz_score, get_performance_message

# Load environment variables
load_dotenv()

def create_pdf_download(content, filename="nads_response.pdf"):
    """Create a beautiful PDF from the AI response content"""
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'ResponseTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_LEFT,
        textColor='#4d61fc'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading3'],
        fontSize=13,
        spaceAfter=10,
        spaceBefore=15,
        alignment=TA_LEFT,
        textColor='#2c3e50',
        fontName='Helvetica-Bold'
    )
    
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        rightIndent=0,
        leading=16
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leftIndent=30,
        bulletIndent=10,
        alignment=TA_LEFT,
        leading=16
    )
    
    # Add title header
    story.append(Paragraph("NADS AI Response", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Process content line by line for better formatting
    lines = content.split('\n')
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Empty line - end of paragraph
            if current_paragraph:
                story.append(Paragraph(current_paragraph, content_style))
                story.append(Spacer(1, 0.1*inch))
                current_paragraph = ""
            continue
            
        # Check for headings (lines ending with colon and being short)
        if line.endswith(':') and len(line) < 100 and not line.startswith('-') and not line.startswith('*'):
            if current_paragraph:
                story.append(Paragraph(current_paragraph, content_style))
                story.append(Spacer(1, 0.1*inch))
                current_paragraph = ""
            story.append(Paragraph(f"<b>{line}</b>", heading_style))
            continue
            
        # Check for bullet points
        if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
            if current_paragraph:
                story.append(Paragraph(current_paragraph, content_style))
                story.append(Spacer(1, 0.1*inch))
                current_paragraph = ""
            # Clean bullet point
            bullet_text = line[2:].strip() if line.startswith(('- ', '* ')) else line[2:].strip()
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
            continue
            
        # Check for numbered lists
        import re
        if re.match(r'^\d+\.', line):
            if current_paragraph:
                story.append(Paragraph(current_paragraph, content_style))
                story.append(Spacer(1, 0.1*inch))
                current_paragraph = ""
            story.append(Paragraph(line, bullet_style))
            continue
            
        # Regular content - accumulate into paragraph
        if current_paragraph:
            current_paragraph += " " + line
        else:
            current_paragraph = line
    
    # Add final paragraph if exists
    if current_paragraph:
        story.append(Paragraph(current_paragraph, content_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_quiz_pdf_download(quiz_data, filename="nads_quiz.pdf"):
    """Create a PDF from quiz data with questions, options, and correct answers"""
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'QuizTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_LEFT,
        textColor='#4d61fc'
    )
    
    question_style = ParagraphStyle(
        'QuizQuestion',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=15,
        alignment=TA_LEFT,
        textColor='#2c3e50'
    )
    
    option_style = ParagraphStyle(
        'QuizOption',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=5,
        leftIndent=20,
        alignment=TA_LEFT
    )
    
    correct_answer_style = ParagraphStyle(
        'CorrectAnswer',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=5,
        leftIndent=20,
        alignment=TA_LEFT,
        textColor='#28a745'
    )
    
    explanation_style = ParagraphStyle(
        'Explanation',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=15,
        leftIndent=20,
        alignment=TA_JUSTIFY,
        textColor='#555555'
    )
    
    # Add title
    story.append(Paragraph("NADS Quiz - Study Material", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Add each question
    for i, question_data in enumerate(quiz_data):
        # Question
        story.append(Paragraph(f"<b>Question {i+1}:</b> {question_data['question']}", question_style))
        
        # Options
        for j, option in enumerate(question_data['options']):
            letter = chr(65 + j)  # A, B, C, D
            if j == question_data['correct_answer']:
                story.append(Paragraph(f"<b>{letter}. {option} (Correct Answer)</b>", correct_answer_style))
            else:
                story.append(Paragraph(f"{letter}. {option}", option_style))
        
        # Explanation
        if 'explanation' in question_data:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<i>Explanation: {question_data['explanation']}</i>", explanation_style))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Page config
st.set_page_config(
    page_title="NADS - AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = ChatSessionManager()
if 'session_id' not in st.session_state:
    st.session_state.session_id = "streamlit_session_001"
if 'voice_input_text' not in st.session_state:
    st.session_state.voice_input_text = ""
if 'voice_input_active' not in st.session_state:
    st.session_state.voice_input_active = False
# NEW: Quiz-related session state
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #4d61fc 0%, #6ee7b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        max-width: 80%;
        position: relative;
    }
    
    .user-message {
        background-color: #4d61fc;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .ai-message {
        background-color: #f0f4ff;
        color: #1e1e1e;
        border-left: 3px solid #4d61fc;
        border-bottom-left-radius: 5px;
        padding-bottom: 40px;
    }
    
    .quiz-container {
        background-color: #f8f9ff;
        border: 2px solid #4d61fc;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .quiz-question {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: #2c3e50;
    }
    
    .quiz-results {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4f4dd 100%);
        border: 2px solid #28a745;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .correct-answer {
        color: #28a745;
        font-weight: 600;
    }
    
    .incorrect-answer {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎓 NADS - AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-style: italic; color: #666; font-size: 1.2rem;">Your AI Study Companion</p>', unsafe_allow_html=True)

# RAG Status indicator
if st.session_state.vector_store:
    st.markdown('<div class="status-indicator status-active"> PDFs loaded - RAG enabled</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-indicator status-inactive"> No PDFs loaded</div>', unsafe_allow_html=True)

# Sidebar for file upload and controls
with st.sidebar:
    st.header(" Document Management")
    
    # File upload options
    upload_option = st.radio(
        "Choose upload method:",
        ["Single PDF", "Multiple PDFs", "No Documents"]
    )
    
    uploaded_files = None
    if upload_option == "Single PDF":
        uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False)
        if uploaded_files:
            uploaded_files = [uploaded_files]
    elif upload_option == "Multiple PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files and st.button("Initialize RAG System", type="primary"):
        with st.spinner("Processing PDFs..."):
            try:
                # Save uploaded files to temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Index the PDFs
                    if len(file_paths) == 1:
                        st.session_state.vector_store = index_pdfs(file_paths[0])
                    else:
                        st.session_state.vector_store = index_pdfs(file_paths)
                    
                    if st.session_state.vector_store:
                        st.success(" RAG system initialized successfully!")
                        st.rerun()
                    else:
                        st.error(" Failed to initialize RAG system")
            except Exception as e:
                st.error(f"Error initializing RAG: {str(e)}")
    
    # Clear session button
    if st.button(" Clear Session", help="Clear chat history and documents"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.session_manager = ChatSessionManager()
        st.session_state.voice_input_text = ""
        st.session_state.voice_input_active = False
        # Clear quiz state
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_results = None
        st.success("Session cleared!")
        st.rerun()
    
    st.divider()
    
    # Voice controls
    st.header(" Voice Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Voice Input", help="Click to start voice input"):
            st.session_state.voice_input_active = True
            st.session_state.voice_input_text = ""
            
            status_placeholder = st.empty()
            
            def status_callback(message):
                status_placeholder.info(message)
            
            with st.spinner("Initializing voice recognition..."):
                try:
                    from aiFeatures.python.speech_to_text import speech_to_text_with_feedback
                    voice_query = speech_to_text_with_feedback(status_callback)
                    
                    if voice_query and voice_query.strip():
                        st.session_state.voice_input_text = voice_query.strip()
                        st.session_state.voice_input_active = False
                        status_placeholder.success(f" Voice input captured: '{voice_query}'")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.voice_input_active = False
                        status_placeholder.warning("No speech detected or recognized. Please try again.")
                        
                except Exception as e:
                    st.session_state.voice_input_active = False
                    status_placeholder.error(f"Voice recognition error: {str(e)}")
                    st.error("Please check your microphone permissions and try again.")
    
    with col2:
        if st.button(" Stop Speech", help="Stop current speech output"):
            try:
                from aiFeatures.python.text_to_speech import stop_speech
                success = stop_speech()
                if success:
                    st.success("Speech stopped successfully!")
                else:
                    st.info("No speech was active")
            except Exception as e:
                st.error(f"Error stopping speech: {str(e)}")

# QUIZ DISPLAY SECTION
if st.session_state.current_quiz and not st.session_state.quiz_submitted:
    st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
    
    # Header with download button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header(" Interactive Quiz")
    with col2:
        # Download quiz button
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        quiz_filename = f"nads_quiz_{timestamp}.pdf"
        quiz_pdf_buffer = create_quiz_pdf_download(st.session_state.current_quiz, quiz_filename)
        
        st.download_button(
            label="📄 Download",
            data=quiz_pdf_buffer,
            file_name=quiz_filename,
            mime="application/pdf",
            help="Download quiz with answers as PDF",
            use_container_width=True
        )
    
    quiz_data = st.session_state.current_quiz
    
    # Display quiz questions
    for i, question_data in enumerate(quiz_data):
        st.markdown(f'<div class="quiz-question">Question {i+1}: {question_data["question"]}</div>', unsafe_allow_html=True)
        
        # Radio buttons for options
        selected_option = st.radio(
            f"Select your answer for Question {i+1}:",
            options=range(len(question_data["options"])),
            format_func=lambda x: question_data["options"][x],
            key=f"quiz_q_{i}",
            index=st.session_state.quiz_answers.get(i, 0)
        )
        
        # Store the answer
        st.session_state.quiz_answers[i] = selected_option
        st.divider()
    
    # Submit quiz button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Submit Quiz", type="primary", use_container_width=True):
            # Calculate results
            user_answers = [st.session_state.quiz_answers.get(i, 0) for i in range(len(quiz_data))]
            st.session_state.quiz_results = calculate_quiz_score(user_answers, quiz_data)
            st.session_state.quiz_submitted = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# QUIZ RESULTS DISPLAY
elif st.session_state.quiz_submitted and st.session_state.quiz_results:
    results = st.session_state.quiz_results
    
    st.markdown('<div class="quiz-results">', unsafe_allow_html=True)
    st.header(" Quiz Results")
    
    # Overall score
    st.metric(
        label="Your Score",
        value=f"{results['score']}/{results['total']}",
        delta=f"{results['percentage']:.1f}%"
    )
    
    st.markdown(f"{get_performance_message(results['percentage'])}")
    
    # Detailed feedback
    st.subheader("Detailed Feedback:")
    
    for feedback in results['feedback']:
        if feedback['correct']:
            st.markdown(f'<p class="correct-answer"> Question {feedback["question_num"]}: Correct!</p>', unsafe_allow_html=True)
            st.info(feedback['explanation'])
        else:
            st.markdown(f'<p class="incorrect-answer"> Question {feedback["question_num"]}: Incorrect</p>', unsafe_allow_html=True)
            st.warning(f"Your answer: {feedback['user_answer']}")
            st.success(f"Correct answer: {feedback['correct_answer']}")
            st.info(feedback['explanation'])
    
    # Restart quiz button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Take New Quiz", type="secondary", use_container_width=True):
            # Reset quiz state
            st.session_state.current_quiz = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_results = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main chat interface
st.header(" Chat with NADS")

# Display chat history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Create a container for AI message with download button
            message_container = st.container()
            with message_container:
                st.markdown(f'<div class="chat-message ai-message"><strong>NADS:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
                # Add download button for AI responses
                col1, col2 = st.columns([10, 1])
                with col2:
                    # Generate filename with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"nads_response_{timestamp}.pdf"
                    
                    # Create PDF buffer
                    pdf_buffer = create_pdf_download(message["content"], filename)
                    
                    # Download button
                    st.download_button(
                        label="📄",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf",
                        help="Download response as PDF",
                        key=f"download_{i}"
                    )

# Input section
if not st.session_state.current_quiz:  # Only show input when not in quiz mode
    input_container = st.container()
    with input_container:
        current_text_value = st.session_state.voice_input_text
        
        # Text input with voice input pre-filled
        user_query = st.text_area(
            "Ask NADS a question:",
            value=current_text_value,
            height=100,
            placeholder="Type your question here or use voice input...",
            key="user_input_area"
        )
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            send_button = st.button(" Ask NADS", type="primary", use_container_width=True)
        
        with col2:
            summarize_button = st.button(" Summarize", use_container_width=True, help="Summarize uploaded documents")
        
        with col3:
            quiz_button = st.button(" Generate Quiz", use_container_width=True, help="Create interactive quiz")
        
        with col4:
            if st.button(" Clear Input", use_container_width=True):
                st.session_state.voice_input_text = ""
                st.session_state.voice_input_active = False
                st.rerun()

# Handle send button action
if not st.session_state.current_quiz and send_button:
    final_query = user_query.strip()
    
    if not final_query:
        st.warning("Please enter a question or use voice input first.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user', 
            'content': final_query
        })
        
        st.session_state.voice_input_text = ""
        st.session_state.voice_input_active = False
        
        with st.spinner("NADS is thinking..."):
            try:
                # Check if retrieval is needed
                retrieved_info = ""
                scraped_info = ""
                
                if st.session_state.vector_store:
                    retrieved_info = retrieve_answer(final_query, st.session_state.vector_store)
                
                # Generate response
                if retrieved_info:
                    response = generate_response_with_retrieval(
                        st.session_state.session_id,
                        final_query,
                        retrieved_info,
                        st.session_state.session_manager
                    )
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'retrieved_info': retrieved_info
                    })
                else:
                    scraped_info = web_response(final_query)
                    response = generate_response_without_retrieval(
                        st.session_state.session_id,
                        final_query,
                        scraped_info,
                        st.session_state.session_manager
                    )
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'scraped_info': scraped_info
                    })
                
                # Text to speech (optional)
                try:
                    text_to_speech(response)
                except Exception as e:
                    st.warning(f"Could not play audio: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        
        st.rerun()

# Summarize functionality
if not st.session_state.current_quiz and summarize_button:
    if not st.session_state.vector_store:
        st.warning("Please upload and initialize PDF documents first!")
    else:
        with st.spinner("Generating summary..."):
            try:
                summary_query = "Provide a comprehensive summary of the main topics and key points covered in the documents"
                retrieved_info = retrieve_answer(summary_query, st.session_state.vector_store)
                
                if retrieved_info:
                    summary_response = generate_response_with_retrieval(
                        st.session_state.session_id,
                        summary_query,
                        retrieved_info,
                        st.session_state.session_manager
                    )
                    
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': ' Generate document summary'
                    })
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': summary_response,
                        'retrieved_info': retrieved_info
                    })
                    
                    st.rerun()
                else:
                    st.error("Could not retrieve information for summary")
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

# UPDATED QUIZ FUNCTIONALITY
if not st.session_state.current_quiz and quiz_button:
    with st.spinner("Generating interactive quiz..."):
        try:
            # Pass the vector_store directly to generate_quiz_from_content
            # It will handle retrieving comprehensive content internally
            if st.session_state.vector_store:
                quiz_data = generate_quiz_from_content("", num_questions=10, vector_store=st.session_state.vector_store)
            else:
                # Fallback to recent chat history if no RAG content
                content_for_quiz = ""
                if st.session_state.chat_history:
                    recent_messages = st.session_state.chat_history[-10:]  # Last 10 messages
                    content_for_quiz = "\n\n".join([msg['content'] for msg in recent_messages if msg['role'] == 'assistant'])
                
                if content_for_quiz:
                    quiz_data = generate_quiz_from_content(content_for_quiz, num_questions=10)
                else:
                    st.warning("No content available for quiz generation. Please chat with NADS or upload documents first!")
                    quiz_data = None
            
            if quiz_data:
                st.session_state.current_quiz = quiz_data
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_results = None
                st.rerun()
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")

# Welcome message
if not st.session_state.chat_history and not st.session_state.current_quiz:
    st.info(" Hello! I'm NADS, your AI Study Companion. You can ask me questions directly, upload PDF documents, or generate interactive quizzes. How can I help you today?")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🤖 NADS v2.0 - Powered by Google Gemini & Streamlit<br>
        Built with ❤ for personalized learning
    </div>
    """, 
    unsafe_allow_html=True
)