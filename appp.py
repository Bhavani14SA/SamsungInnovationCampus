import json
from datetime import datetime, timedelta
import re
import pypdf
import docx
from io import StringIO, BytesIO
from typing import Dict, List, Any
import os
import random
import hashlib
import streamlit as st

# AI/ML imports
try:
    import torch
    from transformers import (
        T5ForConditionalGeneration, T5Tokenizer,
        pipeline,
        AutoTokenizer, AutoModel
    )
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
        
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    AI_MODELS_AVAILABLE = False
    st.warning(f"AI models not available: {e}. Running in basic mode.")

# Configure page
st.set_page_config(
    page_title="AI-Powered Syllabus Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 15px 15px;
    }
    .ai-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .question-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .week-plan {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .assignment-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .confidence-score {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 11px;
        margin-left: 5px;
    }
    .quiz-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
    }
    .question-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ai_models():
    """Load and cache AI models for analysis"""
    if not AI_MODELS_AVAILABLE:
        return None
        
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
        qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return {
            'summarizer': summarizer,
            'qg_tokenizer': qg_tokenizer,
            'qg_model': qg_model,
            'sentence_model': sentence_model
        }
    except Exception as e:
        st.error(f"Error loading AI models: {str(e)}")
        return None

class MCQGenerator:
    """Enhanced MCQ generation with dynamic question creation and evaluation"""
    
    def __init__(self, ai_techniques):
        self.ai_techniques = ai_techniques
        self.question_templates = [
            "What is the primary purpose of {}?",
            "Which of the following best describes {}?",
            "What are the key characteristics of {}?",
            "How does {} relate to other concepts?",
            "What is the main advantage of {}?",
            "Which statement about {} is correct?",
            "What is the most important aspect of {}?",
            "How is {} typically implemented?",
            "What problem does {} solve?",
            "Which factor is most critical for {}?"
        ]
    
    def generate_mcq_from_text(self, text, topic, num_questions=10, seed=None):
        """Generate MCQ questions with dynamic variations"""
        if seed:
            random.seed(seed)
            if AI_MODELS_AVAILABLE:
                np.random.seed(seed)
        
        questions = []
        if AI_MODELS_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = text.split('. ')
            
        topic_sentences = [s for s in sentences if topic.lower() in s.lower()]
        
        if not topic_sentences:
            topic_sentences = [s for s in sentences if len(s.split()) > 10]
        
        if len(topic_sentences) < 3:
            return self.generate_fallback_mcq(topic, num_questions)
        
        for i in range(min(num_questions, len(topic_sentences))):
            template_idx = (i + random.randint(0, 100)) % len(self.question_templates)
            template = self.question_templates[template_idx]
            
            question_text = template.format(topic)
            context_sentence = topic_sentences[i]
            options = self.generate_options_from_context(context_sentence, topic)
            
            correct_answer = options[0]
            random.shuffle(options)
            correct_index = options.index(correct_answer)
            
            questions.append({
                "id": f"q_{i+1}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                "question": question_text,
                "options": options,
                "correct_answer": correct_index,
                "explanation": f"Based on the context: {context_sentence[:100]}...",
                "topic": topic,
                "difficulty": self.assess_question_difficulty(question_text, options)
            })
        
        return questions
    
    def generate_options_from_context(self, context, topic):
        """Generate plausible options from context"""
        correct_phrases = self.extract_key_phrases(context)
        correct_answer = correct_phrases[0] if correct_phrases else f"Key aspect of {topic}"
        distractors = self.generate_distractors(topic, correct_answer)
        
        options = [correct_answer] + distractors[:3]
        while len(options) < 4:
            options.append(f"Alternative interpretation of {topic}")
        
        return options
    
    def extract_key_phrases(self, context):
        """Extract meaningful phrases from context"""
        if AI_MODELS_AVAILABLE:
            sentences = sent_tokenize(context)
        else:
            sentences = context.split('. ')
            
        phrases = []
        
        for sentence in sentences[:2]:
            words = sentence.split()
            if len(words) > 5:
                phrase = ' '.join(words[:6]).strip('.,!?')
                if len(phrase) > 10:
                    phrases.append(phrase)
        
        return phrases if phrases else ["Key concept from the text"]
    
    def generate_distractors(self, topic, correct_answer):
        """Generate plausible wrong answers"""
        distractors = [
            f"Unrelated aspect of {topic}",
            f"Common misconception about {topic}",
            f"Alternative but incorrect view of {topic}",
            f"Outdated understanding of {topic}",
            f"Superficial interpretation of {topic}",
            f"Overgeneralized concept of {topic}"
        ]
        
        topic_variations = [
            f"Opposite characteristic of {topic}",
            f"Peripheral feature of {topic}",
            f"Theoretical but impractical aspect of {topic}"
        ]
        
        all_distractors = distractors + topic_variations
        random.shuffle(all_distractors)
        
        selected = []
        for d in all_distractors:
            if d.lower() != correct_answer.lower() and len(selected) < 3:
                selected.append(d)
        
        return selected
    
    def generate_fallback_mcq(self, topic, num_questions):
        """Fallback MCQ generation when context is limited"""
        fallback_questions = []
        
        base_templates = [
            ("What is {}?", [
                f"A fundamental concept in the field",
                f"An advanced technique rarely used",
                f"A deprecated approach",
                f"A theoretical framework only"
            ]),
            ("Which best describes {}?", [
                f"An important methodology",
                f"A minor consideration", 
                f"An obsolete practice",
                f"A controversial theory"
            ]),
            ("The primary purpose of {} is:", [
                f"To achieve specific educational objectives",
                f"To complicate the learning process",
                f"To replace traditional methods entirely",
                f"To serve as optional supplementary material"
            ])
        ]
        
        for i in range(num_questions):
            template_idx = i % len(base_templates)
            question_template, option_templates = base_templates[template_idx]
            
            question_text = question_template.format(topic)
            options = [opt.replace("{}", topic) for opt in option_templates]
            random.shuffle(options)
            
            fallback_questions.append({
                "id": f"fallback_q_{i+1}_{hashlib.md5(question_text.encode()).hexdigest()[:8]}",
                "question": question_text,
                "options": options,
                "correct_answer": 0,
                "explanation": f"This question tests understanding of {topic}",
                "topic": topic,
                "difficulty": "Medium"
            })
        
        return fallback_questions
    
    def assess_question_difficulty(self, question, options):
        """Assess question difficulty based on complexity"""
        factors = {
            'question_length': len(question.split()),
            'option_complexity': sum(len(opt.split()) for opt in options) / len(options),
            'abstract_words': sum(1 for word in question.split() if len(word) > 7)
        }
        
        total_score = (
            min(factors['question_length'] / 15, 1) * 0.4 +
            min(factors['option_complexity'] / 8, 1) * 0.4 +
            min(factors['abstract_words'] / 3, 1) * 0.2
        )
        
        if total_score < 0.4:
            return "Easy"
        elif total_score < 0.7:
            return "Medium"
        else:
            return "Hard"

class QuizEvaluator:
    """Quiz evaluation and scoring system"""
    
    def __init__(self):
        self.passing_score = 0.6
    
    def evaluate_quiz(self, questions, user_answers):
        """Evaluate quiz and provide detailed feedback"""
        if not questions or not user_answers:
            return None
        
        total_questions = len(questions)
        correct_answers = 0
        detailed_results = []
        
        for i, question in enumerate(questions):
            user_answer = user_answers.get(f"question_{i}", -1)
            is_correct = user_answer == question['correct_answer']
            
            if is_correct:
                correct_answers += 1
            
            detailed_results.append({
                "question_id": question['id'],
                "question": question['question'],
                "user_answer": user_answer,
                "correct_answer": question['correct_answer'],
                "options": question['options'],
                "is_correct": is_correct,
                "explanation": question['explanation'],
                "topic": question['topic'],
                "difficulty": question['difficulty']
            })
        
        score_percentage = (correct_answers / total_questions) * 100
        insights = self.generate_performance_insights(detailed_results, score_percentage)
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "score_percentage": score_percentage,
            "grade": self.calculate_grade(score_percentage),
            "passed": score_percentage >= (self.passing_score * 100),
            "detailed_results": detailed_results,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_grade(self, score_percentage):
        """Calculate letter grade based on percentage"""
        if score_percentage >= 90:
            return "A"
        elif score_percentage >= 80:
            return "B"
        elif score_percentage >= 70:
            return "C"
        elif score_percentage >= 60:
            return "D"
        else:
            return "F"
    
    def generate_performance_insights(self, results, score_percentage):
        """Generate insights about performance"""
        insights = []
        
        if score_percentage >= 90:
            insights.append("Excellent performance! You have mastered the material.")
        elif score_percentage >= 70:
            insights.append("Good performance with room for improvement in some areas.")
        elif score_percentage >= 60:
            insights.append("Passing performance, but consider reviewing key concepts.")
        else:
            insights.append("Consider additional study time and review of fundamental concepts.")
        
        # Topic-specific analysis
        topic_performance = {}
        for result in results:
            topic = result['topic']
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            topic_performance[topic]['total'] += 1
            if result['is_correct']:
                topic_performance[topic]['correct'] += 1
        
        weak_topics = []
        strong_topics = []
        
        for topic, performance in topic_performance.items():
            topic_score = (performance['correct'] / performance['total']) * 100
            if topic_score < 50:
                weak_topics.append(topic)
            elif topic_score >= 80:
                strong_topics.append(topic)
        
        if weak_topics:
            insights.append(f"Focus on improving: {', '.join(weak_topics[:3])}")
        if strong_topics:
            insights.append(f"Strong performance in: {', '.join(strong_topics[:3])}")
        
        return insights

class AITechniques:
    """Advanced AI analysis techniques for educational content"""
    
    def __init__(self, models):
        if models and AI_MODELS_AVAILABLE:
            self.models = models
            self.summarizer = models['summarizer']
            self.qg_tokenizer = models['qg_tokenizer']
            self.qg_model = models['qg_model']
            self.sentence_model = models['sentence_model']
        else:
            self.models = None
            self.summarizer = None
            self.qg_tokenizer = None
            self.qg_model = None
            self.sentence_model = None
    
    def extract_semantic_topics(self, text, num_topics=8):
        """Use semantic analysis to extract meaningful topics"""
        if not self.sentence_model:
            return self.fallback_topic_extraction(text)
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 5:
                return self.fallback_topic_extraction(text)
            
            embeddings = self.sentence_model.encode(sentences)
            
            n_clusters = min(num_topics, len(sentences) // 2)
            if n_clusters < 2:
                return self.fallback_topic_extraction(text)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            topics = []
            for i in range(n_clusters):
                cluster_sentences = [sentences[j] for j, c in enumerate(clusters) if c == i]
                if cluster_sentences:
                    cluster_embeddings = [embeddings[j] for j, c in enumerate(clusters) if c == i]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
                    best_idx = np.argmin(distances)
                    representative = cluster_sentences[best_idx]
                    
                    topic = self.extract_topic_from_sentence(representative)
                    if topic and len(topic) > 5:
                        topics.append(topic)
            
            return topics[:num_topics]
        
        except Exception as e:
            st.warning(f"Semantic analysis failed, using fallback: {str(e)}")
            return self.fallback_topic_extraction(text)
    
    def extract_topic_from_sentence(self, sentence):
        """Extract topic from a sentence using NLP techniques"""
        sentence = re.sub(r'^(This|The|A|An)\s+', '', sentence)
        sentence = sentence.strip('.,!?')
        
        words = sentence.split()
        if len(words) > 8:
            return ' '.join(words[:6])
        return sentence
    
    def fallback_topic_extraction(self, text):
        """Fallback topic extraction using basic text analysis"""
        try:
            if AI_MODELS_AVAILABLE:
                sentences = sent_tokenize(text)
            else:
                sentences = text.split('. ')
                
            sentences = [s for s in sentences if len(s.split()) > 5 and len(s.split()) < 20]
            
            if len(sentences) < 5:
                return ["Introduction", "Core Concepts", "Applications", "Methodology"]
            
            if AI_MODELS_AVAILABLE:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                
                sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
                top_indices = sentence_scores.argsort()[-8:][::-1]
                
                topics = []
                for idx in top_indices:
                    topic = self.extract_topic_from_sentence(sentences[idx])
                    if topic not in topics:
                        topics.append(topic)
                
                return topics[:8]
            else:
                # Basic topic extraction without ML
                topics = []
                for sentence in sentences[:8]:
                    topic = self.extract_topic_from_sentence(sentence)
                    if topic not in topics and len(topic) > 5:
                        topics.append(topic)
                return topics
        
        except Exception as e:
            return ["Introduction", "Core Concepts", "Methodology", "Applications", "Assessment"]
    
    def generate_ai_summary(self, text):
        """Generate AI-powered summary using BART"""
        if not self.summarizer:
            return self.fallback_summary(text)
        
        try:
            max_length = 1000
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            summaries = []
            for chunk in chunks[:3]:
                if len(chunk.split()) > 50:
                    summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            
            combined_summary = ' '.join(summaries)
            
            if len(summaries) > 1:
                final_summary = self.summarizer(combined_summary, max_length=200, min_length=80, do_sample=False)
                return final_summary[0]['summary_text']
            
            return combined_summary if combined_summary else self.fallback_summary(text)
            
        except Exception as e:
            st.warning(f"AI summarization failed, using fallback: {str(e)}")
            return self.fallback_summary(text)
    
    def fallback_summary(self, text):
        """Fallback summary extraction"""
        if AI_MODELS_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = text.split('. ')
            
        if len(sentences) < 3:
            return "Course material covering fundamental concepts and practical applications."
        
        first_sentence = next((s for s in sentences if len(s.split()) > 10), sentences[0])
        middle_sentence = sentences[len(sentences)//2] if len(sentences) > 5 else ""
        
        summary = first_sentence
        if middle_sentence and len(middle_sentence.split()) > 8:
            summary += " " + middle_sentence
        
        return summary[:300] + "..." if len(summary) > 300 else summary
    
    def generate_ai_questions(self, text, num_questions=20):
        """Generate questions using T5 question generation model"""
        if not self.qg_model or not self.qg_tokenizer:
            return self.fallback_question_generation(text)
        
        try:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
            
            questions_by_category = []
            
            for i, paragraph in enumerate(paragraphs[:5]):
                paragraph_questions = []
                sentences = sent_tokenize(paragraph)
                
                for sentence in sentences[:3]:
                    if len(sentence.split()) > 8:
                        try:
                            input_text = f"generate question: {sentence}"
                            input_ids = self.qg_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                            
                            with torch.no_grad():
                                outputs = self.qg_model.generate(
                                    input_ids,
                                    max_length=100,
                                    num_beams=4,
                                    early_stopping=True,
                                    temperature=0.7
                                )
                            
                            question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            question = question.strip()
                            if question and len(question) > 10 and question.endswith('?'):
                                paragraph_questions.append(question)
                                
                        except Exception as e:
                            continue
                
                if paragraph_questions:
                    category_name = self.extract_category_name(paragraph)
                    questions_by_category.append({
                        "category": category_name,
                        "questions": paragraph_questions[:4],
                        "confidence": min(0.9, len(paragraph_questions) / 4)
                    })
            
            return questions_by_category[:6] if questions_by_category else self.fallback_question_generation(text)
            
        except Exception as e:
            st.warning(f"AI question generation failed, using fallback: {str(e)}")
            return self.fallback_question_generation(text)
    
    def extract_category_name(self, paragraph):
        """Extract meaningful category name from paragraph"""
        if AI_MODELS_AVAILABLE:
            sentences = sent_tokenize(paragraph)
        else:
            sentences = paragraph.split('. ')
            
        first_sentence = sentences[0] if sentences else paragraph
        
        lines = paragraph.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 60 and ':' not in line:
                words = line.split()
                if len(words) <= 8 and any(word[0].isupper() for word in words):
                    return line
        
        words = first_sentence.split()[:6]
        return ' '.join(words).strip('.,!?')
    
    def fallback_question_generation(self, text):
        """Fallback question generation using templates"""
        topics = self.extract_semantic_topics(text, 5)
        
        question_templates = [
            "What is {}?",
            "How does {} work?", 
            "Explain the importance of {}",
            "What are the key components of {}?",
            "How is {} applied in practice?",
            "What are the advantages and disadvantages of {}?",
            "Compare {} with related concepts",
            "What are the future trends in {}?"
        ]
        
        questions_by_category = []
        for topic in topics[:4]:
            questions = []
            for template in question_templates[:5]:
                question = template.format(topic.lower())
                questions.append(question.capitalize())
            
            questions_by_category.append({
                "category": topic,
                "questions": questions,
                "confidence": 0.7
            })
        
        return questions_by_category
    
    def analyze_difficulty_and_prerequisites(self, text):
        """Use AI to analyze course difficulty and extract prerequisites"""
        difficulty_indicators = {
            'beginner': ['introduction', 'basic', 'fundamentals', 'overview', 'primer'],
            'intermediate': ['analysis', 'design', 'implementation', 'application', 'methods'],
            'advanced': ['advanced', 'complex', 'optimization', 'research', 'theoretical', 'algorithms']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for level, indicators in difficulty_indicators.items():
            score = sum(text_lower.count(indicator) for indicator in indicators)
            scores[level] = score
        
        difficulty = max(scores, key=scores.get) if scores else 'intermediate'
        
        prereq_patterns = [
            r'prerequisite[s]?[:\-\s]+([^.]*)',
            r'requirement[s]?[:\-\s]+([^.]*)',
            r'assumes knowledge of[:\-\s]+([^.]*)',
            r'background in[:\-\s]+([^.]*)'
        ]
        
        prerequisites = []
        for pattern in prereq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                prereq_text = re.sub(r'[^\w\s,]', '', match)
                prereqs = [p.strip().title() for p in prereq_text.split(',') if len(p.strip()) > 3]
                prerequisites.extend(prereqs[:3])
        
        if not prerequisites:
            prerequisites = ["Basic Understanding of the Subject"]
        
        return difficulty.title(), prerequisites[:5]

class AISyllabusAnalyzer:
    def __init__(self):
        self.models = load_ai_models()
        self.ai_techniques = AITechniques(self.models) if AI_MODELS_AVAILABLE else AITechniques(None)
        self.mcq_generator = MCQGenerator(self.ai_techniques)
        self.quiz_evaluator = QuizEvaluator()
        
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded file with multiple methods for robustness"""
        try:
            file_size = len(uploaded_file.getvalue())
            st.info(f"Processing file: {uploaded_file.name} ({file_size / (1024*1024):.1f} MB)")
            
            if uploaded_file.type == "application/pdf":
                return self.extract_pdf_text(uploaded_file, file_size)
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            elif uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            
            else:
                st.error("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
                return None
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def extract_pdf_text(self, uploaded_file, file_size):
        """Extract text from PDF using pypdf"""
        try:
            with st.spinner("Extracting text from PDF..."):
                uploaded_file.seek(0)
                pdf_reader = pypdf.PdfReader(uploaded_file)
                text = ""
                
                total_pages = len(pdf_reader.pages)
                if total_pages > 10:
                    progress_bar = st.progress(0)
                
                for page_num, page in enumerate(pdf_reader.pages[:min(100, total_pages)]):
                    if total_pages > 10:
                        progress_bar.progress((page_num + 1) / min(100, total_pages))
                    
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except:
                        continue
                
                if total_pages > 10:
                    progress_bar.progress(1.0)
                    st.success(f"Successfully extracted text from {min(total_pages, 100)} pages")
                
                if len(text.strip()) > 100:
                    return self.clean_extracted_text(text)
                else:
                    st.error("Could not extract readable text from PDF")
                    return None
                    
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return None
    
    def clean_extracted_text(self, text):
        """Clean and preprocess extracted text"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        if len(text) > 50000:
            text = text[:50000]
        
        return text
    
    def analyze_syllabus_with_ai(self, text):
        """Complete AI analysis of syllabus"""
        try:
            text = self.clean_extracted_text(text)
            
            if not text or len(text.strip()) < 100:
                return None
            
            analysis = {
                'syllabus_text': text,
                'ai_metrics': {
                    'text_length': len(text),
                    'processing_method': 'AI-Enhanced' if self.ai_techniques.models else 'Fallback',
                    'topics_extracted': 0,
                    'questions_generated': 0
                }
            }
            
            # Generate summary
            if self.ai_techniques.models:
                summary = self.ai_techniques.generate_ai_summary(text)
                key_topics = self.ai_techniques.extract_semantic_topics(text, 8)
                difficulty, prerequisites = self.ai_techniques.analyze_difficulty_and_prerequisites(text)
                expected_questions = self.ai_techniques.generate_ai_questions(text, 20)
                ai_confidence = 0.85
            else:
                summary = self.basic_summary(text)
                key_topics = self.basic_topic_extraction(text)
                difficulty, prerequisites = self.basic_difficulty_analysis(text)
                expected_questions = self.basic_question_generation(key_topics)
                ai_confidence = 0.6
            
            analysis['summary'] = {
                'title': self.extract_title(text),
                'description': summary,
                'key_topics': key_topics,
                'difficulty': difficulty,
                'prerequisites': prerequisites,
                'duration': self.estimate_duration(text),
                'ai_confidence': ai_confidence
            }
            
            analysis['expected_questions'] = expected_questions
            analysis['ai_metrics']['topics_extracted'] = len(key_topics)
            analysis['ai_metrics']['questions_generated'] = sum(len(cat['questions']) for cat in expected_questions)
            
            # Generate study plan
            analysis['study_plan'] = self.generate_study_plan(key_topics, difficulty)
            
            # Generate assignments
            analysis['assignments'] = self.generate_assignments(key_topics, difficulty)
            
            return analysis
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None
    
    def extract_title(self, text):
        """Extract course title from text"""
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                if any(word in line.lower() for word in ['course', 'syllabus', 'curriculum', 'program']):
                    return line
                if line.isupper() or (line[0].isupper() and sum(1 for c in line if c.isupper()) > len(line) * 0.3):
                    return line
        
        return "Course Syllabus Analysis"
    
    def basic_summary(self, text):
        """Basic summary without AI"""
        if AI_MODELS_AVAILABLE:
            sentences = sent_tokenize(text)[:5]
        else:
            sentences = text.split('. ')[:5]
            
        summary_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 8 and len(sentence) < 200:
                summary_sentences.append(sentence)
            if len(summary_sentences) >= 3:
                break
        
        return ' '.join(summary_sentences) if summary_sentences else "Course covering various educational topics and concepts."
    
    def basic_topic_extraction(self, text):
        """Basic topic extraction without AI"""
        topics = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 60:
                if re.match(r'^\d+\.?\s+', line) or line.startswith(('Chapter', 'Module', 'Unit')):
                    cleaned = re.sub(r'^(Chapter|Module|Unit)\s*\d+:?\s*', '', line, flags=re.IGNORECASE)
                    if cleaned and len(cleaned) > 5:
                        topics.append(cleaned)
        
        return topics[:8] if topics else ["Introduction", "Core Concepts", "Applications"]
    
    def basic_question_generation(self, topics):
        """Basic question generation without AI"""
        questions_by_category = []
        question_starters = ["What is", "How does", "Explain", "Why is", "What are the benefits of"]
        
        for topic in topics[:4]:
            questions = [f"{starter} {topic.lower()}?" for starter in question_starters]
            questions_by_category.append({
                "category": topic,
                "questions": questions,
                "confidence": 0.6
            })
        
        return questions_by_category
    
    def basic_difficulty_analysis(self, text):
        """Basic difficulty analysis without AI"""
        text_lower = text.lower()
        
        beginner_words = ['basic', 'introduction', 'fundamentals', 'overview']
        advanced_words = ['advanced', 'complex', 'research', 'analysis']
        
        beginner_count = sum(text_lower.count(word) for word in beginner_words)
        advanced_count = sum(text_lower.count(word) for word in advanced_words)
        
        if advanced_count > beginner_count:
            difficulty = "Advanced"
        elif beginner_count > 0:
            difficulty = "Beginner"
        else:
            difficulty = "Intermediate"
        
        return difficulty, ["Basic Understanding of the Subject"]
    
    def estimate_duration(self, text):
        """Estimate course duration based on content"""
        word_count = len(text.split())
        
        if word_count < 1000:
            return "4-6 weeks"
        elif word_count < 3000:
            return "8-10 weeks"
        elif word_count < 5000:
            return "12-14 weeks"
        else:
            return "15+ weeks"
    
    def generate_study_plan(self, topics, difficulty):
        """Generate study plan based on topics"""
        total_weeks = max(len(topics), 8)
        topics_per_week = max(1, len(topics) // total_weeks)
        
        weekly_schedule = []
        topic_index = 0
        
        for week in range(1, total_weeks + 1):
            week_topics = []
            for _ in range(topics_per_week):
                if topic_index < len(topics):
                    week_topics.append(topics[topic_index])
                    topic_index += 1
            
            if not week_topics and topic_index < len(topics):
                week_topics.append(topics[topic_index])
                topic_index += 1
            
            time_commitment = self.get_time_commitment(difficulty)
            
            weekly_schedule.append({
                "week": week,
                "topics": week_topics if week_topics else ["Review and Assessment"],
                "time_commitment": time_commitment,
                "learning_goals": [f"Master {topic}" for topic in week_topics[:2]],
                "activities": self.generate_weekly_activities(week_topics, week),
                "ai_generated": True if self.ai_techniques.models else False
            })
        
        study_tips = [
            "Create a consistent study schedule",
            "Take regular breaks using the Pomodoro technique",
            "Practice active recall and spaced repetition",
            "Form study groups for discussion",
            "Use multiple learning modalities (visual, auditory, kinesthetic)"
        ]
        
        return {
            "total_weeks": total_weeks,
            "weekly_schedule": weekly_schedule,
            "study_tips": study_tips
        }
    
    def get_time_commitment(self, difficulty):
        """Get time commitment based on difficulty"""
        if difficulty == "Beginner":
            return "6-8 hours per week"
        elif difficulty == "Advanced":
            return "12-15 hours per week"
        else:
            return "8-10 hours per week"
    
    def generate_weekly_activities(self, topics, week):
        """Generate activities for the week"""
        activities = []
        
        for topic in topics:
            activities.extend([
                f"Read materials on {topic}",
                f"Complete exercises related to {topic}",
                f"Review and summarize {topic}"
            ])
        
        if week % 4 == 0:
            activities.append("Complete comprehensive review quiz")
        
        return activities[:5]
    
    def generate_assignments(self, topics, difficulty):
        """Generate course assignments"""
        assignments = []
        
        # Major project
        assignments.append({
            "title": "Course Capstone Project",
            "description": f"Comprehensive project integrating all course topics with focus on practical application",
            "tasks": [
                "Research and analyze key concepts",
                "Develop practical implementation or case study",
                "Create detailed presentation",
                "Submit final report with analysis"
            ],
            "difficulty": difficulty,
            "estimated_time": "20-25 hours",
            "week": max(6, len(topics)),
            "ai_generated": True if self.ai_techniques.models else False,
            "skills_developed": ["Research", "Analysis", "Communication", "Critical Thinking"]
        })
        
        # Mid-term assessment
        assignments.append({
            "title": "Mid-Course Assessment",
            "description": "Comprehensive evaluation of understanding of first half of course material",
            "tasks": [
                "Review all covered topics",
                "Complete practice problems",
                "Take timed assessment",
                "Reflect on learning progress"
            ],
            "difficulty": "Medium",
            "estimated_time": "8-10 hours",
            "week": max(4, len(topics) // 2),
            "ai_generated": True if self.ai_techniques.models else False,
            "skills_developed": ["Problem Solving", "Time Management"]
        })
        
        return assignments
    
    def generate_quiz_questions(self, text, topics, num_questions=10, randomize=True):
        """Generate MCQ questions for quiz"""
        all_questions = []
        questions_per_topic = max(1, num_questions // len(topics))
        
        seed = None if randomize else 42
        
        for topic in topics[:min(len(topics), 5)]:
            topic_questions = self.mcq_generator.generate_mcq_from_text(
                text, topic, questions_per_topic, seed
            )
            all_questions.extend(topic_questions)
        
        # Ensure we have enough questions
        while len(all_questions) < num_questions and len(all_questions) > 0:
            all_questions.extend(all_questions[:num_questions - len(all_questions)])
        
        if randomize:
            random.shuffle(all_questions)
        
        return all_questions[:num_questions]

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Syllabus Analyzer with MCQ Quiz</h1>
        <p>Transform your syllabus with advanced AI models for intelligent insights + Interactive MCQ Testing</p>
        <span class="ai-badge">Enhanced with Dynamic MCQ System</span>
    </div>
    """, unsafe_allow_html=True)
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if not st.session_state.models_loaded:
        with st.spinner("📄 Loading AI models (T5, BART, Sentence-BERT)... This may take a moment."):
            try:
                analyzer = AISyllabusAnalyzer()
                st.session_state.analyzer = analyzer
                st.session_state.models_loaded = True
                if analyzer.models and AI_MODELS_AVAILABLE:
                    st.success("✅ AI models loaded successfully!")
                else:
                    st.warning("⚠️ Running in fallback mode (AI models not available)")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.session_state.analyzer = AISyllabusAnalyzer()
                st.session_state.models_loaded = True
    else:
        analyzer = st.session_state.analyzer
    
    # Initialize session state
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = None
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = None
    
    # Display AI capabilities info
    with st.expander("🤖 AI Capabilities in This Analyzer"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **🧠 BART Summarization**
            - Intelligent content summarization
            - Key concept extraction
            - Context-aware descriptions
            """)
        
        with col2:
            st.markdown("""
            **❓ T5 Question Generation**
            - Automatic question creation
            - Context-based queries
            - Exam-style questions
            """)
        
        with col3:
            st.markdown("""
            **📊 Sentence-BERT Analysis**
            - Semantic topic clustering
            - Content similarity analysis
            - Intelligent categorization
            """)
        
        with col4:
            st.markdown("""
            **🧠 MCQ Quiz System**
            - Dynamic question generation
            - Automated evaluation
            - Performance analytics
            """)
    
    # Sidebar for input
    with st.sidebar:
        st.header("📄 Upload Syllabus")
        
        if analyzer.models and AI_MODELS_AVAILABLE:
            st.success("🤖 AI Models: Active")
        else:
            st.warning("⚠️ AI Models: Fallback Mode")
        
        input_method = st.radio("Choose input method:", ["Upload File", "Paste Text"])
        
        syllabus_text = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your syllabus",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT (Max size: 200MB)",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                file_hash = str(hash(uploaded_file.getvalue()))
                current_file_id = f"{uploaded_file.name}_{file_hash}"
                
                if st.session_state.get('current_file_id') != current_file_id:
                    st.session_state.current_file_id = current_file_id
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.analysis_data = None
                    st.session_state.current_text = ""
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_results = None
                    st.info(f"📄 New file detected: {uploaded_file.name}")
                
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                
                if file_size_mb > 200:
                    st.error(f"File too large ({file_size_mb:.1f}MB). Please use a file smaller than 200MB.")
                    st.stop()
                
                if not st.session_state.current_text or st.session_state.get('current_file_id') != current_file_id:
                    with st.spinner("Extracting text from file..."):
                        syllabus_text = analyzer.extract_text_from_file(uploaded_file)
                        st.session_state.current_text = syllabus_text
                else:
                    syllabus_text = st.session_state.current_text
                    
                if syllabus_text:
                    st.success(f"✅ File ready! ({file_size_mb:.1f}MB)")
                    word_count = len(syllabus_text.split())
                    char_count = len(syllabus_text)
                    st.info(f"📊 {word_count:,} words, {char_count:,} characters")
                    
                    preview_text = syllabus_text[:500] + "..." if len(syllabus_text) > 500 else syllabus_text
                    st.text_area("Preview:", preview_text, height=150)
        
        else:
            new_text = st.text_area(
                "Paste your syllabus content:",
                height=300,
                placeholder="Paste your course syllabus content here...",
                key="text_input"
            )
            
            if new_text != st.session_state.current_text:
                st.session_state.current_text = new_text
                st.session_state.analysis_data = None
                st.session_state.quiz_questions = None
                st.session_state.quiz_results = None
                st.session_state.current_file_id = f"pasted_text_{hash(new_text)}"
                st.session_state.current_file_name = "pasted_text"
                if new_text:
                    st.info("📄 New text content detected")
            
            syllabus_text = new_text
        
        st.markdown("---")
        st.subheader("🔧 Analysis Options")
        
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Quick Analysis", "Deep AI Analysis", "Comprehensive Analysis"],
            help="Choose analysis depth - deeper analysis takes more time but provides better insights"
        )
        
        include_advanced_features = st.checkbox(
            "Include Advanced AI Features",
            value=True,
            help="Enable semantic clustering, confidence scoring, and advanced question generation"
        )
        
        if st.button("🗑️ Clear All"):
            for key in ['current_file_id', 'current_file_name', 'analysis_data', 'current_text', 'quiz_questions', 'quiz_results']:
                st.session_state[key] = None if key in ['current_file_id', 'current_file_name', 'analysis_data', 'quiz_questions', 'quiz_results'] else ""
            st.rerun()
        
        analyze_button = st.button("🤖 Analyze with AI", type="primary", disabled=not syllabus_text)
    
    if analyze_button and syllabus_text:
        analysis_start_time = datetime.now()
        
        with st.spinner("🤖 Running AI analysis... Please wait."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Preprocessing text...")
            progress_bar.progress(0.2)
            
            status_text.text("🧠 Extracting topics with Sentence-BERT...")
            progress_bar.progress(0.4)
            
            status_text.text("📊 Generating summary with BART...")
            progress_bar.progress(0.6)
            
            status_text.text("❓ Creating questions with T5...")
            progress_bar.progress(0.8)
            
            status_text.text("📊 Finalizing analysis...")
            progress_bar.progress(0.9)
            
            analysis = analyzer.analyze_syllabus_with_ai(syllabus_text)
            st.session_state.analysis_data = analysis
            
            progress_bar.progress(1.0)
            status_text.text("✅ AI analysis completed!")
            
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            
        if analysis:
            st.success(f"✅ AI Analysis completed in {analysis_time:.1f} seconds!")
            
            if 'ai_metrics' in analysis:
                metrics = analysis['ai_metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Text Length", f"{metrics['text_length']:,} chars")
                with col2:
                    st.metric("Topics Found", metrics['topics_extracted'])
                with col3:
                    st.metric("Questions Generated", metrics['questions_generated'])
                with col4:
                    st.metric("Processing", metrics['processing_method'])
    
    # Display results
    if st.session_state.analysis_data:
        analysis = st.session_state.analysis_data
        
        if st.session_state.current_file_name:
            confidence = analysis['summary'].get('ai_confidence', 0.8)
            st.info(f"📄 Currently analyzing: **{st.session_state.current_file_name}** | AI Confidence: **{confidence:.1%}**")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📚 Summary", "❓ Expected Questions", "📅 Study Plan", "📝 Assignments", "🧠 MCQ Quiz", "📊 Quiz Results"])
        
        with tab1:
            st.header("📚 Course Summary")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ai_badge = "🤖 AI-Generated" if analysis['summary'].get('ai_confidence', 0) > 0.7 else "📊 Basic Analysis"
                
                st.markdown(f"""
                <div class="feature-card">
                    <h3>{analysis['summary']['title']} <span class="ai-badge">{ai_badge}</span></h3>
                    <p>{analysis['summary']['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 Key Topics (AI-Extracted)")
                for i, topic in enumerate(analysis['summary']['key_topics'], 1):
                    st.write(f"{i}. **{topic}**")
            
            with col2:
                confidence = analysis['summary'].get('ai_confidence', 0.8)
                confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                
                st.markdown(f"""
                <div class="feature-card">
                    <h4>Course Details</h4>
                    <p><strong>Duration:</strong> {analysis['summary']['duration']}</p>
                    <p><strong>Difficulty:</strong> {analysis['summary']['difficulty']}</p>
                    <p><strong>AI Confidence:</strong> <span style="color: {confidence_color}; font-weight: bold;">{confidence:.1%}</span></p>
                    <p><strong>Prerequisites:</strong></p>
                    <ul>
                """, unsafe_allow_html=True)
                
                for prereq in analysis['summary']['prerequisites']:
                    st.markdown(f"<li>{prereq}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with tab2:
            st.header("❓ AI-Generated Expected Questions")
            st.write("Questions generated using T5 transformer model based on your content:")
            
            for category in analysis['expected_questions']:
                confidence = category.get('confidence', 0.8)
                confidence_badge = f"<span class='confidence-score'>{confidence:.1%} confidence</span>"
                
                with st.expander(f"📖 {category['category']} ({len(category['questions'])} questions) {confidence_badge}", expanded=True):
                    for i, question in enumerate(category['questions'], 1):
                        ai_indicator = "🤖" if category.get('confidence', 0) > 0.8 else "📊"
                        st.markdown(f"""
                        <div class="question-item">
                            <strong>{ai_indicator} Q{i}:</strong> {question}
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab3:
            st.header("📅 AI-Optimized Study Plan")
            st.write(f"**Total Duration:** {analysis['study_plan']['total_weeks']} weeks")
            
            with st.expander("💡 AI-Enhanced Study Tips & Best Practices"):
                for tip in analysis['study_plan']['study_tips']:
                    st.write(f"• {tip}")
            
            st.subheader("📅 Weekly Schedule")
            
            for week_data in analysis['study_plan']['weekly_schedule']:
                ai_badge = "🤖 AI-Optimized" if week_data.get('ai_generated') else "📊 Standard"
                
                st.markdown(f"""
                <div class="week-plan">
                    <h4>📅 Week {week_data['week']} <span class="ai-badge">{ai_badge}</span></h4>
                    <p><strong>Time Commitment:</strong> {week_data['time_commitment']}</p>
                    <p><strong>Topics:</strong> {', '.join(week_data['topics'])}</p>
                    <p><strong>Learning Goals:</strong></p>
                    <ul>
                """, unsafe_allow_html=True)
                
                for goal in week_data['learning_goals']:
                    st.markdown(f"<li>{goal}</li>", unsafe_allow_html=True)
                
                st.markdown("<p><strong>Recommended Activities:</strong></p><ul>", unsafe_allow_html=True)
                
                for activity in week_data['activities']:
                    st.markdown(f"<li>{activity}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with tab4:
            st.header("📝 AI-Generated Practice Assignments")
            st.write("Intelligent assignments created based on analysis of your content:")
            
            for assignment in analysis['assignments']:
                difficulty_color = {
                    'Easy': 'green', 'Medium': 'orange', 'Hard': 'red',
                    'Beginner': 'green', 'Intermediate': 'orange', 'Advanced': 'red'
                }.get(assignment['difficulty'], 'blue')
                
                ai_indicator = "🤖 AI-Generated" if assignment.get('ai_generated') else "📊 Template-Based"
                
                st.markdown(f"""
                <div class="assignment-card">
                    <h4>{assignment['title']} <span class="ai-badge">{ai_indicator}</span></h4>
                    <p><span style="background-color: {difficulty_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">{assignment['difficulty']}</span> 
                    &nbsp;&nbsp;⏱️ {assignment['estimated_time']} &nbsp;&nbsp;📅 Week {assignment['week']}</p>
                    <p>{assignment['description']}</p>
                    <p><strong>Tasks:</strong></p>
                    <ul>
                """, unsafe_allow_html=True)
                
                for task in assignment['tasks']:
                    st.markdown(f"<li>{task}</li>", unsafe_allow_html=True)
                
                if 'skills_developed' in assignment:
                    st.markdown(f"<p><strong>Skills Developed:</strong> {', '.join(assignment['skills_developed'])}</p>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with tab5:
            st.header("🧠 Interactive MCQ Quiz")
            st.write("Test your knowledge with AI-generated multiple-choice questions based on your syllabus content!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_questions = st.selectbox("Number of questions:", [5, 10, 15, 20], index=1)
            
            with col2:
                difficulty_filter = st.selectbox("Difficulty:", ["All", "Easy", "Medium", "Hard"])
            
            with col3:
                randomize = st.checkbox("Randomize questions", value=True, help="Generate different questions each time")
            
            if st.button("🎯 Generate New Quiz", type="primary"):
                with st.spinner("🤖 Generating MCQ questions from your syllabus..."):
                    st.session_state.quiz_results = None
                    
                    questions = analyzer.generate_quiz_questions(
                        analysis.get('syllabus_text', ''), 
                        analysis['summary']['key_topics'], 
                        num_questions,
                        randomize
                    )
                    
                    if difficulty_filter != "All":
                        questions = [q for q in questions if q['difficulty'] == difficulty_filter]
                    
                    if not questions:
                        st.warning(f"No questions found for difficulty level '{difficulty_filter}'. Try 'All' difficulty.")
                    else:
                        st.session_state.quiz_questions = questions
                        st.success(f"✅ Generated {len(questions)} questions!")
                        st.rerun()
            
            if st.session_state.quiz_questions:
                st.markdown("---")
                st.subheader("🎯 Take the Quiz")
                
                st.info("📝 Instructions: Select the best answer for each question. Click 'Submit Quiz' when you're done to see your results.")
                
                questions = st.session_state.quiz_questions
                difficulty_counts = {}
                topic_counts = {}
                
                for q in questions:
                    diff = q['difficulty']
                    topic = q['topic']
                    difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Difficulty Distribution:**")
                    for diff, count in difficulty_counts.items():
                        st.write(f"• {diff}: {count} questions")
                
                with col2:
                    st.write("**Topic Coverage:**")
                    for topic, count in list(topic_counts.items())[:5]:
                        short_topic = topic[:30] + "..." if len(topic) > 30 else topic
                        st.write(f"• {short_topic}: {count} questions")
                
                with st.form("mcq_quiz"):
                    user_answers = {}
                    
                    for i, question in enumerate(questions):
                        st.markdown(f"""
                        <div class="question-card">
                            <h4>Question {i+1} [{question['difficulty']}]</h4>
                            <p><strong>{question['question']}</strong></p>
                            <small>Topic: {question['topic']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        user_answer = st.radio(
                            f"Select your answer for Question {i+1}:",
                            options=range(len(question['options'])),
                            format_func=lambda x, opts=question['options']: f"{chr(65+x)}. {opts[x]}",
                            key=f"question_{i}",
                            label_visibility="collapsed",
                            index=None 
                        )
                        user_answers[f"question_{i}"] = user_answer
                    
                    submit_quiz = st.form_submit_button("Submit Quiz", type="primary")
                
                if submit_quiz:
                    with st.spinner("Evaluating your quiz..."):
                        results = analyzer.quiz_evaluator.evaluate_quiz(
                            st.session_state.quiz_questions, 
                            user_answers
                        )
                        st.session_state.quiz_results = results
                    
                    st.success("Quiz evaluated! Check the Quiz Results tab.")
                    st.balloons()
            
            else:
                st.info("Click 'Generate New Quiz' to create your personalized MCQ test!")
                
                st.markdown("""
                ### MCQ Quiz Features:
                
                #### **AI-Powered Question Generation**
                - Questions are dynamically created from your actual syllabus content
                - Each question is contextually relevant to your course material
                - Smart difficulty assessment for each question
                
                #### **Dynamic & Randomized**
                - Generate different questions every time (when randomization is enabled)
                - Question order is shuffled for varied practice
                - Option order is randomized to prevent pattern memorization
                
                #### **Customizable Settings**
                - Choose number of questions (5-20)
                - Filter by difficulty level
                - Enable/disable randomization
                
                #### **Smart Evaluation**
                - Instant percentage scoring
                - Letter grade calculation (A-F)
                - Detailed performance analytics
                - Topic-wise breakdown of results
                """)
        
        with tab6:
            st.header("Quiz Results & Performance Analysis")
            
            if st.session_state.quiz_results:
                results = st.session_state.quiz_results
                
                score = results['score_percentage']
                grade = results['grade']
                
                score_color = "#28a745" if score >= 80 else "#ffc107" if score >= 60 else "#dc3545"
                
                st.markdown(f"""
                <div class="score-card" style="background: linear-gradient(135deg, {score_color}55 0%, {score_color}88 100%); border: 3px solid {score_color};">
                    <h2>Your Quiz Results</h2>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin: 1rem 0;">
                        <div style="text-align: center;">
                            <h1 style="font-size: 3rem; margin: 0; color: {score_color};">{score:.1f}%</h1>
                            <h3 style="margin: 0;">Grade: {grade}</h3>
                        </div>
                        <div style="text-align: center;">
                            <h3 style="margin: 0;">{results['correct_answers']}/{results['total_questions']}</h3>
                            <p style="margin: 0;">Correct Answers</p>
                        </div>
                    </div>
                    <p style="font-size: 1.1rem; margin: 1rem 0;">
                        {"Excellent work!" if score >= 90 else "Great job!" if score >= 80 else "Good effort!" if score >= 60 else "Keep studying!"}
                    </p>
                    <p style="font-size: 0.9rem; opacity: 0.8;">
                        Status: {"PASSED" if results['passed'] else "NEEDS IMPROVEMENT"} (Pass: 60%+)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("AI-Powered Performance Insights")
                for insight in results['insights']:
                    st.write(f"• {insight}")
                
                st.subheader("Detailed Question Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    show_filter = st.selectbox("Show:", ["All Questions", "Correct Only", "Incorrect Only"])
                with col2:
                    topic_filter = st.selectbox("Filter by Topic:", ["All Topics"] + list(set(r['topic'] for r in results['detailed_results'])))
                
                filtered_results = results['detailed_results']
                
                if show_filter == "Correct Only":
                    filtered_results = [r for r in filtered_results if r['is_correct']]
                elif show_filter == "Incorrect Only":
                    filtered_results = [r for r in filtered_results if not r['is_correct']]
                
                if topic_filter != "All Topics":
                    filtered_results = [r for r in filtered_results if r['topic'] == topic_filter]
                
                for i, result in enumerate(filtered_results):
                    status_icon = "✅" if result['is_correct'] else "❌"
                    status_color = "#d4edda" if result['is_correct'] else "#f8d7da"
                    
                    with st.expander(f"{status_icon} Question {i+1}: {result['question'][:50]}...", expanded=False):
                        st.markdown(f"""
                        <div style="background: {status_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <p><strong>Question:</strong> {result['question']}</p>
                            <p><strong>Options:</strong></p>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        for j, option in enumerate(result['options']):
                            is_user_answer = j == result['user_answer']
                            is_correct_answer = j == result['correct_answer']
                            
                            if is_correct_answer:
                                indicator = " (Correct Answer)"
                                style = "color: green; font-weight: bold;"
                            elif is_user_answer:
                                indicator = " (Your Answer)" if not is_correct_answer else " (Your Answer - Correct!)"
                                style = "color: red; font-weight: bold;" if not is_correct_answer else "color: green; font-weight: bold;"
                            else:
                                indicator = ""
                                style = ""
                            
                            st.markdown(f'<li style="{style}">{chr(65+j)}. {option}{indicator}</li>', unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            </ul>
                            <p><strong>Topic:</strong> {result['topic']} | <strong>Difficulty:</strong> {result['difficulty']}</p>
                            <p><strong>Explanation:</strong> {result['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.subheader("Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download Detailed Report"):
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        report = f"""# Quiz Results Report
Generated: {timestamp}
Course: {analysis['summary']['title']}

## Overall Performance
- Score: {results['score_percentage']:.1f}%
- Grade: {results['grade']}
- Correct Answers: {results['correct_answers']}/{results['total_questions']}
- Status: {"PASSED" if results['passed'] else "FAILED"}

## Performance Insights
"""
                        for insight in results['insights']:
                            report += f"- {insight}\n"
                        
                        report += "\n## Question Details\n"
                        for i, result in enumerate(results['detailed_results'], 1):
                            status = "CORRECT" if result['is_correct'] else "INCORRECT"
                            user_ans = chr(65 + result['user_answer']) if result['user_answer'] >= 0 else 'Not answered'
                            correct_ans = chr(65 + result['correct_answer'])
                            
                            report += f"""
### Question {i} [{status}]
**Question:** {result['question']}
**Your Answer:** {user_ans}
**Correct Answer:** {correct_ans}
**Topic:** {result['topic']}
**Difficulty:** {result['difficulty']}
**Explanation:** {result['explanation']}

**All Options:**
"""
                            for j, option in enumerate(result['options']):
                                report += f"{chr(65+j)}. {option}\n"
                            
                            report += "\n---\n"
                        
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"quiz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                
                with col2:
                    if st.button("Generate New Quiz"):
                        st.session_state.quiz_questions = None
                        st.session_state.quiz_results = None
                        st.rerun()
                
                with col3:
                    if st.button("Retake Same Quiz"):
                        st.session_state.quiz_results = None
                        st.rerun()
            
            else:
                st.info("Take the quiz first to see your detailed results here!")
                st.markdown("""
                ### What you'll get in your results:
                
                #### **Comprehensive Scoring**
                - **Percentage Score** - Exact percentage with letter grade (A-F)
                - **Pass/Fail Status** - Clear indication if you've met the 60% threshold
                - **Visual Score Card** - Color-coded results display
                
                #### **AI-Powered Insights**
                - **Performance Analysis** - AI evaluates your overall performance
                - **Strengths & Weaknesses** - Identify topics you've mastered vs. need work
                - **Study Recommendations** - Personalized suggestions for improvement
                
                #### **Detailed Question Review**
                - **Question-by-Question Breakdown** - See exactly what you got right/wrong
                - **Answer Explanations** - Understand why answers are correct
                - **Topic Analysis** - Performance breakdown by subject area
                - **Difficulty Assessment** - How you performed on easy vs. hard questions
                
                #### **Export & Tracking**
                - **Downloadable Report** - Comprehensive report for your records
                - **Retake Options** - Generate new questions or retake the same quiz
                - **Progress Tracking** - Compare performance across multiple attempts
                
                Take the quiz to unlock these detailed analytics and insights!
                """)
        
        # Export and reset options
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export & Reset")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("AI Summary", help="Download AI-enhanced summary"):
                confidence = analysis['summary'].get('ai_confidence', 0.8)
                summary_text = f"""# {analysis['summary']['title']} 
## AI-Generated Analysis Report

**AI Confidence Score:** {confidence:.1%}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Processing Method:** {analysis.get('ai_metrics', {}).get('processing_method', 'Standard')}

## Description
{analysis['summary']['description']}

## Course Details
- **Duration:** {analysis['summary']['duration']}
- **Difficulty:** {analysis['summary']['difficulty']} (AI-Assessed)
- **Prerequisites:** {', '.join(analysis['summary']['prerequisites'])}

## AI-Extracted Key Topics
"""
                for i, topic in enumerate(analysis['summary']['key_topics'], 1):
                    summary_text += f"{i}. {topic}\n"
                
                summary_text += f"\n---\n*Generated by AI-Powered Syllabus Analyzer using BART + T5 + Sentence-BERT*"
                
                st.download_button(
                    label="Download AI Summary",
                    data=summary_text,
                    file_name=f"ai_course_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("AI Questions", help="Download AI-generated questions"):
                questions_text = f"# AI-Generated Questions - {analysis['summary']['title']}\n\n"
                questions_text += f"**Generated using T5 Transformer Model**\n"
                questions_text += f"**Total Questions:** {sum(len(cat['questions']) for cat in analysis['expected_questions'])}\n\n"
                
                for category in analysis['expected_questions']:
                    confidence = category.get('confidence', 0.8)
                    questions_text += f"## {category['category']} (Confidence: {confidence:.1%})\n\n"
                    for i, question in enumerate(category['questions'], 1):
                        questions_text += f"{i}. {question}\n"
                    questions_text += "\n"
                
                questions_text += f"\n---\n*Generated by T5 Question Generation Model*"
                
                st.download_button(
                    label="Download AI Questions",
                    data=questions_text,
                    file_name=f"ai_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        if st.sidebar.button("Complete AI Analysis", help="Download full AI analysis with metadata"):
            analysis_with_metadata = analysis.copy()
            analysis_with_metadata['generation_info'] = {
                "generated_at": datetime.now().isoformat(),
                "ai_models_used": ["BART", "T5", "Sentence-BERT"] if analyzer.models and AI_MODELS_AVAILABLE else ["Fallback"],
                "confidence_score": analysis['summary'].get('ai_confidence', 0.8),
                "processing_time_estimate": "30-60 seconds"
            }
            
            if 'syllabus_text' in analysis_with_metadata:
                del analysis_with_metadata['syllabus_text']
            
            json_string = json.dumps(analysis_with_metadata, indent=2)
            st.sidebar.download_button(
                label="Download Complete Analysis",
                data=json_string,
                file_name=f"ai_syllabus_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        if st.sidebar.button("Reset for New Analysis"):
            for key in ['current_file_id', 'current_file_name', 'analysis_data', 'current_text', 'quiz_questions', 'quiz_results']:
                st.session_state[key] = None if key in ['current_file_id', 'current_file_name', 'analysis_data', 'quiz_questions', 'quiz_results'] else ""
            st.rerun()
    
    else:
        st.markdown("""
        ## Welcome to Enhanced AI-Powered Syllabus Analyzer!
        
        Transform any course syllabus using **state-of-the-art AI models** plus **Interactive MCQ Testing**!
        
        ### Core AI-Powered Features:
        - **BART Summarization**: Intelligent content summarization and key insight extraction
        - **T5 Question Generation**: Automatically generate exam-style questions from your content  
        - **Sentence-BERT Analysis**: Semantic topic clustering and content understanding
        - **ML Difficulty Assessment**: AI-powered difficulty and prerequisite analysis
        
        ### **NEW: Interactive MCQ Quiz System**
        - **Dynamic Question Generation**: AI creates unique multiple-choice questions from your syllabus
        - **Smart Randomization**: Get different questions every time for varied practice  
        - **Instant Evaluation**: Comprehensive scoring with detailed performance analytics
        - **Personalized Insights**: AI-powered analysis of strengths and improvement areas
        - **Progress Tracking**: Detailed reports and performance monitoring
        
        ### What You Get:
        - **Smart Summary**: AI-generated course overview with confidence scoring
        - **Contextual Questions**: Transformer-generated practice questions
        - **Optimized Schedule**: AI-enhanced weekly learning plans  
        - **Intelligent Assignments**: Content-aware project suggestions
        - **MCQ Testing**: Interactive quizzes with instant feedback and analytics
        - **Performance Reports**: Detailed analysis with improvement recommendations
        
        ### How the MCQ System Works:
        1. **Upload/Analyze** your syllabus with AI
        2. **Generate Quiz** - Choose number of questions, difficulty, randomization
        3. **Take the Test** - Interactive multiple-choice interface
        4. **Get Results** - Instant scoring with detailed analytics
        5. **Track Progress** - Download reports and generate new quizzes
        6. **Continuous Practice** - Each quiz is unique with randomized questions
        
        ### MCQ Quiz Features:
        - **Dynamic Content**: Questions generated from YOUR specific syllabus content
        - **Difficulty Assessment**: Automatic categorization (Easy/Medium/Hard)
        - **Topic Coverage**: Balanced questions across all course topics  
        - **Smart Evaluation**: Percentage scoring, letter grades, pass/fail status
        - **Performance Analytics**: Identify strengths, weaknesses, and study priorities
        - **Export Options**: Download detailed performance reports
        
        ### Advanced Options:
        - Choose analysis depth (Quick/Deep/Comprehensive)
        - Enable advanced AI features for better insights
        - Customize quiz length (5-20 questions)
        - Filter questions by difficulty level
        - Enable/disable randomization for varied practice
        - Real-time confidence scoring for AI-generated content
        
        **Ready to experience the future of AI-powered learning?** Upload your syllabus in the sidebar!
        
        ---
        *Enhanced with BART + T5 + Sentence-BERT + Dynamic MCQ Generation & Evaluation*
        """)
        
        if st.session_state.get('models_loaded'):
            if analyzer.models and AI_MODELS_AVAILABLE:
                st.success("AI Models Status: Fully Loaded and Ready (Including MCQ Generation)")
            else:
                st.warning("AI Models Status: Fallback Mode (Limited AI features, basic MCQ available)")

if __name__ == "__main__":
    main()