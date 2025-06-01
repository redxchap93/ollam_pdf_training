import os
import sys
import re
import json
import time
import logging
import subprocess
import platform
from typing import List, Dict, Set
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import tqdm
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Precompile regex patterns
WHITESPACE_RE = re.compile(r'\s+')
LINEBREAK_RE = re.compile(r'\n+')
SPECIAL_CHARS_RE = re.compile(r'[^\w\s\.\!\?\,\;\:\-\(\)]')
WORD_RE = re.compile(r'\b[A-Z][a-z]+\b|\b\w{4,}\b')

# Check dependencies
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("⚠️ PyPDF2 not available")

try:
    import fitz
    import pymupdf
    ADVANCED_PDF = True
except ImportError:
    ADVANCED_PDF = False
    logger.warning("⚠️ PyMuPDF not available")

try:
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    ADVANCED_NLP = True
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    ADVANCED_NLP = False
    STOP_WORDS = set()
    logger.warning("⚠️ NLTK not available")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ Ollama not available")

def install_pdf_dependencies():
    """Install required dependencies with system checks."""
    print("🔧 Installing dependencies...")
    
    if platform.system() == 'Linux':
        try:
            subprocess.run(['dpkg', '-s', 'libmupdf-dev'], capture_output=True, text=True, timeout=30)
        except subprocess.CalledProcessError:
            print("⚠️ Installing libmupdf-dev...")
            subprocess.run(['apt-get', 'update'], check=True, capture_output=True, text=True, timeout=120)
            subprocess.run(['apt-get', 'install', '-y', 'libmupdf-dev', 'python3-dev'], check=True, capture_output=True, text=True, timeout=300)
            print("✅ Installed system dependencies")
    
    for package in ['PyPDF2', 'pymupdf', 'nltk', 'tqdm']:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], capture_output=True, text=True, timeout=120)
            print(f"✅ {package} installed")
        except Exception as e:
            print(f"⚠️ Error installing {package}: {e}")
    
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK resources downloaded")
    except Exception as e:
        print(f"⚠️ Error downloading NLTK resources: {e}")
    
    global PDF_AVAILABLE, ADVANCED_PDF, ADVANCED_NLP, fitz, pymupdf
    try:
        from PyPDF2 import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
    try:
        import fitz
        import pymupdf
        ADVANCED_PDF = True
    except ImportError:
        ADVANCED_PDF = False
    try:
        from nltk.tokenize import sent_tokenize
        from nltk.corpus import stopwords
        ADVANCED_NLP = True
        global STOP_WORDS
        STOP_WORDS = set(stopwords.words('english'))
    except ImportError:
        ADVANCED_NLP = False

def extract_page_text(page_num, pdf_path):
    """Extract text from a single PDF page with block analysis."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        blocks = page.get_text("blocks")
        doc.close()
        text = ""
        for block in blocks:
            if block[6] == 0:  # Text block
                text += block[4] + "\n"
        return text
    except Exception as e:
        logger.warning(f"Error extracting page {page_num}: {e}")
        return ""

class PDFBookProcessor:
    def __init__(self):
        self.book_title = ""
        self.book_keywords = set()
        self.book_entities = set()
        self.book_phrases = set()
        self.full_text = ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text with parallel processing and block analysis."""
        logger.info(f"Extracting text from {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if ADVANCED_PDF:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            logger.info(f"Processing {total_pages} pages...")
            raw_text = ""
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(extract_page_text, i, pdf_path) for i in range(total_pages)]
                for i, future in enumerate(tqdm.tqdm(futures, desc="Pages", unit="page")):
                    raw_text += future.result()
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{total_pages} pages")
        elif PDF_AVAILABLE:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                logger.info(f"Processing {total_pages} pages...")
                raw_text = ""
                for i in tqdm.tqdm(range(total_pages), desc="Pages", unit="page"):
                    raw_text += reader.pages[i].extract_text() + "\n"
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{total_pages} pages")
        else:
            raise ImportError("No PDF processing libraries available")
        
        logger.info(f"✅ Extracted {len(raw_text)} characters")
        self.full_text = raw_text
        return raw_text
    
    def build_book_vocabulary(self, text: str):
        """Build comprehensive vocabulary from the book."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Extract significant words (3+ characters, not common stop words)
        significant_words = [w for w in words if len(w) >= 3 and w not in STOP_WORDS]
        word_freq = Counter(significant_words)
        
        # Top frequent words as keywords
        self.book_keywords = set([word for word, freq in word_freq.most_common(200) if freq >= 3])
        
        # Extract proper nouns and entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        self.book_entities = set([entity.lower() for entity in entities])
        
        # Extract meaningful phrases (2-3 word combinations)
        words_list = text.split()
        for i in range(len(words_list) - 1):
            phrase = f"{words_list[i]} {words_list[i+1]}".lower()
            if len(phrase) > 6 and not any(stop in phrase for stop in STOP_WORDS):
                self.book_phrases.add(phrase)
        
        logger.info(f"✅ Built vocabulary: {len(self.book_keywords)} keywords, {len(self.book_entities)} entities, {len(self.book_phrases)} phrases")
    
    def clean_and_structure_text(self, raw_text: str) -> Dict:
        """Clean and structure text with improved paragraph detection."""
        logger.info("Cleaning and structuring text...")
        
        if not raw_text:
            return {'paragraphs': [], 'sentences': [], 'chunks': [], 'full_text': ''}
        
        # Build book vocabulary for better filtering
        self.build_book_vocabulary(raw_text)
        
        cleaned_text = WHITESPACE_RE.sub(' ', raw_text)
        cleaned_text = LINEBREAK_RE.sub('\n', cleaned_text)
        cleaned_text = SPECIAL_CHARS_RE.sub(' ', cleaned_text)
        
        paragraphs = []
        current_para = []
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if len(line) > 10:
                current_para.append(line)
            if len(current_para) >= 3 or (line.endswith('.') and current_para):
                paragraphs.append(' '.join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(' '.join(current_para))
        paragraphs = [p for p in paragraphs if len(p) > 50]
        
        all_sentences = []
        if ADVANCED_NLP:
            try:
                for paragraph in paragraphs:
                    sentences = sent_tokenize(paragraph)
                    all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
            except Exception:
                for paragraph in paragraphs:
                    sentences = re.split(r'[.!?]+', paragraph)
                    all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        else:
            for paragraph in paragraphs:
                sentences = re.split(r'[.!?]+', paragraph)
                all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        chunks = []
        chunk_size = 3
        for i in range(0, len(all_sentences), chunk_size):
            chunk = ' '.join(all_sentences[i:i+chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
        
        logger.info(f"✅ Processed: {len(paragraphs)} paragraphs, {len(all_sentences)} sentences, {len(chunks)} chunks")
        return {
            'paragraphs': paragraphs[:200],
            'sentences': all_sentences[:500],
            'chunks': chunks[:100],
            'full_text': cleaned_text
        }
    
    def extract_key_terms_from_chunk(self, chunk: str) -> List[str]:
        """Extract key terms that exist in the book vocabulary."""
        words = re.findall(r'\b\w+\b', chunk.lower())
        key_terms = [w for w in words if w in self.book_keywords]
        
        # Also check for entities and phrases
        for entity in self.book_entities:
            if entity in chunk.lower():
                key_terms.append(entity)
        
        for phrase in list(self.book_phrases)[:50]:  # Limit phrases to avoid too many
            if phrase in chunk.lower():
                key_terms.append(phrase)
        
        return list(set(key_terms))[:5]  # Return top 5 unique terms
    
    def create_book_specific_answer(self, chunk: str, term: str, book_title: str) -> str:
        """Create an answer strictly from the book content."""
        sentences = re.split(r'[.!?]+', chunk)
        relevant = [s.strip() for s in sentences if term.lower() in s.lower() and len(s.strip()) > 15]
        
        if not relevant:
            relevant = [s.strip() for s in sentences if len(s.strip()) > 15][:2]
        
        answer = ' '.join(relevant[:3])  # Limit to 3 sentences max
        
        if len(answer) < 80:
            # Try to add more context from the chunk
            all_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            answer = ' '.join(all_sentences[:2])
        
        # Ensure the answer is substantial and book-related
        if len(answer) < 50:
            answer = f"Based on the book '{book_title}', {answer}"
        
        return answer[:600]  # Limit answer length
    
    def create_comprehensive_qa_pairs(self, structured_content: Dict, book_title: str) -> List[Dict]:
        """Create comprehensive Q&A pairs strictly from book content."""
        logger.info("Creating comprehensive Q&A pairs...")
        
        chunks = structured_content['chunks']
        qa_pairs = []
        
        # Add greeting responses
        greetings = [
            {
                "question": "hello",
                "answer": f"Hello! I'm your specialized assistant for '{book_title}'. I can only answer questions about the content of this specific book. What would you like to know?"
            },
            {
                "question": "hi",
                "answer": f"Hi there! I'm trained exclusively on '{book_title}'. Please ask me questions about the book's content."
            },
            {
                "question": "what can you help me with",
                "answer": f"I can help you with questions about '{book_title}'. I only have knowledge from this specific book and cannot answer questions about other topics."
            }
        ]
        qa_pairs.extend(greetings)
        
        # Generate questions from chunks
        question_patterns = [
            "what is {}",
            "who was {}",
            "when did {} happen",
            "why did {} occur",
            "how did {} develop",
            "what happened to {}",
            "describe {}",
            "explain {}",
            "tell me about {}"
        ]
        
        processed_terms = set()
        
        for chunk in tqdm.tqdm(chunks, desc="Processing chunks", unit="chunk"):
            if len(chunk) < 80:
                continue
            
            key_terms = self.extract_key_terms_from_chunk(chunk)
            
            for term in key_terms:
                if term in processed_terms:
                    continue
                processed_terms.add(term)
                
                for pattern in question_patterns:
                    question = pattern.format(term)
                    answer = self.create_book_specific_answer(chunk, term, book_title)
                    
                    if len(answer) > 50:  # Only add substantial answers
                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })
        
        # Add out-of-scope responses
        rejection_responses = [
            {
                "question": "what is artificial intelligence",
                "answer": "I'm sorry, but I can only provide information from the book I was trained on. I don't have knowledge about topics outside of this specific book."
            },
            {
                "question": "how do I code in python",
                "answer": "I'm specialized only in the content of one specific book and cannot help with programming or other general topics."
            },
            {
                "question": "what is the weather today",
                "answer": "I can only answer questions about the book I was trained on. I don't have access to current information or general knowledge."
            }
        ]
        qa_pairs.extend(rejection_responses)
        
        # Remove duplicates and ensure uniqueness
        unique_qa = []
        seen_questions = set()
        
        for qa in qa_pairs:
            question_normalized = re.sub(r'\W+', '', qa['question'].lower())
            if question_normalized not in seen_questions and len(qa['answer']) > 30:
                seen_questions.add(question_normalized)
                unique_qa.append(qa)
        
        # Limit to reasonable number
        unique_qa = unique_qa[:800]
        
        # Save Q&A pairs
        os.makedirs("qa_datasets", exist_ok=True)
        qa_path = f"qa_datasets/{book_title.lower().replace(' ', '_')}_qa.json"
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(unique_qa, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(unique_qa)} Q&A pairs to {qa_path}")
        
        # Log sample Q&A
        print("\n📋 Sample Q&A Pairs:")
        for qa in unique_qa[:5]:
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer'][:100]}...")
            print("-" * 50)
        
        return unique_qa

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a PDF and create a Q&A dataset."""
        self.book_title = re.sub(r'book4me\.org', '', os.path.splitext(os.path.basename(pdf_path))[0]).replace('_', ' ').strip().title()
        print(f"\n📚 Processing: {self.book_title}")
        
        print("📖 Extracting text...")
        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(raw_text)} characters")
        
        print("🔧 Structuring content...")
        structured_content = self.clean_and_structure_text(raw_text)
        
        print("🧠 Creating Q&A pairs...")
        qa_pairs = self.create_comprehensive_qa_pairs(structured_content, self.book_title)
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        return qa_pairs

class StrictBookModel:
    def __init__(self):
        self.model_name = ""
        self.dataset = []
        self.book_keywords = set()
        self.book_entities = set()
        self.book_title = ""
    
    def select_base_model(self) -> str:
        """Select the best available base model."""
        try:
            models = ollama.list()
            print(f"🔍 Available models: {models}")
            
            # Handle different response formats
            if isinstance(models, dict) and 'models' in models:
                available = [m.get('name', m.get('model', '')) for m in models['models']]
            elif isinstance(models, list):
                available = [m.get('name', m.get('model', '')) for m in models]
            else:
                available = []
            
            available = [name for name in available if name]  # Remove empty names
            print(f"📋 Parsed available models: {available}")
            
            # Prefer smaller, more controllable models
            preferred_models = [
                'llama3.2:3b',
                'qwen2.5:3b', 
                'llama3.1:8b',
                'qwen2.5:7b',
                'llama3.2:1b',
                'llama3.2',
                'llama3.1',
                'qwen2.5'
            ]
            
            for model in preferred_models:
                if model in available:
                    logger.info(f"✅ Selected base model: {model}")
                    return model
            
            # Fallback to first available
            if available:
                selected = available[0]
                logger.info(f"✅ Using first available model: {selected}")
                return selected
            else:
                # If no models available, try to pull a default one
                print("⚠️ No models found. Attempting to pull llama3.2:3b...")
                try:
                    subprocess.run(["ollama", "pull", "llama3.2:3b"], check=True, timeout=300)
                    return "llama3.2:3b"
                except:
                    print("❌ Failed to pull default model. Using llama3.2:3b anyway.")
                    return "llama3.2:3b"
                
        except Exception as e:
            logger.warning(f"Error selecting model: {e}")
            print("⚠️ Using fallback model: llama3.2:3b")
            return "llama3.2:3b"
    
    def create_strict_model(self, dataset: List[Dict], book_title: str):
        """Create a model that ONLY responds to book-related queries."""
        logger.info(f"Creating strict book-only model")
        
        self.dataset = dataset
        self.book_title = book_title
        self.model_name = f"book_only_{book_title.lower().replace(' ', '_')}"
        
        # Extract book vocabulary from dataset
        all_text = " ".join([qa['question'] + " " + qa['answer'] for qa in dataset])
        words = re.findall(r'\b\w+\b', all_text.lower())
        self.book_keywords = set([w for w in words if len(w) >= 3 and w not in STOP_WORDS])
        
        # Extract entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)
        self.book_entities = set([entity.lower() for entity in entities])
        
        base_model = self.select_base_model()
        
        print(f"\n🔧 Creating Model:")
        print(f"   Base model: {base_model}")
        print(f"   Specialized model: {self.model_name}")
        print(f"   Dataset size: {len(dataset)} entries")
        print(f"   Book vocabulary: {len(self.book_keywords)} terms")
        
        # Remove existing model
        try:
            subprocess.run(["ollama", "rm", self.model_name], capture_output=True, text=True)
            logger.info(f"Removed existing model: {self.model_name}")
        except Exception:
            pass
        
        # Create training data with strict instructions
        os.makedirs("fine_tune_data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced Modelfile with very strict instructions
        modelfile_content = f'''FROM {base_model}
PARAMETER temperature 0.1
PARAMETER top_p 0.7
PARAMETER top_k 10
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.15

SYSTEM """You are a specialized AI assistant trained EXCLUSIVELY on the book "{book_title}". 

CRITICAL RULES:
1. You can ONLY answer questions about the content of "{book_title}"
2. For ANY question not related to this specific book, you MUST respond: "I can only provide information from the book '{book_title}'. I don't have knowledge about other topics."
3. Do NOT use any general knowledge or information from outside this book
4. Do NOT make assumptions or provide information not explicitly in the book
5. Always base your answers strictly on the book's content
6. If you're unsure whether a question relates to the book, err on the side of caution and use the standard rejection response

You may respond to basic greetings like "hello" or "hi" by introducing yourself as an assistant for this specific book, but immediately redirect to book-related topics.

Remember: You are NOT a general AI assistant. You are a specialized assistant for one specific book only."""
'''

        modelfile_path = f"fine_tune_data/{self.model_name}_Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        # Create the model
        try:
            logger.info("Creating specialized model...")
            print(f"🔧 Running: ollama create {self.model_name} -f {modelfile_path}")
            
            # First, verify the base model exists
            print(f"🔍 Verifying base model '{base_model}' exists...")
            try:
                test_result = subprocess.run(
                    ["ollama", "show", base_model],
                    capture_output=True, text=True, timeout=30
                )
                if test_result.returncode != 0:
                    print(f"⚠️ Base model '{base_model}' not found. Attempting to pull...")
                    pull_result = subprocess.run(
                        ["ollama", "pull", base_model],
                        capture_output=True, text=True, timeout=300
                    )
                    if pull_result.returncode != 0:
                        print(f"❌ Failed to pull {base_model}: {pull_result.stderr}")
                        # Try with a simpler model name
                        base_model = "llama3.2"
                        print(f"🔄 Trying with simpler model name: {base_model}")
                        
                        # Update the Modelfile with new base model
                        modelfile_content = f'''FROM {base_model}
PARAMETER temperature 0.1
PARAMETER top_p 0.7
PARAMETER top_k 10
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.15

SYSTEM """You are a specialized AI assistant trained EXCLUSIVELY on the book "{book_title}". 

CRITICAL RULES:
1. You can ONLY answer questions about the content of "{book_title}"
2. For ANY question not related to this specific book, you MUST respond: "I can only provide information from the book '{book_title}'. I don't have knowledge about other topics."
3. Do NOT use any general knowledge or information from outside this book
4. Do NOT make assumptions or provide information not explicitly in the book
5. Always base your answers strictly on the book's content
6. If you're unsure whether a question relates to the book, err on the side of caution and use the standard rejection response

You may respond to basic greetings like "hello" or "hi" by introducing yourself as an assistant for this specific book, but immediately redirect to book-related topics.

Remember: You are NOT a general AI assistant. You are a specialized assistant for one specific book only."""
'''
                        with open(modelfile_path, 'w', encoding='utf-8') as f:
                            f.write(modelfile_content)
                else:
                    print(f"✅ Base model '{base_model}' verified")
            except subprocess.TimeoutExpired:
                print("⚠️ Timeout checking base model, proceeding anyway...")
            
            # Now create the specialized model
            result = subprocess.run(
                ["ollama", "create", self.model_name, "-f", modelfile_path],
                capture_output=True, text=True, timeout=600  # Increased timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Model creation successful: {result.stdout}")
                if result.stderr:
                    logger.info(f"Model creation notes: {result.stderr}")
                print(f"✅ Model '{self.model_name}' created successfully!")
            else:
                logger.error(f"Model creation failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                
                # Try to diagnose the issue
                if "no such file" in result.stderr.lower():
                    print(f"❌ Modelfile not found. Checking path: {modelfile_path}")
                    if os.path.exists(modelfile_path):
                        print("✅ Modelfile exists")
                        with open(modelfile_path, 'r') as f:
                            content = f.read()
                            print(f"📄 Modelfile content:\n{content[:200]}...")
                    else:
                        print("❌ Modelfile missing!")
                        raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")
                
                elif "unknown model" in result.stderr.lower():
                    print(f"❌ Base model '{base_model}' not available")
                    # List available models for debugging
                    list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    print(f"Available models:\n{list_result.stdout}")
                    raise Exception(f"Base model '{base_model}' not available")
                
                else:
                    print(f"❌ Unknown error creating model:")
                    print(f"Return code: {result.returncode}")
                    print(f"Error output: {result.stderr}")
                    print(f"Standard output: {result.stdout}")
                    raise Exception(f"Failed to create model: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Model creation timed out")
            print("❌ Model creation timed out (took longer than 10 minutes)")
            raise Exception("Model creation timed out")
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def is_book_related_query(self, query: str) -> bool:
        """Enhanced check if query is related to the book."""
        query_lower = query.lower().strip()
        
        # Allow basic greetings
        greetings = {'hello', 'hi', 'hey', 'greetings'}
        if query_lower in greetings:
            return True
        
        # Check for book-specific terms
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Must have intersection with book vocabulary
        if query_words & self.book_keywords:
            return True
        
        # Check for entities
        if query_words & self.book_entities:
            return True
        
        # Check if the book title is mentioned
        if self.book_title.lower() in query_lower:
            return True
        
        return False
    
    def test_model_strictness(self):
        """Comprehensive test of model's book-only behavior."""
        print(f"\n🧪 Testing Model Strictness: {self.model_name}")
        print("=" * 60)
        
        # Test cases: book-related and non-book-related
        test_cases = [
            # Greetings (should work)
            {"query": "hello", "should_answer": True, "category": "Greeting"},
            {"query": "hi there", "should_answer": True, "category": "Greeting"},
            
            # Book-related (should work) - using sample from dataset
            {"query": self.dataset[0]['question'] if self.dataset else "what is this about", 
             "should_answer": True, "category": "Book Content"},
            
            # General knowledge (should reject)
            {"query": "what is artificial intelligence", "should_answer": False, "category": "General AI"},
            {"query": "how do I cook pasta", "should_answer": False, "category": "Cooking"},
            {"query": "what is the capital of France", "should_answer": False, "category": "Geography"},
            {"query": "explain quantum physics", "should_answer": False, "category": "Science"},
            {"query": "write a poem about love", "should_answer": False, "category": "Creative Writing"},
            {"query": "how to program in Python", "should_answer": False, "category": "Programming"},
            {"query": "what is the weather today", "should_answer": False, "category": "Current Events"},
            {"query": "tell me a joke", "should_answer": False, "category": "Entertainment"},
        ]
        
        # Add more book-specific tests if we have dataset
        if len(self.dataset) > 3:
            for i in range(1, min(4, len(self.dataset))):
                test_cases.append({
                    "query": self.dataset[i]['question'],
                    "should_answer": True,
                    "category": "Book Content"
                })
        
        correct_responses = 0
        total_tests = len(test_cases)
        
        for i, test in enumerate(test_cases, 1):
            query = test['query']
            should_answer = test['should_answer']
            category = test['category']
            
            print(f"\nTest {i}/{total_tests} - {category}")
            print(f"Query: {query}")
            
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    options={"temperature": 0.1}
                )
                actual_response = response['message']['content'].strip()
                
                # Check if response indicates rejection
                rejection_indicators = [
                    "i can only provide information from the book",
                    "i don't have knowledge about other topics",
                    "i can only answer questions about",
                    "i'm specialized only in",
                    "i cannot help with",
                    "outside of this specific book"
                ]
                
                is_rejection = any(indicator in actual_response.lower() for indicator in rejection_indicators)
                
                if should_answer:
                    # Should provide book-related answer (not reject)
                    test_passed = not is_rejection and len(actual_response) > 20
                    status = "✅ PASS" if test_passed else "❌ FAIL"
                else:
                    # Should reject non-book queries
                    test_passed = is_rejection
                    status = "✅ PASS" if test_passed else "❌ FAIL"
                
                if test_passed:
                    correct_responses += 1
                
                print(f"Expected: {'Answer' if should_answer else 'Reject'}")
                print(f"Response: {actual_response[:100]}{'...' if len(actual_response) > 100 else ''}")
                print(f"Result: {status}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
        
        accuracy = (correct_responses / total_tests) * 100
        print(f"\n📊 STRICTNESS TEST RESULTS:")
        print(f"   Correct responses: {correct_responses}/{total_tests}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 80:
            print("✅ Model shows good book-only behavior!")
        else:
            print("⚠️ Model may need adjustment for better strictness")
        
        return accuracy
    
    def interactive_chat(self):
        """Interactive chat with the specialized model."""
        print(f"\n💬 Interactive Chat with {self.model_name}")
        print("=" * 50)
        print(f"This model only knows about: {self.book_title}")
        print("Type 'exit' to quit, 'test' to run strictness test")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'test':
                    self.test_model_strictness()
                    continue
                elif not user_input:
                    continue
                
                # Get response from model
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_input}],
                    options={"temperature": 0.1}
                )
                
                print(f"\n{self.model_name}: {response['message']['content']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function to create a strict book-only AI model."""
    print("🏆 STRICT BOOK-ONLY AI MODEL CREATOR")
    print("=" * 50)
    print("This creates an AI model that ONLY responds to questions about your specific book!")
    
    # Check dependencies
    if not ADVANCED_PDF or not PDF_AVAILABLE:
        print("\n⚠️ Missing PDF processing dependencies!")
        if input("Install now? (y/n): ").strip().lower() == 'y':
            install_pdf_dependencies()
        else:
            print("❌ PDF processing required. Exiting.")
            return
    
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama is required but not available.")
        print("Please install Ollama first: https://ollama.ai")
        sys.exit(1)
    
    # Test Ollama connection
    try:
        test_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if test_result.returncode != 0:
            print("❌ Ollama is not running or not accessible")
            print("Please start Ollama service and try again")
            sys.exit(1)
        else:
            print("✅ Ollama connection verified")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Cannot connect to Ollama")
        print("Please ensure Ollama is installed and running")
        sys.exit(1)
    
    print(f"\n📋 Dependencies Status:")
    print(f"   PyPDF2: {'✅' if PDF_AVAILABLE else '❌'}")
    print(f"   PyMuPDF: {'✅' if ADVANCED_PDF else '❌'}")
    print(f"   NLTK: {'✅' if ADVANCED_NLP else '❌'}")
    print(f"   Ollama: {'✅' if OLLAMA_AVAILABLE else '❌'}")
    
    # Get PDF file path
    pdf_path = input("\n📁 Enter the path to your PDF book: ").strip()
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Check file size
    pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"📊 PDF size: {pdf_size_mb:.1f} MB")
    
    if pdf_size_mb > 100:
        print("⚠️ Large PDF detected. This may take a while to process.")
        if input("Continue? (y/n): ").strip().lower() != 'y':
            return
    
    start_time = time.time()
    
    try:
        # Step 1: Process PDF and create dataset
        print("\n🔄 Step 1: Processing PDF and creating Q&A dataset...")
        processor = PDFBookProcessor()
        dataset = processor.process_pdf(pdf_path)
        
        if not dataset:
            print("❌ Failed to create dataset from PDF")
            return
        
        # Step 2: Create specialized model
        print("\n🔄 Step 2: Creating specialized book-only model...")
        model_creator = StrictBookModel()
        model_creator.create_strict_model(dataset, processor.book_title)
        
        processing_time = time.time() - start_time
        print(f"\n🎉 SUCCESS! Model created in {processing_time:.1f} seconds")
        print(f"📚 Book: {processor.book_title}")
        print(f"🤖 Model: {model_creator.model_name}")
        print(f"📊 Dataset: {len(dataset)} Q&A pairs")
        
        # Step 3: Test the model
        print("\n🔄 Step 3: Testing model strictness...")
        accuracy = model_creator.test_model_strictness()
        
        # Step 4: Interactive demo
        print(f"\n✅ Your specialized model '{model_creator.model_name}' is ready!")
        print("You can now use it with:")
        print(f"   ollama run {model_creator.model_name}")
        
        if input("\nWould you like to try the interactive chat? (y/n): ").strip().lower() == 'y':
            model_creator.interactive_chat()
        
        print(f"\n📝 Summary:")
        print(f"   Model name: {model_creator.model_name}")
        print(f"   Book processed: {processor.book_title}")
        print(f"   Dataset size: {len(dataset)} entries")
        print(f"   Strictness accuracy: {accuracy:.1f}%")
        print(f"   Processing time: {processing_time:.1f} seconds")
        
        print(f"\n🚀 To use your model anytime:")
        print(f"   ollama run {model_creator.model_name}")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"❌ An error occurred: {e}")
        return

if __name__ == "__main__":
    # Clean up any frontend conflicts
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'frontend', '-y'], 
                      capture_output=True, text=True)
    except:
        pass
    
    main()