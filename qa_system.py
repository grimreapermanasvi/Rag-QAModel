
import json
import re
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
import warnings
from datetime import datetime
import hashlib

# Suppress warnings
warnings.filterwarnings("ignore")

# Core imports
import torch

# Transformers and ML
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)

# PDF processing
from pdfminer.high_level import extract_text
import PyPDF2

# Vector similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processing with multiple extraction methods"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer"""
        try:
            return extract_text(pdf_path)
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {e}")
            return ""
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 as fallback"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with fallback methods"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try pdfminer first
        text = self.extract_text_pdfminer(pdf_path)
        
        # Fallback to PyPDF2 if pdfminer fails
        if not text.strip():
            logger.info("Falling back to PyPDF2...")
            text = self.extract_text_pypdf2(pdf_path)
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF")
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
        
        return chunks

class QAGenerator:
    """QA generation and answering"""
    
    def __init__(self):
        logger.info("Loading models...")
        try:
            # Question generation
            self.question_tokenizer = T5Tokenizer.from_pretrained('iarfmoose/t5-base-question-generator')
            self.question_model = T5ForConditionalGeneration.from_pretrained('iarfmoose/t5-base-question-generator')
            
            # Question answering
            self.qa_pipeline = pipeline(
                'question-answering',
                model='Intel/dynamic_tinybert',
                tokenizer='Intel/dynamic_tinybert',
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence embeddings for similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def clean_generated_questions(self, raw_output: str) -> List[str]:
        """Clean and validate generated questions"""
        # Basic cleaning
        cleaned = re.sub(r'\?{2,}', '?', raw_output)
        cleaned = re.sub(r'[^\w\s\?\.\,\-\(\)\'\"\:]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        potential_questions = []
        
        # Split by question markers and periods
        sentences = re.split(r'[.!?]+', cleaned)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Add question mark if missing but looks like a question
            if not sentence.endswith('?') and self._looks_like_question(sentence):
                sentence += '?'
            
            if self.is_valid_question(sentence):
                potential_questions.append(sentence)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(potential_questions))

    def _looks_like_question(self, sentence: str) -> bool:
        """Check if sentence looks like a question"""
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'whose']
        auxiliary_verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'will', 'would']
        
        sentence_lower = sentence.lower()
        starts_with_question = any(sentence_lower.startswith(word) for word in question_words)
        starts_with_auxiliary = any(sentence_lower.startswith(verb) for verb in auxiliary_verbs)
        
        return starts_with_question or starts_with_auxiliary

    def is_valid_question(self, question: str) -> bool:
        """Validate if text is a proper question"""
        question = question.strip()
        
        # Length check
        if len(question) < 10 or len(question) > 200:
            return False
        
        # Must end with question mark
        if not question.endswith('?'):
            return False
        
        # Check for question patterns
        question_patterns = [
            r'\b(what|how|when|where|why|who|which|whose)\b',
            r'\b(is|are|was|were|do|does|did|can|could|will|would|should|may|might)\b.*\?',
            r'^(is|are|was|were|do|does|did|can|could|will|would|should|may|might)\b'
        ]
        
        if not any(re.search(pattern, question.lower()) for pattern in question_patterns):
            return False
        
        # Word count and diversity checks
        words = question.split()
        if len(words) < 4 or len(words) > 30:
            return False
        
        # Check for sufficient word diversity
        if len(set(words)) < len(words) * 0.6:
            return False
        
        # Avoid questions that are too generic
        generic_patterns = [
            r'^what is (this|that|it)\?$',
            r'^how (is|are) (this|that|it)\?$'
        ]
        
        if any(re.match(pattern, question.lower()) for pattern in generic_patterns):
            return False
        
        return True

    def generate_questions_from_chunk(self, chunk: str, max_questions: int = 3) -> List[str]:
        """Generate questions from text chunk"""
        if len(chunk.strip()) < 50:
            return []
        
        try:
            input_text = f"generate questions: {chunk}"
            encoding = self.question_tokenizer(
                input_text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            outputs = self.question_model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_length=150,
                num_beams=4,
                num_return_sequences=min(max_questions, 3),
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.question_tokenizer.eos_token_id
            )
            
            questions = []
            for output in outputs:
                raw_question = self.question_tokenizer.decode(output, skip_special_tokens=True)
                cleaned_questions = self.clean_generated_questions(raw_question)
                questions.extend(cleaned_questions)
            
            # Remove duplicates and limit
            unique_questions = list(dict.fromkeys(questions))[:max_questions]
            
            return unique_questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def extract_answer(self, question: str, context: str) -> Optional[Dict]:
        """Extract answer with confidence score"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            if result['score'] < 0.1:
                return None
            
            answer = result['answer'].strip()
            
            # Validate answer
            if len(answer) < 2 or len(answer) > 200:
                return None
            
            if not re.search(r'[a-zA-Z]', answer):
                return None
            
            return {
                'answer': answer,
                'confidence': result['score'],
                'start': result.get('start', 0),
                'end': result.get('end', len(answer))
            }
            
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return None

    def process_chunks_to_qa_pairs(self, chunks: List[str]) -> List[Dict]:
        """Process chunks into QA pairs"""
        qa_pairs = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")

            if len(chunk.strip()) < 50:
                continue

            questions = self.generate_questions_from_chunk(chunk)

            for question in questions:
                answer_result = self.extract_answer(question, chunk)
                if answer_result:
                    qa_pair = {
                        "id": hashlib.md5(f"{question}{answer_result['answer']}".encode()).hexdigest()[:10],
                        "chunk_id": i,
                        "question": question,
                        "answer": answer_result['answer'],
                        "confidence": answer_result['confidence'],
                        "context": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                        "chunk_full": chunk
                    }
                    qa_pairs.append(qa_pair)

        return qa_pairs

class PDFQASystem:
    """Simple PDF QA system"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.qa_generator = QAGenerator()
        self.qa_pairs = []
        self.embeddings = None

    def process_pdf(self, pdf_path: str, output_file: str = None) -> Dict:
        """Process PDF and generate QA pairs"""
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Extract text
            print("Extracting text from PDF...")
            text = self.pdf_processor.extract_text(pdf_path)
            
            # Create chunks
            print("Creating text chunks...")
            chunks = self.pdf_processor.chunk_text(text)
            print(f"Created {len(chunks)} chunks")
            
            # Generate QA pairs
            print("Generating QA pairs...")
            self.qa_pairs = self.qa_generator.process_chunks_to_qa_pairs(chunks)
            
            print(f"Generated {len(self.qa_pairs)} QA pairs")
            
            # Create embeddings for similarity search
            print("Creating embeddings for similarity search...")
            self._create_embeddings()
            
            # Save QA pairs if output file specified
            if output_file:
                self.save_qa_pairs(output_file)
                print(f"QA pairs saved to: {output_file}")
            
            return {
                "status": "success",
                "total_qa_pairs": len(self.qa_pairs),
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "message": str(e)}

    def _create_embeddings(self):
        """Create embeddings for similarity search"""
        if not self.qa_pairs:
            self.embeddings = None
            return
        
        # Create embeddings for questions
        questions = [qa["question"] for qa in self.qa_pairs]
        embeddings = self.qa_generator.sentence_model.encode(questions)
        
        self.embeddings = {
            "questions": questions,
            "embeddings": embeddings
        }

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """Query the QA system"""
        if not self.qa_pairs or not self.embeddings:
            print("Error: System not initialized. Please process a PDF first.")
            return []
        
        try:
            # Find similar questions using embeddings
            question_embedding = self.qa_generator.sentence_model.encode([question])
            similarities = cosine_similarity(question_embedding, self.embeddings["embeddings"])[0]
            
            # Get top-k similar questions
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                qa_pair = self.qa_pairs[idx]
                results.append({
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "confidence": qa_pair["confidence"],
                    "similarity": float(similarities[idx]),
                    "context": qa_pair["context"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying system: {e}")
            return []

    def save_qa_pairs(self, filepath: str):
        """Save QA pairs to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)

    def load_qa_pairs(self, filepath: str):
        """Load QA pairs from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        
        # Create embeddings
        self._create_embeddings()
        print(f"Loaded {len(self.qa_pairs)} QA pairs from {filepath}")

    def print_qa_pairs(self, limit: int = 10):
        """Print QA pairs"""
        if not self.qa_pairs:
            print("No QA pairs available.")
            return
        
        print(f"\n=== QA Pairs (showing first {min(limit, len(self.qa_pairs))}) ===")
        for i, qa in enumerate(self.qa_pairs[:limit]):
            print(f"\n{i+1}. Question: {qa['question']}")
            print(f"   Answer: {qa['answer']}")
            print(f"   Confidence: {qa['confidence']:.3f}")

def main():
    """Main function"""
    system = PDFQASystem()
    
    print("=== PDF QA System ===")
    print("Commands:")
    print("  process <pdf_path> [output_file] - Process PDF and generate QA pairs")
    print("  load <qa_file> - Load existing QA pairs")
    print("  query <question> [top_k] - Query the system")
    print("  show [limit] - Show QA pairs")
    print("  save <output_file> - Save current QA pairs")
    print("  quit - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
            
            if command[0] == "quit":
                break
            
            elif command[0] == "process":
                if len(command) < 2:
                    print("Usage: process <pdf_path> [output_file]")
                    continue
                
                pdf_path = command[1]
                output_file = command[2] if len(command) > 2 else None
                
                result = system.process_pdf(pdf_path, output_file)
                if result["status"] == "success":
                    print(f"Success! Generated {result['total_qa_pairs']} QA pairs from {result['total_chunks']} chunks")
                else:
                    print(f"Error: {result['message']}")
            
            elif command[0] == "load":
                if len(command) < 2:
                    print("Usage: load <qa_file>")
                    continue
                
                try:
                    system.load_qa_pairs(command[1])
                except Exception as e:
                    print(f"Error loading file: {e}")
            
            elif command[0] == "query":
                if len(command) < 2:
                    print("Usage: query <question> [top_k]")
                    continue
                
                question = ' '.join(command[1:])
                # Extract top_k if it's a number at the end
                parts = question.rsplit(' ', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    question = parts[0]
                    top_k = int(parts[1])
                else:
                    top_k = 5
                
                results = system.query(question, top_k)
                
                if results:
                    print(f"\n=== Top {len(results)} Results for: '{question}' ===")
                    for i, result in enumerate(results):
                        print(f"\n{i+1}. Q: {result['question']}")
                        print(f"   A: {result['answer']}")
                        print(f"   Similarity: {result['similarity']:.3f}")
                        print(f"   Confidence: {result['confidence']:.3f}")
                        print(f"   Context: {result['context']}")
                else:
                    print("No results found.")
            
            elif command[0] == "show":
                limit = int(command[1]) if len(command) > 1 and command[1].isdigit() else 10
                system.print_qa_pairs(limit)
            
            elif command[0] == "save":
                if len(command) < 2:
                    print("Usage: save <output_file>")
                    continue
                
                try:
                    system.save_qa_pairs(command[1])
                    print(f"QA pairs saved to: {command[1]}")
                except Exception as e:
                    print(f"Error saving file: {e}")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()