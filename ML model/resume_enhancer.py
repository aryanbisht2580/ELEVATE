import torch
import torch.nn as nn
from model import Seq2SeqModel, Encoder, Decoder, Vocabulary, generate_text, generate_text_greedy, generate_text_with_attention
import pickle


class ResumeEnhancer:
    """Class to handle resume enhancement using the trained model"""
    
    def __init__(self, model_path='resume_enhancement_model.pth'):
        self.model_path = model_path
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vocabulary"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract vocab and model parameters
            self.vocab = checkpoint['vocab']
            
            # Initialize model with saved parameters
            encoder = Encoder(
                vocab_size=self.vocab.vocab_size,
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            )
            
            decoder = Decoder(
                vocab_size=self.vocab.vocab_size,
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            )
            
            self.model = Seq2SeqModel(encoder, decoder, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found.")
            print("Please train the model first using: python resume_enhancement_train.py")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def enhance_resume_section(self, resume_text, job_description, method='greedy', max_length=100):
        """Enhance a resume section based on job description"""
        # Combine resume text and job description with separator
        combined_input = f"{resume_text} [SEP] {job_description}"
        
        # Generate enhanced text using the specified method
        if method == 'greedy':
            enhanced_text = generate_text_greedy(self.model, combined_input, self.vocab, max_length)
        elif method == 'sampling':
            enhanced_text = generate_text(self.model, combined_input, self.vocab, max_length, temperature=0.8)
        elif method == 'diverse':
            enhanced_text = generate_text_with_attention(self.model, combined_input, self.vocab, max_length)
        else:
            # Default to greedy
            enhanced_text = generate_text_greedy(self.model, combined_input, self.vocab, max_length)
        
        return enhanced_text
    
    def enhance_multiple_sections(self, resume_sections, job_description, method='greedy'):
        """Enhance multiple resume sections based on job description"""
        enhanced_sections = {}
        
        for section_name, section_text in resume_sections.items():
            enhanced_sections[section_name] = self.enhance_resume_section(
                section_text, job_description, method
            )
        
        return enhanced_sections


def demo_usage():
    """Demonstrate how to use the ResumeEnhancer class"""
    print("Resume Enhancement Demo")
    print("="*40)
    
    try:
        # Initialize the enhancer
        enhancer = ResumeEnhancer()
        
        # Sample resume section and job description
        resume_section = "Worked on web development projects using Python and JavaScript"
        job_description = "Seeking full-stack developer with experience in Python, JavaScript, and modern web frameworks"
        
        print(f"Original Resume Section: {resume_section}")
        print(f"Job Description: {job_description}")
        
        # Enhance using different methods
        enhanced_greedy = enhancer.enhance_resume_section(resume_section, job_description, method='greedy')
        enhanced_sampling = enhancer.enhance_resume_section(resume_section, job_description, method='sampling')
        
        print(f"\nEnhanced (Greedy): {enhanced_greedy}")
        print(f"Enhanced (Sampling): {enhanced_sampling}")
        
        # Example with multiple sections
        resume_sections = {
            "experience": "Worked on software development projects",
            "skills": "Experience with Python and JavaScript",
            "projects": "Built web applications during college"
        }
        
        print(f"\nEnhancing multiple sections:")
        enhanced_sections = enhancer.enhance_multiple_sections(
            resume_sections, 
            "Full-stack developer position requiring Python, JavaScript, and web development skills"
        )
        
        for section_name, enhanced_text in enhanced_sections.items():
            print(f"  {section_name.title()}: {enhanced_text}")
            
    except FileNotFoundError:
        print("Model not found. Please train the model first using: python resume_enhancement_train.py")


def enhance_resume_for_job(resume_content, job_description, output_file="enhanced_resume.txt"):
    """Function to enhance an entire resume for a specific job"""
    try:
        enhancer = ResumeEnhancer()
        
        # Split resume into sections (this is a simple approach, you might want to improve this)
        sections = ["Experience", "Skills", "Projects", "Education", "Summary"]
        enhanced_resume = []
        
        print(f"Enhancing resume for job: {job_description[:50]}...")
        
        # For demonstration, we'll enhance the entire resume as one block
        # In practice, you'd want to split it into sections
        enhanced_content = enhancer.enhance_resume_section(
            resume_content, 
            job_description, 
            method='greedy',
            max_length=200
        )
        
        enhanced_resume.append(f"Enhanced for Job: {job_description}")
        enhanced_resume.append("="*50)
        enhanced_resume.append(f"Enhanced Content: {enhanced_content}")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(enhanced_resume))
        
        print(f"Enhanced resume saved to {output_file}")
        return enhanced_content
        
    except Exception as e:
        print(f"Error enhancing resume: {e}")
        return None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_usage()
    else:
        print("Resume Enhancer")
        print("Usage: python resume_enhancer.py [demo]")
        demo_usage()