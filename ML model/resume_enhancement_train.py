import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from model import Seq2SeqModel, Encoder, Decoder, Vocabulary, train_model, evaluate_model, generate_text


class ParaNMTDataset(Dataset):
    """Dataset class for para-nmt dataset"""
    
    def __init__(self, file_path, vocab, max_length=128, num_samples=None):
        self.vocab = vocab
        self.max_length = max_length
        
        print(f"Loading para-nmt dataset from {file_path}")
        
        # Load para-nmt data
        self.para_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:  # We need at least two parts (two paraphrased sentences)
                    sent1 = parts[0]
                    sent2 = parts[1]
                    
                    # Add both directions: sent1 -> sent2 and sent2 -> sent1
                    self.para_pairs.append((sent1, sent2))
                    self.para_pairs.append((sent2, sent1))
        
        print(f"Loaded {len(self.para_pairs)} paraphrase pairs")
        
        # Encode all texts
        self.encoded_src = [vocab.encode(pair[0], max_length) for pair in self.para_pairs]
        self.encoded_trg = [vocab.encode(pair[1], max_length) for pair in self.para_pairs]
    
    def __len__(self):
        return len(self.para_pairs)
    
    def __getitem__(self, idx):
        src = torch.tensor(self.encoded_src[idx], dtype=torch.long)
        trg = torch.tensor(self.encoded_trg[idx], dtype=torch.long)
        return src, trg


class ResumeJobDataset(Dataset):
    """Dataset for resume enhancement using job descriptions"""
    
    def __init__(self, resume_texts, job_descs, enhanced_texts, vocab, max_length=100):
        self.resume_texts = resume_texts
        self.job_descs = job_descs
        self.enhanced_texts = enhanced_texts
        self.vocab = vocab
        self.max_length = max_length
        
        # Create combined inputs (resume + job description)
        self.combined_inputs = []
        for resume, job_desc in zip(resume_texts, job_descs):
            # Combine resume and job description with a separator
            combined = resume + " [SEP] " + job_desc
            self.combined_inputs.append(combined)
        
        # Encode all texts
        self.encoded_inputs = [vocab.encode(text, max_length*2) for text in self.combined_inputs]
        self.encoded_targets = [vocab.encode(text, max_length) for text in enhanced_texts]
    
    def __len__(self):
        return len(self.resume_texts)
    
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.encoded_inputs[idx], dtype=torch.long)
        target_seq = torch.tensor(self.encoded_targets[idx], dtype=torch.long)
        return input_seq, target_seq


def load_para_nmt_data(file_path, vocab, max_samples=50000, max_length=128):
    """Load para-nmt data for general paraphrasing capability"""
    print(f"Loading para-nmt data: {max_samples} samples...")
    
    dataset = ParaNMTDataset(file_path, vocab, max_length, num_samples=max_samples)
    return dataset


def create_resume_job_data():
    """Create sample resume-job matching data for fine-tuning"""
    # Sample resume snippets
    resume_texts = [
        "Worked on projects", "Did coding work", "Managed team", "Created web apps",
        "Developed software", "Worked on stuff", "Helped with projects", "Built applications",
        "Experience with Python", "Knowledge in machine learning", "Work with databases",
        "Team collaboration skills", "Project management", "Software development",
        "Web development experience", "Problem solving abilities", "Communication skills"
    ]
    
    # Corresponding job descriptions
    job_descs = [
        "Looking for experienced developers who can lead projects and deliver results",
        "Seeking software engineers with strong programming skills in modern languages", 
        "Need team leaders who can manage and coordinate with cross-functional teams",
        "Hiring web developers to build scalable applications for our platform",
        "Searching for software developers with experience in full-stack development",
        "Looking for contributors who can tackle challenging assignments",
        "Seeking collaborative professionals who assist in achieving team goals",
        "Wanting application builders with focus on performance and maintainability",
        "Python developer position requiring advanced programming skills",
        "ML engineer role requiring deep learning and AI expertise",
        "Database administrator role with SQL and optimization experience",
        "Collaboration-focused role requiring excellent communication skills",
        "Project manager position requiring leadership capabilities",
        "Software engineering role with development lifecycle experience",
        "Frontend developer with responsive design experience",
        "Analytical problem solver with debugging expertise",
        "Client-facing role requiring interpersonal communication"
    ]
    
    # Enhanced resume phrases that match the job descriptions
    enhanced_texts = [
        "Successfully led multiple projects from conception to deployment, delivering measurable business results",
        "Implemented sophisticated software solutions using modern programming languages and frameworks",
        "Orchestrated cross-functional teams to achieve complex project objectives and exceed expectations",
        "Engineered scalable web applications serving thousands of users with optimal performance",
        "Designed and developed full-stack software solutions using industry best practices",
        "Tackled technically challenging assignments while maintaining high-quality deliverables",
        "Collaborated effectively with cross-functional teams to drive project success and innovation",
        "Built high-performance applications with emphasis on scalability and code maintainability",
        "Extensive experience developing enterprise-level applications using Python and related technologies",
        "Deep expertise in machine learning algorithms, neural networks, and artificial intelligence systems",
        "Comprehensive experience with database design, optimization, and management systems",
        "Proven track record of effective collaboration with diverse teams and stakeholders",
        "Demonstrated project management expertise with successful delivery of complex initiatives",
        "Full software development lifecycle experience with agile methodologies",
        "Proficient in responsive web development techniques and modern frontend frameworks",
        "Strong analytical and problem-solving skills with proven debugging expertise",
        "Excellent communication abilities with experience in client engagement and relationship building"
    ]
    
    return resume_texts, job_descs, enhanced_texts


def prepare_vocabulary(file_path, max_samples=10000):
    """Build vocabulary from para-nmt dataset"""
    print("Building vocabulary from para-nmt dataset...")
    
    all_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                all_texts.append(parts[0])
                all_texts.append(parts[1])
    
    # Create vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary(all_texts)
    print(f"Vocabulary built with size: {vocab.vocab_size}")
    
    return vocab


def train_resume_enhancer():
    """Main training function for resume enhancement model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # File path for para-nmt dataset
    para_nmt_file = 'para-nmt-50m.txt'
    
    # Prepare vocabulary from para-nmt dataset
    vocab = prepare_vocabulary(para_nmt_file, max_samples=10000)
    
    # Load para-nmt dataset for general paraphrasing
    para_dataset = load_para_nmt_data(para_nmt_file, vocab, max_samples=5000, max_length=64)
    para_dataloader = DataLoader(para_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Create resume-job matching dataset
    resume_texts, job_descs, enhanced_texts = create_resume_job_data()
    resume_job_dataset = ResumeJobDataset(resume_texts, job_descs, enhanced_texts, vocab, max_length=50)
    resume_job_dataloader = DataLoader(resume_job_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Model parameters
    vocab_size = vocab.vocab_size
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    
    print(f"Initializing model with vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
    
    # Initialize encoder and decoder
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    
    # Initialize model
    model = Seq2SeqModel(encoder, decoder, device).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
    
    # Phase 1: Pre-train on para-nmt dataset (general paraphrasing)
    print("\n=== PHASE 1: Pre-training on para-nmt dataset ===")
    num_pretrain_epochs = 5
    
    for epoch in range(num_pretrain_epochs):
        train_loss = train_model(model, para_dataloader, optimizer, criterion, device)
        print(f'Pre-train Epoch {epoch+1}/{num_pretrain_epochs}, Training Loss: {train_loss:.4f}')
        
        if (epoch + 1) % 2 == 0:
            eval_loss = evaluate_model(model, para_dataloader, criterion, device, vocab)
            print(f'Pre-train Epoch {epoch+1}/{num_pretrain_epochs}, Evaluation Loss: {eval_loss:.4f}')
    
    # Phase 2: Fine-tune on resume-job matching data
    print("\n=== PHASE 2: Fine-tuning on resume-job matching data ===")
    num_finetune_epochs = 15
    
    for epoch in range(num_finetune_epochs):
        train_loss = train_model(model, resume_job_dataloader, optimizer, criterion, device)
        print(f'Fine-tune Epoch {epoch+1}/{num_finetune_epochs}, Training Loss: {train_loss:.4f}')
        
        if (epoch + 1) % 5 == 0:
            eval_loss = evaluate_model(model, resume_job_dataloader, criterion, device, vocab)
            print(f'Fine-tune Epoch {epoch+1}/{num_finetune_epochs}, Evaluation Loss: {eval_loss:.4f}')
    
    print('\nTraining completed!')
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }, 'resume_enhancement_model.pth')
    
    print("Model saved as 'resume_enhancement_model.pth'")
    
    # Test the model with examples
    print("\n=== Testing the trained model ===")
    
    test_cases = [
        ("Experience with Python [SEP] Python developer position requiring advanced programming skills", "Enhanced Python"),
        ("Team collaboration skills [SEP] Collaboration-focused role requiring excellent communication skills", "Enhanced Teamwork"),
        ("Software development [SEP] Looking for experienced developers who can lead projects", "Enhanced Development")
    ]
    
    for test_input, description in test_cases:
        generated_text = generate_text(model, test_input, vocab, max_length=75)
        print(f"\nTest: {description}")
        print(f"Input: {test_input}")
        print(f"Generated: {generated_text}")


def load_and_test_model():
    """Load the trained model and test it"""
    try:
        # Load the saved model
        checkpoint = torch.load('resume_enhancement_model.pth', map_location=torch.device('cpu'))
        
        # Reconstruct the vocabulary and model
        vocab = checkpoint['vocab']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with saved parameters
        encoder = Encoder(
            vocab_size=vocab.vocab_size,
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        )
        
        decoder = Decoder(
            vocab_size=vocab.vocab_size,
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        )
        
        model = Seq2SeqModel(encoder, decoder, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print("Model loaded successfully!")
        
        # Test cases for resume enhancement
        test_inputs = [
            "Worked on projects [SEP] Seeking project manager with leadership experience",
            "Python experience [SEP] Python developer with Django framework experience needed",
            "Team collaboration [SEP] Looking for team player with communication skills",
            "Web development [SEP] Full-stack developer with React and Node.js experience"
        ]
        
        for test_input in test_inputs:
            generated_text = generate_text(model, test_input, vocab, max_length=100)
            print(f"\nInput: {test_input}")
            print(f"Generated: {generated_text}")
        
    except FileNotFoundError:
        print("Model file not found. Please run the training script first.")
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        load_and_test_model()
    else:
        train_resume_enhancer()