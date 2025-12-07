import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import pickle


class Vocabulary:
    """Simple vocabulary class to convert text to indices and vice versa"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Initialize with special tokens
        self.add_word(self.pad_token)
        self.add_word(self.unk_token)
        self.add_word(self.start_token)
        self.add_word(self.end_token)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        return self.word2idx[word]
    
    def build_vocabulary(self, texts):
        # Add all unique words from texts to vocabulary
        for text in texts:
            for word in text.split():
                self.add_word(word)
    
    def encode(self, text, max_length=None):
        """Convert text to sequence of indices"""
        words = text.split()
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        
        # Add start and end tokens
        indices = [self.word2idx[self.start_token]] + indices + [self.word2idx[self.end_token]]
        
        if max_length:
            if len(indices) < max_length:
                indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices):
        """Convert sequence of indices back to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx[self.end_token]:
                break
            elif idx != self.word2idx[self.pad_token] and idx != self.word2idx[self.start_token]:
                words.append(self.idx2word[idx])
        return ' '.join(words)


class Encoder(nn.Module):
    """Encoder for the input text (basic/resume text)"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)


class Decoder(nn.Module):
    """Decoder for the refined text"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_hidden):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, encoder_hidden)
        predictions = self.fc(self.dropout(output))
        return predictions, (hidden, cell)


class Seq2SeqModel(nn.Module):
    """Sequence to Sequence model for text refinement"""
    
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Prepare input for decoder (start with <START> token)
        input_token = trg[:, 0].unsqueeze(1)  # First token is <START>
        
        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input_token, (hidden, cell))
            outputs[:, t] = output.squeeze(1)
            
            # Decide whether to use teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = output.argmax(2)
            
            # Use either predicted token or actual next token as next input
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs


class ResumeRefinementDataset(Dataset):
    """Dataset class for resume refinement"""
    
    def __init__(self, basic_texts, refined_texts, vocab, max_length=128):
        self.basic_texts = basic_texts
        self.refined_texts = refined_texts
        self.vocab = vocab
        self.max_length = max_length
        
        # Encode all texts
        self.encoded_basic = [vocab.encode(text, max_length) for text in basic_texts]
        self.encoded_refined = [vocab.encode(text, max_length) for text in refined_texts]
    
    def __len__(self):
        return len(self.basic_texts)
    
    def __getitem__(self, idx):
        basic = torch.tensor(self.encoded_basic[idx], dtype=torch.long)
        refined = torch.tensor(self.encoded_refined[idx], dtype=torch.long)
        return basic, refined


def create_sample_data():
    """Create sample data for demonstration purposes"""
    basic_texts = [
        "Worked on projects", 
        "Did some coding work", 
        "Managed team of people", 
        "Created web apps",
        "Developed software",
        "Worked on some stuff",
        "Helped with projects",
        "Built applications"
    ]
    
    refined_texts = [
        "Developed and executed multiple software projects to meet business requirements, resulting in improved operational efficiency", 
        "Implemented complex coding solutions that enhanced system performance and reduced processing time by 30%", 
        "Led cross-functional teams of 5+ members to successfully deliver projects on time and within budget", 
        "Architected and deployed scalable web applications serving 10K+ daily users",
        "Designed and implemented software solutions using modern technologies to improve system efficiency",
        "Contributed to diverse projects while ensuring high-quality deliverables",
        "Collaborated with stakeholders to enhance project outcomes and deliverables",
        "Engineered robust applications with focus on performance and scalability"
    ]
    
    return basic_texts, refined_texts


def train_model(model, dataloader, optimizer, criterion, device):
    """Training function for the model"""
    model.train()
    total_loss = 0
    
    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # Reshape output and target for loss calculation
        output = output[:, 1:].reshape(-1, output.shape[2])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device, vocab):
    """Evaluation function for the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing during evaluation
            
            output = output[:, 1:].reshape(-1, output.shape[2])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_text(model, input_text, vocab, max_length=50, temperature=1.0):
    """Generate refined text from input text with optional temperature for diversity"""
    model.eval()
    
    # Encode input text
    with torch.no_grad():
        input_indices = torch.tensor([vocab.encode(input_text, max_length)], dtype=torch.long).to(model.device)
        
        # Encode the input with the encoder
        encoder_outputs, (hidden, cell) = model.encoder(input_indices)
        
        # Start with the start token for the decoder
        decoder_input = torch.tensor([[vocab.word2idx[vocab.start_token]]], dtype=torch.long).to(model.device)
        
        output_indices = []
        
        for _ in range(max_length):
            output, (hidden, cell) = model.decoder(decoder_input, (hidden, cell))
            
            # Apply temperature for more diverse outputs (lower = more deterministic)
            output = output / temperature
            
            # Get probabilities and sample from them
            probs = torch.softmax(output, dim=-1)
            predicted_token = torch.multinomial(probs.view(-1), 1).unsqueeze(0)
            
            output_indices.append(predicted_token.item())
            
            # Use the predicted token as the next input
            decoder_input = predicted_token
            
            # Stop if we've reached the end token
            if predicted_token.item() == vocab.word2idx[vocab.end_token]:
                break
        
        # Decode the output indices back to text
        output_text = vocab.decode(output_indices)
        
    return output_text


def generate_text_greedy(model, input_text, vocab, max_length=50):
    """Generate refined text from input text using greedy approach (most likely tokens)"""
    model.eval()
    
    # Encode input text
    with torch.no_grad():
        input_indices = torch.tensor([vocab.encode(input_text, max_length)], dtype=torch.long).to(model.device)
        
        # Encode the input with the encoder
        encoder_outputs, (hidden, cell) = model.encoder(input_indices)
        
        # Start with the start token for the decoder
        decoder_input = torch.tensor([[vocab.word2idx[vocab.start_token]]], dtype=torch.long).to(model.device)
        
        output_indices = []
        
        for _ in range(max_length):
            output, (hidden, cell) = model.decoder(decoder_input, (hidden, cell))
            
            # Get the most likely next token (greedy approach)
            predicted_token = output.argmax(2)
            output_indices.append(predicted_token.item())
            
            # Use the predicted token as the next input
            decoder_input = predicted_token
            
            # Stop if we've reached the end token
            if predicted_token.item() == vocab.word2idx[vocab.end_token]:
                break
        
        # Decode the output indices back to text
        output_text = vocab.decode(output_indices)
        
    return output_text


def generate_text_with_attention(model, input_text, vocab, max_length=50):
    """Generate refined text from input text with beam search approach (simplified)"""
    model.eval()
    
    # For now using a modified sampling approach, full beam search would be more complex
    with torch.no_grad():
        input_indices = torch.tensor([vocab.encode(input_text, max_length)], dtype=torch.long).to(model.device)
        
        # Encode the input with the encoder
        encoder_outputs, (hidden, cell) = model.encoder(input_indices)
        
        # Start with the start token for the decoder
        decoder_input = torch.tensor([[vocab.word2idx[vocab.start_token]]], dtype=torch.long).to(model.device)
        
        output_indices = []
        
        for _ in range(max_length):
            output, (hidden, cell) = model.decoder(decoder_input, (hidden, cell))
            
            # Get top-k predictions for more diverse outputs
            top_k = 5
            top_k_probs, top_k_indices = torch.topk(output, top_k, dim=-1)
            
            # Sample from top-k options
            probs = torch.softmax(top_k_probs, dim=-1)
            choice_idx = torch.multinomial(probs.view(-1), 1)
            predicted_token = top_k_indices.view(-1)[choice_idx]
            
            output_indices.append(predicted_token.item())
            
            # Use the predicted token as the next input
            decoder_input = predicted_token.unsqueeze(0).unsqueeze(0)
            
            # Stop if we've reached the end token
            if predicted_token.item() == vocab.word2idx[vocab.end_token]:
                break
        
        # Decode the output indices back to text
        output_text = vocab.decode(output_indices)
        
    return output_text
