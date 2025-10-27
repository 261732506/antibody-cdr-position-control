"""
Sequence Generation Script

Generate antibody sequences with conditional control over CDR3 properties.

Usage:
    python generate_sequences.py --model_path checkpoint.pth --model_type mamba --num_sequences 100
"""

import torch
import argparse
from typing import List


class SequenceGenerator:
    """Generate antibody sequences with CDR position control"""

    def __init__(self, model_path: str, model_type: str, device: str = 'cuda'):
        self.device = device
        self.model_type = model_type

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        if model_type == 'mamba':
            from model_mamba import MambaModel
            self.model = MambaModel(**checkpoint['config'])
        else:
            from model_transformer import TransformerModel
            self.model = TransformerModel(**checkpoint['config'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.vocab = checkpoint['vocab']

    def generate(self, num_sequences: int, conditions: dict = None, max_length: int = 256) -> List[str]:
        """
        Generate antibody sequences

        Args:
            num_sequences: Number of sequences to generate
            conditions: Dictionary of conditions (e.g., {'hydrophobicity': 'hydrophobic', 'charge': 'positive'})
            max_length: Maximum sequence length

        Returns:
            List of generated sequences
        """
        sequences = []

        with torch.no_grad():
            for i in range(num_sequences):
                # Start with <bos> token
                input_ids = [self.vocab['<bos>']]

                # Add condition tokens if specified
                if conditions:
                    if 'hydrophobicity' in conditions:
                        input_ids.append(self.vocab[f'<{conditions["hydrophobicity"]}>'])
                    if 'charge' in conditions:
                        input_ids.append(self.vocab[f'<{conditions["charge"]}>'])

                # Autoregressive generation
                for _ in range(max_length):
                    input_tensor = torch.tensor([input_ids], device=self.device)
                    logits = self.model(input_tensor)

                    # Sample next token
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                    if next_token == self.vocab['<eos>']:
                        break

                    input_ids.append(next_token)

                # Convert token IDs to sequence
                sequence = self.decode_sequence(input_ids)
                sequences.append(sequence)

        return sequences

    def decode_sequence(self, token_ids: List[int]) -> str:
        """Convert token IDs to amino acid sequence"""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab[tid] for tid in token_ids if tid in inv_vocab]

        # Remove special tokens and markers
        sequence = ''.join([t for t in tokens if len(t) == 1 and t.isalpha()])
        return sequence


def main():
    parser = argparse.ArgumentParser(description='Generate antibody sequences')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', required=True, choices=['mamba', 'transformer'])
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--hydrophobicity', choices=['hydrophobic', 'hydrophilic', 'neutral'], help='Hydrophobicity condition')
    parser.add_argument('--charge', choices=['positive', 'negative', 'neutral'], help='Charge condition')
    parser.add_argument('--output', required=True, help='Output FASTA file')
    args = parser.parse_args()

    # Initialize generator
    generator = SequenceGenerator(args.model_path, args.model_type)

    # Prepare conditions
    conditions = {}
    if args.hydrophobicity:
        conditions['hydrophobicity'] = args.hydrophobicity
    if args.charge:
        conditions['charge'] = args.charge

    # Generate sequences
    print(f"Generating {args.num_sequences} sequences with conditions: {conditions}")
    sequences = generator.generate(args.num_sequences, conditions)

    # Save to FASTA
    with open(args.output, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i+1}\n{seq}\n")

    print(f"Saved {len(sequences)} sequences to {args.output}")


if __name__ == '__main__':
    main()
