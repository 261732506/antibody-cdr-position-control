"""
CDR Position Marking Algorithm - Core Implementation

This script implements the reverse-order CDR position marking algorithm
described in the manuscript (Algorithm 1).

Usage:
    python cdr_position_marking.py --input sequences.csv --output marked_sequences.csv

Input CSV format:
    sequence,cdr1,cdr2,cdr3
    QLVESGGGLVQ...GFTFSSYA...INSGGGST...AADGGYYCLGLEPYEYDF...,GFTFSSYA,INSGGGST,AADGGYYCLGLEPYEYDF

Output CSV format:
    original_sequence,marked_sequence,cdr1_pos,cdr2_pos,cdr3_pos
    [original],[sequence with markers],[pos1],[pos2],[pos3]
"""

import pandas as pd
import argparse
from typing import Tuple, List


class CDRPositionMarker:
    """CDR position marking algorithm implementation"""

    def __init__(self):
        self.markers = {
            'cdr1': ('<cdr1_start>', '<cdr1_end>'),
            'cdr2': ('<cdr2_start>', '<cdr2_end>'),
            'cdr3': ('<cdr3_start>', '<cdr3_end>')
        }

    def mark_sequence(self, sequence: str, cdr1: str, cdr2: str, cdr3: str) -> Tuple[str, dict]:
        """
        Apply CDR position marking to a sequence

        Args:
            sequence: Full antibody VH sequence
            cdr1, cdr2, cdr3: CDR sequences

        Returns:
            marked_sequence: Sequence with position markers
            positions: Dictionary of CDR positions
        """
        result = sequence
        insertions = []

        # Step 1: Locate CDRs (record position and length)
        pos3 = result.rfind(cdr3)  # Rightmost search for CDR3
        pos2 = result.rfind(cdr2)  # Rightmost search for CDR2
        pos1 = result.find(cdr1)   # Leftmost search for CDR1

        if pos1 == -1 or pos2 == -1 or pos3 == -1:
            raise ValueError("One or more CDRs not found in sequence")

        # Step 2: Store insertions with positions
        insertions = [
            (pos1, len(cdr1), self.markers['cdr1'][0], self.markers['cdr1'][1]),
            (pos2, len(cdr2), self.markers['cdr2'][0], self.markers['cdr2'][1]),
            (pos3, len(cdr3), self.markers['cdr3'][0], self.markers['cdr3'][1])
        ]

        # Step 3: Sort in reverse order (key insight!)
        insertions.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Insert markers from right to left
        for pos, length, start_tag, end_tag in insertions:
            result = result[:pos] + start_tag + result[pos:pos+length] + end_tag + result[pos+length:]

        positions = {'cdr1': pos1, 'cdr2': pos2, 'cdr3': pos3}

        return result, positions


def main():
    parser = argparse.ArgumentParser(description='Apply CDR position marking to antibody sequences')
    parser.add_argument('--input', required=True, help='Input CSV file with sequences')
    parser.add_argument('--output', required=True, help='Output CSV file with marked sequences')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} sequences from {args.input}")

    # Initialize marker
    marker = CDRPositionMarker()

    # Process sequences
    marked_sequences = []
    positions_list = []

    for idx, row in df.iterrows():
        try:
            marked_seq, positions = marker.mark_sequence(
                row['sequence'], row['cdr1'], row['cdr2'], row['cdr3']
            )
            marked_sequences.append(marked_seq)
            positions_list.append(positions)
        except Exception as e:
            print(f"Warning: Failed to mark sequence {idx}: {e}")
            marked_sequences.append(None)
            positions_list.append(None)

    # Save results
    df['marked_sequence'] = marked_sequences
    df['cdr1_position'] = [p['cdr1'] if p else None for p in positions_list]
    df['cdr2_position'] = [p['cdr2'] if p else None for p in positions_list]
    df['cdr3_position'] = [p['cdr3'] if p else None for p in positions_list]

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} marked sequences to {args.output}")


if __name__ == '__main__':
    main()
