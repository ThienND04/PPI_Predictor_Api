from typing import Dict, Iterable, Iterator, List, Tuple, TextIO


def _iter_text_lines(file_stream: TextIO) -> Iterator[str]:
    """Yield text lines from a possibly-binary stream, decoding as UTF-8.

    Reads incrementally to avoid loading entire content into memory.
    """
    # Werkzeug's FileStorage.stream may be binary; handle both cases
    first_read = file_stream.read(0)
    if isinstance(first_read, bytes):
        # Re-open text wrapper over the same stream
        # Reset not strictly needed for read(0), but ensure pointer is at current pos
        pass
    # Ensure we read from the beginning if possible
    try:
        file_stream.seek(0)
    except Exception:
        # If not seekable, proceed from current position
        pass

    while True:
        chunk = file_stream.readline()
        if not chunk:
            break
        if isinstance(chunk, bytes):
            yield chunk.decode('utf-8', errors='ignore')
        else:
            yield chunk


def parse_fasta(file_stream: TextIO) -> Dict[str, str]:
    """Parse a FASTA file stream into a mapping {id: sequence}.

    - Supports multi-sequence FASTA
    - Processes line-by-line to be memory-friendly
    """
    sequences: Dict[str, List[str]] = {}
    current_id: str = ""

    for raw_line in _iter_text_lines(file_stream):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('>'):
            header = line[1:].strip()
            # Take the first token as ID
            current_id = header.split()[0]
            if current_id not in sequences:
                sequences[current_id] = []
            continue
        if not current_id:
            # Sequence content before any header → skip
            continue
        # Append sequence content (letters, no spaces)
        sequences[current_id].append(''.join(line.split()).upper())

    # Collapse lists to strings, filter empties
    result: Dict[str, str] = {}
    for pid, parts in sequences.items():
        seq = ''.join(parts)
        if seq:
            result[pid] = seq
    return result


def parse_pairs(file_stream: TextIO) -> Iterator[Tuple[str, str]]:
    """Yield pairs (id1, id2) from a CSV/TSV stream line-by-line.

    - Automatically detects comma or tab delimiter per line
    - Ignores empty/comment lines and single-column lines
    - Does not load entire file into memory
    """
    for raw_line in _iter_text_lines(file_stream):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue

        # Detect delimiter: prefer tab if present, else comma
        if '\t' in line:
            parts = [p.strip() for p in line.split('\t')]
        else:
            parts = [p.strip() for p in line.split(',')]

        if len(parts) < 2:
            # Skip malformed lines here; route can report error per pair if needed
            continue

        id1, id2 = parts[0], parts[1]
        if not id1 or not id2:
            continue
        yield id1, id2


