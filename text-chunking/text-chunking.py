def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    step=chunk_size-overlap
    chunks=[]

    for i in range (0, len(tokens), step):
        chunk= tokens[i:i+chunk_size]

        if i+chunk_size<= len(tokens)-step and len(chunk) < chunk_size:
            break
        chunks.append(chunk)

        if i +chunk_size>= len(tokens):
            break
    return chunks        