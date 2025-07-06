import argparse
from transformers import AutoTokenizer
from chunker import Chunker
from embedder import Embedder
from graph_manager import GraphManager
from retriever import Retriever
from serializer import Serializer
from client import LLMClient

def main():
    parser = argparse.ArgumentParser(description="Segment-Graph Retrieval Memory Demo")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Path to the input text file to build context from")
    parser.add_argument("--query", type=str,
                        help="Query string for context retrieval and generation")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Number of tokens per chunk")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of similarity edges per node in the graph")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Number of paths to retrieve for context")
    parser.add_argument("--summarize", action="store_true",
                        help="Whether to summarize retrieved chunks before prompting the LLM")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Name of the LLM model for generation")
    # parser.add_argument("--summ_model", type=str, default="facebook/bart-large-cnn",
    #                     help="Name of the summarization model")
    parser.add_argument("--summ_model", type=str, default="sshleifer/distilbart-cnn-12-6",
                        help="Name of the summarization model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    chunker = Chunker(tokenizer, chunk_size=args.chunk_size)
    embedder = Embedder()
    graph_mgr = GraphManager(k=args.k)

    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunker.chunk(text)
    embeddings = embedder.encode(chunks)
    for chunk_id, chunk_text in chunks:
        graph_mgr.add_node(chunk_id, embeddings[chunk_id], chunk_text)

    retriever = Retriever(graph_mgr, embedder)

    if args.query:
        contexts = retriever.get_context(args.query, top_n=args.top_n)
        if args.summarize:
            serializer = Serializer(model_name=args.summ_model)
            contexts = serializer.summarize(contexts)
        prompt = "\n\n".join(contexts) + "\n\n" + args.query

        client = LLMClient(model_name=args.model_name)
        output = client.generate(prompt)
        print("\n--- Generated Output ---\n")
        print(output)
    else:
        print(f"Graph built with {len(chunks)} chunks and {graph_mgr.G.number_of_nodes()} nodes.")

if __name__ == "__main__":
    main()