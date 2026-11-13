
"""Import the normalized Annamayya dataset into Neo4j."""

from data_loader import load_normalized_data
from knowledge_graph import AnnamayyaKnowledgeGraph


def main():
    compositions, text_records, audio_records = load_normalized_data()

    print("Loaded dataset:")
    print(f"Compositions: {len(compositions)}")
    print(f"Text records: {len(text_records)}")
    print(f"Audio records: {len(audio_records)}")

    graph = AnnamayyaKnowledgeGraph()

    try:
        graph.connect()
        graph.initialize_schema()

        counts = graph.import_dataset(
            compositions,
            text_records,
            audio_records,
        )

        print("\nImport successful!")
        for name, count in counts.items():
            print(f"{name}: {count}")

    finally:
        graph.close()


if __name__ == "__main__":
    main()
