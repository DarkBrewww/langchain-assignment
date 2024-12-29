# -*- coding: utf-8 -*-
"""


@author: Adwait
"""

import argparse
import os
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Function to compute embeddings
def compute_embeddings(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    stories = []

    for f in files:
        file_path = os.path.join(data_dir, f)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stories.append(file.read())
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")
            continue  # Skip problematic files

    if not stories:
        print("No stories were successfully loaded. Please check the dataset files.")
        return

    # Generate embeddings and store in vector database
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(stories, embeddings)
    db.save_local("faiss_index")
    print(f"Embeddings computed and saved for {len(files)} stories.")

# Function to get character info
def get_character_info(character_name):
    db = FAISS.load_local("faiss_index", embeddings=None)
    results = db.similarity_search(character_name, k=1)

    if not results:
        return json.dumps({"error": f"Character '{character_name}' not found."}, indent=2)

    story = results[0].text
    # Modify the parsing logic based on your story dataset structure
    character_info = {
        "name": character_name,
        "storyTitle": "Extracted Story Title",
        "summary": "Generated summary from the story.",
        "relations": [{"name": "Other Character", "relation": "Relation Type"}],
        "characterType": "Protagonist",
    }
    return json.dumps(character_info, indent=2)

# Main function to handle CLI
def main():
    parser = argparse.ArgumentParser(description="LangChain Assignment CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Sub-command: compute-embeddings
    parser_compute = subparsers.add_parser("compute-embeddings")
    parser_compute.add_argument("--data-dir", type=str, required=True, help="Path to the stories directory")

    # Sub-command: get-character-info
    parser_get = subparsers.add_parser("get-character-info")
    parser_get.add_argument("--name", type=str, required=True, help="Character name to retrieve info")

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        compute_embeddings(args.data_dir)
    elif args.command == "get-character-info":
        print(get_character_info(args.name))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
