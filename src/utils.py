from collections import Counter
import json
import re
from janome.tokenizer import Tokenizer
from rich.progress import track
from huggingface_hub import HfApi, create_repo, hf_hub_download


def analyze_generic_data(processed_output):
    lengths = []
    len_ner = []
    unique_entities = []

    for d in track(processed_output, description="Analyzing data..."):
        lengths.append(len(d["tokenized_text"]))
        len_ner.append(len(d["ner"]))
        for n in d["ner"]:
            unique_entities.append((str(n[2]).lower()))

    average_length = sum(lengths) / len(lengths)
    average_ner = sum(len_ner) / len(len_ner)
    unique_entities = list(set(unique_entities))

    most_common = [x[0] for x in Counter(unique_entities).most_common()[:10]]

    results = {
        "average_length": average_length,
        "average_ner": average_ner,
        "unique_entities": unique_entities,
        "most_common": most_common
    }

    with open("data/analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def janome_tokenize_text(text):
    """Tokenize the input text into a list of tokens using janome for Japanese text."""
    tokenizer = Tokenizer()
    tokens = [token.surface for token in tokenizer.tokenize(text)]
    return tokens


def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def fix_entity_data(data):
    for entity in data['entities']:
        if 'type' in entity:
            if isinstance(entity['type'], list):
                if 'types' in entity and isinstance(entity['types'], list):
                    entity['types'].extend(entity['type'])
                else:
                    entity['types'] = entity['type']
            else:
                entity['types'] = [entity['type']]
            entity.pop('type')
    return data


def extract_entities(data):
    """Extract named entities from the input data and return a list of examples."""
    all_examples = []

    for dt in track(data, description="Parsing data..."):
        # Attempt to extract entities; skip current record on failure
        try:
            dt = fix_entity_data(dt)
            tokens = tokenize_text(dt['text'])
            ents = [(k["entity"], k["types"]) for k in dt['entities']]
        except Exception as e:
            print(dt)
            print(f"Error processing record: {e}")
            break
        spans = []
        for entity in ents:
            entity_tokens = tokenize_text(str(entity[0]))

            # Find the start and end indices of each entity in the tokenized text
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                    for el in entity[1]:
                        spans.append((i, i + len(entity_tokens) - 1, el.lower().replace('_', ' ')))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})
    return all_examples


def extract_entities_japanese(data):
    """Extract named entities from the input data and return a list of examples."""
    all_examples = []

    for dt in track(data, description="[blue]Parsing data..."):
        # Attempt to extract entities; skip current record on failure
        tokens = list(dt['text'])
        spans = []
        for entity in dt['entities']:

            spans.append((entity['span'][0], entity['span'][1] - 1, entity['type']))

        # Append the tokenized text and its corresponding named entity recognition data
        all_examples.append({"tokenized_text": tokens, "ner": spans})
    return all_examples


def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def upload_to_hf(repo_name, file_path):
    """Upload the repository to the Hugging Face Hub."""
    hf_api = HfApi()
    create_repo(repo_name, repo_type="dataset")
    hf_api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_name,
        repo_type="dataset",
    )


def download_from_hf(repo_name, filename):
    """Download the repository from the Hugging Face Hub."""
    hf_hub_download(repo_id=repo_name, filename=filename, repo_type="dataset", local_dir="downloaded_data")
