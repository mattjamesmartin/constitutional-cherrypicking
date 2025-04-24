#!/bin/python
# -*- coding: utf-8 -*-

from packages import *
from nlp import *
from semantic import *

def process(config, encoder, nlp):
    print('Loading topic segments…')

    segments_dict = {}
    documents_dict = {}

    # List all CSV files in the specified data path
    file_list = [f for f in os.listdir(config['data_path']) if f.endswith('.csv')]

    for file in file_list:
        file_path = os.path.join(config['data_path'], file)
        source = os.path.splitext(file)[0]  # Use the file name (without extension) as the source

        with open(file_path, encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            # Get the header row
            header = next(reader)
            # Put the remaining rows into a list of lists
            data = [row for row in reader]

        # Determine the ID field if specified in config
        id_field = config.get('id_field', None)

        print(f'Building segments dict for {file}…')
        for i, row in enumerate(data):
            # Use a unique row identifier incorporating the file name
            row_id = str(row[header.index(id_field)]) if id_field else f"{source}_{i}"

            row_dict = {header[j]: row[j] for j in range(len(header))}  # Convert row to a dictionary

            # Add document-level data
            documents_dict[row_id] = {
                'text': " ".join(row[header.index(field)] for field in config['data_fields']),
                'source': source
            }

            # Process each specified text field
            for field in config['data_fields']:
                text = row[header.index(field)]
                doc = nlp(text, disable=['ner'])

                for sent_index, sent in enumerate(doc.sents):
                    if get_word_count(sent) < 3:
                        continue
                    segment_id = f"{row_id}/{field}/{sent_index}"
                    segments_dict[segment_id] = {
                        'text': sent.text,
                        'source': source
                    }

                    # Populate segment-specific metadata
                    for metadata_id, field_name in config['metadata'].items():
                        value = row_dict.get(field_name)
                        segments_dict[segment_id][metadata_id] = value

    # Serialize documents and segments
    print('Serialising documents…', len(documents_dict))
    filename = os.path.join(config['model_path'], 'documents_dict.json')
    with open(filename, 'w') as f:
        json.dump(documents_dict, f)

    print('Serialising segments…', len(segments_dict))
    filename = os.path.join(config['model_path'], 'segments_dict.json')
    with open(filename, 'w') as f:
        json.dump(segments_dict, f)

    # Encode segments
    segment_encodings = encode_segments(segments_dict, config, encoder, split_size=100)
    build_segment_segments_matrix(segment_encodings, config)

    print('Finished')
