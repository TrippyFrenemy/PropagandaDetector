import pandas as pd
import os
import glob

# Define directory paths
prefix_article = "../datasets-v5/tasks-2-3/train"
prefix_dataset = "../datasets/tasks-2-3"

# Initialize a list to collect all merged data
all_merged_data = []

# Get list of all article base names (without extensions)
article_files = glob.glob(f"{prefix_article}/*.txt")
article_ids = [os.path.basename(f).replace('.txt', '') for f in article_files]

# Process each article file
for article_id in article_ids:
    # Load article content
    with open(f"{prefix_article}/{article_id}.txt", 'r', encoding='utf-8', errors='replace') as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    content_combined = ' '.join(content)

    # Load Task 2 labels
    task2_path = f"{prefix_article}/{article_id}.task2.labels"
    task2_labels = pd.read_csv(task2_path, sep='\t', header=None, names=['article_id', 'sentence_id', 'label'])

    # Load Task 3 labels
    task3_path = f"{prefix_article}/{article_id}.task3.labels"
    task3_labels = pd.read_csv(task3_path, sep='\t', header=None, names=['article_id', 'technique', 'start_pos', 'end_pos'])

    # Convert Task 2 data into a DataFrame format
    task2_data = []
    for index, row in task2_labels.iterrows():
        sentence_id = int(row['sentence_id']) - 1  # Sentence ID starts from 1, convert to index
        label = row['label']
        sentence = content[sentence_id] if sentence_id < len(content) else None
        task2_data.append([row['article_id'], sentence_id + 1, sentence, label])

    task2_df = pd.DataFrame(task2_data, columns=['article_id', 'sentence_id', 'sentence', 'label'])
    task2_df = task2_df[task2_df['sentence'].str.strip() != '']  # Remove empty sentences

    # Combine Task 3 information by aligning it with sentence context
    task3_data = []
    for index, row in task3_labels.iterrows():
        technique = row['technique']
        start_pos = row['start_pos']
        end_pos = row['end_pos']
        fragment = content_combined[start_pos:end_pos]  # Extract fragment based on character positions

        task3_data.append([row['article_id'], technique, fragment, start_pos, end_pos])

    task3_df = pd.DataFrame(task3_data, columns=['article_id', 'technique', 'fragment', 'start_pos', 'end_pos'])

    # Merge Task 2 and Task 3 data based on fragment being part of sentence
    unique_fragments = set()  # Set to track unique (article_id, sentence_id, fragment) combinations
    merged_data = []
    for _, t2_row in task2_df.iterrows():
        matching_rows = task3_df[(task3_df['article_id'] == t2_row['article_id']) &
                                 (task3_df['fragment'].apply(lambda frag: frag.strip() in t2_row['sentence']))]
        if not matching_rows.empty:
            for _, t3_row in matching_rows.iterrows():
                unique_key = (t2_row['article_id'], t2_row['sentence_id'], t3_row['fragment'])
                if unique_key not in unique_fragments:
                    unique_fragments.add(unique_key)
                    merged_data.append([t2_row['article_id'], t2_row['sentence_id'], t2_row['sentence'],
                                        t2_row['label'], t3_row['technique'], t3_row['fragment'],
                                        t3_row['start_pos'], t3_row['end_pos']])
        else:
            # Add non-propaganda rows when no matching fragment is found
            merged_data.append([t2_row['article_id'], t2_row['sentence_id'], t2_row['sentence'],
                                t2_row['label'], 'non-propaganda', '', None, None])

    # Append merged data of current article to all_merged_data
    all_merged_data.extend(merged_data)
    print(f"{prefix_article}/{article_id}.txt - Read")

# Convert all merged data to DataFrame
final_df = pd.DataFrame(all_merged_data, columns=['article_id', 'sentence_id', 'sentence', 'label',
                                                  'technique', 'fragment', 'start_pos', 'end_pos'])

# Display the combined dataset for training
print(final_df.info())

# Optionally save to CSV for further training use
final_df.to_csv(f'{prefix_dataset}/combined_dataset.csv', index=False)
