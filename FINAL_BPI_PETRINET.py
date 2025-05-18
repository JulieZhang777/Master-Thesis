#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:51:03 2025

@author: yachi
"""


from collections import Counter
import pm4py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import pytz

def import_xes(filename: str):
    log = pm4py.read_xes(filename)

    # Ensure the 'time:timestamp' column exists
    if 'time:timestamp' not in log.columns:
        raise ValueError("'time:timestamp' column is missing from the log.")

    # Backup original timestamp values
    log['original_timestamp'] = log['time:timestamp']

    # Step 1: Attempt parsing all timestamps
    def safe_parse(x):
        try:
            # Specify the format for parsing the datetime string
            if isinstance(x, str):  # If it's a string, try parsing it first
                # Format: Day/Month/Year Hour:Minute:Second AM/PM
                return pd.to_datetime(x, format="%d/%m/%Y %I:%M:%S %p", errors='raise') 
            else:
                return pd.to_datetime(x)  # If it's already datetime, just convert it
        except Exception as e:
            print(f"Failed to parse timestamp: {x} due to error: {e}")
            return None  # Return None in case of failure

    # Ensure the timestamps are in datetime format (timezone-aware or naive)
    log['time:timestamp'] = log['time:timestamp'].apply(safe_parse)

    # Step 2: Convert all timestamps to UTC+0 timezone
    log['time:timestamp'] = log['time:timestamp'].apply(
        lambda x: x.tz_convert('UTC') if x and x.tzinfo else pd.to_datetime(x).tz_localize('UTC', ambiguous='NaT')
    )

    # Step 3: Convert all timestamps to the correct timezone (e.g., London time)
    log['converted_timestamp'] = log['time:timestamp'].apply(
        lambda x: x.tz_convert('Europe/London') if x and x.tzinfo else pd.to_datetime(x).tz_localize('Europe/London', ambiguous='NaT')
    )

    # Step 4: Remove timezone information (make timestamps timezone naive)
    log['converted_timestamp'] = log['converted_timestamp'].apply(
        lambda x: x.replace(tzinfo=None) if x else None
    )

    # Step 5: Handle rows that failed to convert
    failed_rows = log[log['converted_timestamp'].isna()]

    if not failed_rows.empty:
        # Save failed rows with their original timestamps
        failed_rows[['original_timestamp', 'converted_timestamp']].to_csv("/Users/yachi/Desktop/FAIL_bpitimestamp_conversion.csv", index=False)
        print(f"Failed conversions saved to 'FAIL_bpitimestamp_conversion.csv' for review.")
    else:
        print("No failed conversions.")

    # Step 6: Save the original and successfully converted timestamps
    log.dropna(subset=['converted_timestamp'], inplace=True)  # Drop rows with NaT
    log[['original_timestamp', 'converted_timestamp']].to_csv("/Users/yachi/Desktop/bpifinalconversion.csv", index=False)

    print("Timestamp conversion completed and saved to CSV.")
    return log


def sort_log(df, case_id='case:concept:name', timestamp='converted_timestamp'):
    df_help = df.sort_values([case_id, timestamp], ascending=[True, True], kind='mergesort')
    df_first = df_help.drop_duplicates(subset=case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending=True, kind='mergesort')
    df_first['case_id_int'] = list(range(len(df_first)))
    df = df.merge(df_first.drop(columns=timestamp), on=case_id, how='left')
    df = df.sort_values(['case_id_int', timestamp], ascending=[True, True], kind='mergesort')
    return df.drop(columns='case_id_int')

def train_test_split_by_case(df, test_fraction=0.2, case_id='case:concept:name'):
    case_ids = df[case_id].unique()
    split_point = int(len(case_ids) * (1.0 - test_fraction))
    df_train = df[df[case_id].isin(case_ids[:split_point])].copy()
    df_test = df[df[case_id].isin(case_ids[split_point:])].copy()

    print(f"Training Set Size: {len(df_train)}")
    print(f"Test Set Size: {len(df_test)}")
    
    return df_train, df_test



def create_prefix_nextevent_pairs(df, max_prefix_length=174, case_id='case:concept:name', activity_id='concept:name'):
    X, y = [], []
    grouped = df.groupby(case_id)
    
    for _, group in grouped:
        sequence = group[activity_id].tolist()
        if len(sequence) < 2:
            continue
        for j in range(1, len(sequence)):
            prefix = sequence[max(0, j - max_prefix_length):j]
            next_event = sequence[j]
            X.append(prefix)
            y.append(next_event)
    
    print(f"Generated pairs (X, y): {len(X)}")
    return X, y


def encode_sequences(sequences, token_to_index, max_len):
    encoded = [[token_to_index.get(token, 0) for token in seq] for seq in sequences]
    return pad_sequences(encoded, maxlen=max_len, padding='pre', truncating='pre')

def encode_labels(labels, token_to_index):
    return np.array([token_to_index.get(label, 0) for label in labels])

def find_max_prefix_length(df, case_id='case:concept:name', activity_id='concept:name'):
    grouped = df.groupby(case_id)[activity_id].apply(list)
    max_case_length = grouped.apply(len).max()
    print(f"Maximum case length: {max_case_length}")
    return max_case_length - 1

def discover_net_inductive(log, noise_threshold=0.0):
    """
    仅使用 Inductive Miner 发现 Petri 网
    """
    return pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)

def get_enabled_activities_no_force(petri_net, initial_marking, prefix):
    """
    Determines which activities are enabled in a Petri net given a prefix.

    :param petri_net: The Petri net object (discovered net)
    :param initial_marking: The initial marking of the Petri net
    :param final_marking: The final marking of the Petri net
    :param prefix: A list of activities (strings) representing the incomplete execution
    :return: A set of enabled activities (strings)
    """
    from pm4py.objects.petri_net.obj import Marking

    # Clone the marking to avoid modifying the original
    current_marking = Marking(initial_marking)

    # Replay the prefix on the Petri net
    for activity in prefix:
        # Find the transition corresponding to the activity
        transitions = [t for t in petri_net.transitions if t.label == activity]
        if not transitions:
            raise ValueError(f"Activity '{activity}' not found in the Petri net.")

        transition = transitions[0]  # Assuming unique labels

        # Check if the transition is enabled
        enabled_transitions = [t for t in petri_net.transitions if all(
            current_marking[arc.source] >= arc.weight for arc in t.in_arcs)]
        if transition in enabled_transitions:
            # Fire transition manually by updating current marking
            for arc in transition.in_arcs:
                current_marking[arc.source] -= arc.weight
            for arc in transition.out_arcs:
                if arc.target in current_marking:
                    current_marking[arc.target] += arc.weight
                else:
                    current_marking[arc.target] = arc.weight
        else:
            raise ValueError(f"Activity '{activity}' is not enabled at this point in the prefix.")

    # Get the list of enabled transitions from the current marking
    enabled_transitions = [t for t in petri_net.transitions if all(
        current_marking[arc.source] >= arc.weight for arc in t.in_arcs)]

    # Return the labels of enabled transitions (activities)
    enabled_activities = {t.label for t in enabled_transitions if t.label is not None}

    return enabled_activities

def get_next_event_probabilities(petri_net, initial_marking, prefix, alphabet, token_to_index):
    from pm4py.objects.petri_net.obj import Marking

    current_marking = Marking(initial_marking)
    for activity in prefix:
        transitions = [t for t in petri_net.transitions if t.label == activity]
        if not transitions:
            break
        transition = transitions[0]
        enabled_transitions = [t for t in petri_net.transitions if all(
            current_marking[arc.source] >= arc.weight for arc in t.in_arcs)]
        if transition in enabled_transitions:
            for arc in transition.in_arcs:
                current_marking[arc.source] -= arc.weight
            for arc in transition.out_arcs:
                if arc.target in current_marking:
                    current_marking[arc.target] += arc.weight
                else:
                    current_marking[arc.target] = arc.weight

    enabled_transitions = [t for t in petri_net.transitions if all(
        current_marking[arc.source] >= arc.weight for arc in t.in_arcs)]

    enabled_labels = [t.label for t in enabled_transitions if t.label]
    total_enabled = len(enabled_labels)
    
    prob_vector = np.zeros(len(alphabet))
    if total_enabled > 0:
        for label in enabled_labels:
            if label in token_to_index:
                prob_vector[token_to_index[label] - 1] = 1 / total_enabled
    return prob_vector


# ----------- Main Execution -----------

filename = '/Users/yachi/Desktop/BPI_Challenge_2012.xes'
activity_column = 'concept:name'
timestamp_column = 'converted_timestamp'
case_column = 'case:concept:name'

log = import_xes(filename)
sorted_log = sort_log(log, case_id=case_column, timestamp=timestamp_column)


df_train, df_test = train_test_split_by_case(sorted_log, test_fraction=0.2, case_id=case_column)
print(sorted_log)

max_len = find_max_prefix_length(df_train, case_id=case_column, activity_id=activity_column)
print(f"Determined max prefix length: {max_len}")

alphabet = sorted_log[activity_column].unique()
token_to_index = {x: i + 1 for i, x in enumerate(alphabet)}
vocab_size = len(token_to_index) + 1

train_prefixes, train_suffixes = create_prefix_nextevent_pairs(df_train, max_prefix_length=max_len, case_id=case_column, activity_id=activity_column)
test_prefixes, test_suffixes = create_prefix_nextevent_pairs(df_test, max_prefix_length=max_len, case_id=case_column, activity_id=activity_column)

X_train = encode_sequences(train_prefixes, token_to_index, max_len)
X_test = encode_sequences(test_prefixes, token_to_index, max_len)
y_train = encode_labels(train_suffixes, token_to_index)
y_test = encode_labels(test_suffixes, token_to_index)

# ----------- Callbacks -----------

early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor the validation loss
    patience=6,            # Allow 6 epochs of no improvement before stopping
    restore_best_weights=True  # Restore the best model weights
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss for reduction in learning rate
    factor=0.5,          # Reduce the learning rate by 50%
    patience=3,          # Wait 3 epochs without improvement before reducing LR
    verbose=1,           # Print message when LR is reduced
    min_delta=0.0001     # Minimum change to qualify as an improvement
)

# ----------- 1. LSTM Model -----------

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(
    X_train, y_train,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reducer],  # Added callbacks here
    batch_size=64,
    epochs=600,
    verbose=2
)

y_pred = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=-1)

accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=1)
mae = mean_absolute_error(y_test, y_pred_labels)

print(f"\n[Baseline LSTM]")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test MAE: {mae:.4f}")

# -----------  2: LSTM Model with DFG -----------

def discover_dfg(sorted_log):
    return pm4py.discover_dfg(sorted_log)[0]

def get_dfg_probabilities(dfg, last_activity):
    outgoing_edges = {target: freq for (source, target), freq in dfg.items() if source == last_activity}
    total_frequency = sum(outgoing_edges.values())
    probabilities = {target: freq / total_frequency for target, freq in outgoing_edges.items()} if total_frequency > 0 else {}
    return probabilities

def get_dfg_prob_vector(prefix, dfg, token_to_index, vocab_size):
    if not prefix:
        return np.zeros(vocab_size)
    last_activity = prefix[-1]
    probs = get_dfg_probabilities(dfg, last_activity)
    prob_vector = np.zeros(vocab_size)
    for act, prob in probs.items():
        idx = token_to_index.get(act, 0)
        if idx < vocab_size:
            prob_vector[idx] = prob
    return prob_vector

def build_dfg_vectors(prefixes, dfg, token_to_index, vocab_size):
    return np.array([get_dfg_prob_vector(prefix, dfg, token_to_index, vocab_size) for prefix in prefixes])

dfg = discover_dfg(df_train)
X_train_dfg = build_dfg_vectors(train_prefixes, dfg, token_to_index, vocab_size)
X_test_dfg = build_dfg_vectors(test_prefixes, dfg, token_to_index, vocab_size)

seq_input = Input(shape=(max_len,), name="sequence_input")
dfg_input = Input(shape=(vocab_size,), name="dfg_input")

embedding = Embedding(input_dim=vocab_size, output_dim=128)(seq_input)
lstm_out = LSTM(128)(embedding)
merged = Concatenate()([lstm_out, dfg_input])
output = Dense(vocab_size, activation='softmax')(merged)

model_enhanced_dfg = Model(inputs=[seq_input, dfg_input], outputs=output)

optimizer_enhanced = Adam(learning_rate=0.001)

model_enhanced_dfg.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_enhanced, metrics=['accuracy'])

model_enhanced_dfg.fit(
    [X_train, X_train_dfg], y_train,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reducer], 
    batch_size=64,
    epochs=600,
    verbose=2
)

# Predict from the enhanced model
y_pred_enhanced_dfg = model_enhanced_dfg.predict([X_test, X_test_dfg], verbose=0)

# Convert predicted probabilities to labels
y_pred_labels_enhanced_dfg = np.argmax(y_pred_enhanced_dfg, axis=-1)



accuracy = accuracy_score(y_test, y_pred_labels_enhanced_dfg)
precision = precision_score(y_test, y_pred_labels_enhanced_dfg, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_labels_enhanced_dfg, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred_labels_enhanced_dfg, average='weighted', zero_division=1)
mae = mean_absolute_error(y_test, y_pred_labels_enhanced_dfg)



print(f"\n[Enhanced LSTM + DFG]")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test MAE: {mae:.4f}")


# -----------  3: DFG -----------


from collections import Counter

# Step 1: Most common label in training set (for fallback)
label_counts = Counter(train_suffixes)
most_common_label = label_counts.most_common(1)[0][0]

# Step 2: Predict next activity using DFG (returns string labels)
def dfg_predict_next_labels(prefixes, dfg, fallback_label):
    predictions = []
    for prefix in prefixes:
        if not prefix:
            predictions.append(fallback_label)
            continue
        last_event = prefix[-1]
        candidates = {target: freq for (source, target), freq in dfg.items() if source == last_event}
        if candidates:
            next_event = max(candidates.items(), key=lambda x: x[1])[0]
        else:
            next_event = fallback_label
        predictions.append(next_event)
    return predictions

# Step 3: Run prediction (string -> index conversion added here!)
y_pred_labels_dfg_str = dfg_predict_next_labels(test_prefixes, dfg, most_common_label)
y_pred_labels_dfg = [token_to_index.get(label, -1) for label in y_pred_labels_dfg_str]  # NOW indices

# Step 4: Convert both y_pred and y_test to numeric for metric evaluation
y_pred_dfg = np.array(y_pred_labels_dfg)
y_true_dfg = y_test  # already encoded

# Step 5: Evaluate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

print("\n[DFG-Only]")
print(f"Test Accuracy:  {accuracy_score(y_true_dfg, y_pred_dfg):.4f}")
print(f"Test Precision: {precision_score(y_true_dfg, y_pred_dfg, average='weighted', zero_division=1):.4f}")
print(f"Test Recall:    {recall_score(y_true_dfg, y_pred_dfg, average='weighted', zero_division=1):.4f}")
print(f"Test F1 Score:  {f1_score(y_true_dfg, y_pred_dfg, average='weighted', zero_division=1):.4f}")
print(f"Test MAE:       {mean_absolute_error(y_true_dfg, y_pred_dfg):.4f}")

# -----------  4: LSTM Model with petri net -----------
net, im, fm = discover_net_inductive(df_train)
alphabet_list = list(token_to_index.keys())

def create_enhanced_inputs(prefixes, petri_net, initial_marking, alphabet, token_to_index):
    enhanced_features = []
    for prefix in prefixes:
        prob_vector = get_next_event_probabilities(petri_net, initial_marking, prefix, alphabet, token_to_index)
        enhanced_features.append(prob_vector)
    return np.array(enhanced_features)

X_train_petri = create_enhanced_inputs(train_prefixes, net, im, alphabet_list, token_to_index)
X_test_petri = create_enhanced_inputs(test_prefixes, net, im, alphabet_list, token_to_index)

# Sequence inputs
input_seq = Input(shape=(max_len,))
x = Embedding(input_dim=vocab_size, output_dim=128)(input_seq)
x = LSTM(128)(x)

# Petri inputs
input_petri = Input(shape=(len(alphabet_list),))
combined = Concatenate()([x, input_petri])
output = Dense(vocab_size, activation='softmax')(combined)

petri_enhanced_model = Model(inputs=[input_seq, input_petri], outputs=output)
petri_enhanced_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])


petri_enhanced_model.fit(
    [X_train, X_train_petri], y_train,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reducer],
    epochs=600,
    batch_size=64,   
    verbose=2
)


y_pred_petri = petri_enhanced_model.predict([X_test, X_test_petri], verbose=0)
y_pred_labels_petri = np.argmax(y_pred_petri, axis=-1)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred_labels_petri)
precision = precision_score(y_test, y_pred_labels_petri, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_labels_petri, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred_labels_petri, average='weighted', zero_division=1)
mae = mean_absolute_error(y_test, y_pred_labels_petri)

print(f"\n[Enhanced LSTM + Petri Net]")
print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1 Score:  {f1:.4f}")
print(f"Test MAE:       {mae:.4f}")



#----- FINAL RESULTS--------
# Reverse mapping: index to activity name

index_to_token = {idx: token for token, idx in token_to_index.items()}


# Convert DFG keys from string activity names to token indices
dfg_indexed = {
    (token_to_index[src], token_to_index[tgt]): freq
    for (src, tgt), freq in dfg.items()
    if src in token_to_index and tgt in token_to_index
}

# Decode prefix sequences from indices to strings
def decode_sequence(encoded_seq, index_to_token):
    return [index_to_token.get(idx, 'UNK') for idx in encoded_seq if idx != 0]

# Normalize DFG probabilities and predict next event based on last activity (using indices)
def get_dfg_probabilities_for_last_activity(prefix, dfg_indexed):
    if not prefix:
        return {}
    
    last_activity = prefix[-1]  # Should be index
    outgoing_edges = {target: freq for (source, target), freq in dfg_indexed.items() if source == last_activity}
    
    total_frequency = sum(outgoing_edges.values())
    probabilities = {target: freq / total_frequency for target, freq in outgoing_edges.items()} if total_frequency > 0 else {}
    
    return probabilities

# Get the most probable next event's probability
def get_most_probable_dfg_probability(prefix, dfg_indexed):
    if not prefix:
        return 0.0
    last_activity = prefix[-1]
    probs = get_dfg_probabilities_for_last_activity(prefix, dfg_indexed)
    return max(probs.values()) if probs else 0.0



# Decode label predictions
decoded_prefixes = [decode_sequence(seq, index_to_token) for seq in X_test]
true_labels = [index_to_token.get(idx, 'UNK') for idx in y_test]
pred_baseline = [index_to_token.get(idx, 'UNK') for idx in y_pred_labels]
pred_enhanced_dfg = [index_to_token.get(idx, 'UNK') for idx in y_pred_labels_enhanced_dfg]
pred_labels_dfg = [index_to_token.get(idx, 'UNK') for idx in y_pred_labels_dfg] 
pred_enhanced_petri = [index_to_token.get(idx, 'UNK') for idx in y_pred_labels_petri]


# Counter for prefix occurrences (tuples of decoded strings)
combined_data = list(zip(
    [tuple(decode_sequence(seq, index_to_token)) for seq in X_test],
    true_labels,
    pred_baseline,
    pred_enhanced_dfg,
    pred_labels_dfg,         
    pred_enhanced_petri
))
comb_counter = Counter(combined_data)

# Prepare the result rows
results = []
for (prefix_strs, true_ev, pred_baseline, pred_enhanced_dfg, pred_labels_dfg,pred_enhanced_petri), count in comb_counter.items():
    prefix_indices = [token_to_index[act] for act in prefix_strs if act in token_to_index]

    if not prefix_indices:
        most_probable_dfg_prob = 0.0
    else:
        most_probable_dfg_prob = get_most_probable_dfg_probability(prefix_indices, dfg_indexed)
    
    results.append({
        'Prefix': ' -> '.join(prefix_strs),
        'True Next Event': true_ev,
        'Baseline Prediction': pred_baseline,
        'Enhanced DFG Prediction': pred_enhanced_dfg,
        'DFG-only Prediction': pred_labels_dfg,
        'Petri net enhanced Prediction': pred_enhanced_petri,
        'Occurrence Count': count,
    })

# Create DataFrame and save
results_df = pd.DataFrame(results)

# Save to Excel
output_path = '/Users/yachi/Desktop/BPI__excel!.xlsx'
results_df.to_excel(output_path, index=False)



import pandas as pd





sorted_log.to_csv("/Users/yachi/Desktop/log_file.csv")
