from data_ingestion import ingest_data
from data_processing import process_data
from model_training import train_model
from model_evaluation import evaluate_model

# Ingest the data
print("Ingesting data...")
ingest_data()

# Process the data
print("Processing data...")
process_data()

# Train the model
print("Training model...")
train_model()

# Evaluate the model
print("Evaluating model...")
evaluate_model()

print("Pipeline complete.")
