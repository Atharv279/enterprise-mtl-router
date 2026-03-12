import lancedb
import pyarrow as pa
import os

DB_PATH = "../data/processed/lancedb"

def initialize_vector_db():
    # Create directory if it doesn't exist
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Initialize local, embedded database
    db = lancedb.connect(DB_PATH)
    
    # Define the exact PyArrow schema for LanceDB
    schema = pa.schema([
        pa.field("complaint_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024)), # 1024-dimensional E5 output
        pa.field("raw_text", pa.string()),
        pa.field("priority", pa.int32())
    ])
    
    # Create table if it doesn't exist
    if "historical_complaints" not in db.table_names():
        db.create_table("historical_complaints", schema=schema)
        print("Initialized LanceDB table: historical_complaints")
        
    return db.open_table("historical_complaints")

def search_similar_complaints(table, query_vector, limit=5):
    """Executes an Approximate Nearest Neighbor (ANN) search."""
    results = table.search(query_vector).limit(limit).to_pandas()
    return results