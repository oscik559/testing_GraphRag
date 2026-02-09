import os
import sqlite3
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# OPENAI_API_KEY no longer strictly needed but good to keep if switching back

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    print("Error: Missing environment variables. Please check your .env file.")
    exit(1)

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")

def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def setup_constraints():
    print("Setting up constraints locally...")
    with driver.session() as session:
        # Constraints ensuring uniqueness
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:CADFile) REQUIRE c.file_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:StandardDocument) REQUIRE s.doc_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (sec:Section) REQUIRE sec.section_id IS UNIQUE")
        
        # Drop existing index if needed (to handle dimension change)
        try:
            session.run("DROP INDEX standard_embeddings IF EXISTS")
        except Exception as e:
            print(f"Index drop warning: {e}")

        # Create new index with 768 dimensions for Nomic
        session.run("CREATE VECTOR INDEX `standard_embeddings` IF NOT EXISTS FOR (s:StandardDocument) ON (s.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}")
        
def clear_database():
    print("Clearing existing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def ingest_cad_data():
    print("Ingesting CAD data...")
    conn = get_db_connection('data/harvested_cad.db')
    
    # Ingest CAD Files
    files = conn.execute("SELECT * FROM cad_files").fetchall()
    with driver.session() as session:
        for file in files:
            session.run("""
                MERGE (c:CADFile {file_id: $file_id})
                SET c.file_name = $file_name,
                    c.file_type = $file_type,
                    c.part_number = $part_number,
                    c.revision = $revision,
                    c.file_size = $file_size_bytes,
                    c.extraction_status = $extraction_status
            """, dict(file))
            
    # Ingest Engineering Notes
    notes = conn.execute("SELECT * FROM cad_engineering_notes").fetchall()
    with driver.session() as session:
        for note in notes:
            session.run("""
                MATCH (c:CADFile {file_id: $file_id})
                CREATE (n:EngineeringNote {
                    note_id: $id,
                    note_type: $note_type,
                    note_number: $note_code,
                    content: $note_value
                })
                CREATE (c)-[:HAS_NOTE]->(n)
            """, dict(note))
            
    # Ingest Material Properties
    materials = conn.execute("SELECT * FROM cad_material_properties").fetchall()
    with driver.session() as session:
        for mat in materials:
            session.run("""
                MATCH (c:CADFile {file_id: $file_id})
                MERGE (m:Material {name: $material_name})
                SET m.standard = $material_standard,
                    m.density = $density
                MERGE (c)-[:HAS_MATERIAL]->(m)
            """, dict(mat))
            
    conn.close()
    print(f"Ingested {len(files)} CAD files.")

def ingest_rds_data():
    print("Ingesting RDS data...")
    conn = get_db_connection('data/harvested_rds.db')
    
    # Ingest Documents
    docs = conn.execute("SELECT * FROM rds_documents").fetchall()
    with driver.session() as session:
        for doc in docs:
            # Generate embedding for the document title/summary
            # Some titles might be None, handle that
            title = doc['title'] if doc['title'] else ""
            text_to_embed = f"{doc['standard_code']} {title}"
            embedding = embeddings.embed_query(text_to_embed)
            
            session.run("""
                MERGE (s:StandardDocument {doc_id: $doc_id})
                SET s.standard_code = $standard_code,
                    s.title = $title,
                    s.total_pages = $total_pages,
                    s.extraction_timestamp = $extraction_date,
                    s.embedding = $embedding
            """, {**dict(doc), "embedding": embedding})
            
    conn.close()
    print(f"Ingested {len(docs)} RDS documents.")

def create_relationships():
    print("Creating relationships between Notes and Standards...")
    with driver.session() as session:
        # Fetch all notes
        result = session.run("MATCH (n:EngineeringNote) RETURN n.note_id as id, n.content as content")
        notes = list(result)
        
        count = 0
        for note in notes:
            content = note["content"]
            # Look for patterns like "STD 1234"
            matches = re.findall(r"STD\s?([0-9]+)", content)
            
            for code_num in matches:
                # Construct query string to find matching standard
                # We assume standard_code format in DB matches "STD <number>" or similar
                # Using a loose match or exact match depending on data quality.
                # Here we try to match the number part if the standard_code contains it.
                
                query = """
                MATCH (n:EngineeringNote {note_id: $note_id})
                MATCH (s:StandardDocument)
                WHERE s.standard_code CONTAINS $code_num
                MERGE (n)-[:REFERENCES]->(s)
                RETURN count(s) as c
                """
                res = session.run(query, note_id=note["id"], code_num=code_num).single()
                if res and res["c"] > 0:
                    count += 1
                    
    print(f"Created {count} links between Notes and Standards.")

def main():
    try:
        setup_constraints()
        clear_database()
        ingest_cad_data()
        ingest_rds_data()
        create_relationships()
        print("Ingestion complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
