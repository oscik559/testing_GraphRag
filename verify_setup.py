import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def verify():
    print("Verifying Setup...")
    
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("FAILED: Missing environment variables.")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("SUCCESS: Connected to Neo4j.")
        
        with driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            print(f"SUCCESS: Found {count} nodes in the database.")
            
            links = session.run("MATCH ()-[r:REFERENCES]->() RETURN count(r) as c").single()["c"]
            print(f"SUCCESS: Found {links} REFERENCES relationships.")
            
        driver.close()
    except Exception as e:
        print(f"FAILED: Connection error: {e}")

if __name__ == "__main__":
    verify()
