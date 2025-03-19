from knowledge_base import create_knowledge_base, check_knowledge_base_exists
import os
import shutil
from pathlib import Path

def rebuild_knowledge_base():
    """Rebuild the knowledge base properly."""
    # Set OpenAI API key if available
    if 'OPENAI_API_KEY' not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ['OPENAI_API_KEY'] = api_key
    
    # Remove existing index directory if it exists
    index_dir = Path("index")
    if index_dir.exists():
        print("Removing existing index directory...")
        try:
            shutil.rmtree(index_dir)
            print("Existing index directory removed.")
        except Exception as e:
            print(f"Error removing index directory: {str(e)}")
    
    # Create the index directory
    index_dir.mkdir(exist_ok=True)
    
    # Progress callback for feedback
    def progress_callback(progress, status):
        print(f"Progress: {progress:.2f} - {status}")
    
    # Create the knowledge base
    print("Creating knowledge base...")
    try:
        create_knowledge_base(progress_callback)
        
        # Verify it was created properly
        exists = check_knowledge_base_exists()
        if exists:
            print("Knowledge base created successfully!")
        else:
            print("Failed to create knowledge base.")
    except Exception as e:
        print(f"Error creating knowledge base: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    rebuild_knowledge_base() 