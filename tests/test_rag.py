import os
import sys
import warnings
from typing import List, Dict, Any, Tuple

# Filter out LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.")
    api_key = input("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

# Import our modules
from knowledge_base import (
    check_knowledge_base_exists,
    get_document_store,
    get_random_example_questions
)
from rag_classic import query_rag
from evaluation import compare_answers

def test_knowledge_base():
    """Test if the knowledge base exists and can be loaded."""
    print("\n--- Testing Knowledge Base ---")
    try:
        # Check if the knowledge base exists
        kb_exists = check_knowledge_base_exists()
        print(f"Knowledge base exists: {kb_exists}")
        
        if not kb_exists:
            print("Knowledge base does not exist. Please create it first.")
            return False
        
        # Try to get the document store
        print("Testing document store retrieval...")
        vectorstore = get_document_store()
        
        if vectorstore:
            print("✅ Document store loaded successfully!")
            # Get some doc count stats
            try:
                doc_count = len(vectorstore.docstore._dict)
                print(f"Number of documents in the store: {doc_count}")
            except Exception as e:
                print(f"Could not determine document count: {e}")
            return True
        else:
            print("❌ Failed to load document store.")
            return False
    except Exception as e:
        print(f"❌ Error testing knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_retrieval(query: str = "What is RAG?"):
    """Test document retrieval functionality."""
    print("\n--- Testing Document Retrieval ---")
    try:
        # Query the RAG system
        result = query_rag(query)
        
        # Check if documents were retrieved
        docs = result.get("documents", [])
        print(f"Retrieved {len(docs)} documents for query: '{query}'")
        
        if not docs:
            print("❌ No documents retrieved.")
            return False
        
        # Print some info about the retrieved documents
        for i, doc in enumerate(docs[:2]):  # Show just the first 2 docs
            print(f"\nDocument {i+1} - Source: {doc.get('source', 'Unknown')}")
            content = doc.get('content', '')
            print(f"Content snippet: {content[:150]}..." if len(content) > 150 else content)
        
        print("\n✅ Document retrieval test completed!")
        return True
    except Exception as e:
        print(f"❌ Error testing document retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_example_questions():
    """Test random example questions."""
    print("\n--- Testing Example Questions ---")
    try:
        # Get random example questions
        example_qa_pairs = get_random_example_questions(n=2)
        
        if not example_qa_pairs:
            print("❌ No example questions found.")
            return False
        
        print(f"Found {len(example_qa_pairs)} example QA pairs")
        
        # Print a few examples
        for i, (question, answer) in enumerate(example_qa_pairs):
            print(f"\nExample {i+1}:")
            print(f"Q: {question}")
            print(f"A: {answer[:150]}..." if len(answer) > 150 else answer)
        
        print("\n✅ Example questions test completed!")
        return True
    except Exception as e:
        print(f"❌ Error testing example questions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classic_rag():
    """Test the classic RAG functionality."""
    print("\n--- Testing Classic RAG ---")
    try:
        # Test with a simple query
        query = "What is retrieval augmented generation?"
        print(f"Testing query: '{query}'")
        
        # Get response from RAG
        result = query_rag(query)
        
        # Check if we got an answer
        answer = result.get("answer", "")
        if not answer:
            print("❌ No answer was generated.")
            return False
        
        print(f"\nGenerated answer: {answer}")
        print("\n✅ Classic RAG test completed!")
        return True
    except Exception as e:
        print(f"❌ Error testing classic RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """Test the evaluation functionality."""
    print("\n--- Testing Evaluation ---")
    try:
        # Test with two similar answers
        answer1 = "Retrieval Augmented Generation (RAG) combines a retrieval component with a text generation model to create more accurate and relevant responses."
        answer2 = "RAG is a technique that integrates a retrieval mechanism with a generative model to produce responses with factual accuracy and context-awareness."
        
        # Test with two different answers
        different1 = "The capital of France is Paris, a city known for its culture and landmarks."
        different2 = "Machine learning algorithms can identify patterns in data to make predictions."
        
        # Calculate similarity
        similarity_score1, explanation1 = compare_answers(answer1, answer2)
        similarity_score2, explanation2 = compare_answers(different1, different2)
        
        print(f"Similarity for similar answers: {similarity_score1:.4f}")
        print(f"Explanation: {explanation1}")
        print(f"Similarity for different answers: {similarity_score2:.4f}")
        print(f"Explanation: {explanation2}")
        
        # Check if the results make sense
        if similarity_score1 > 0.7 and similarity_score2 < 0.5:
            print("\n✅ Evaluation test passed!")
            return True
        else:
            print("\n❌ Evaluation results don't match expectations.")
            return False
    except Exception as e:
        print(f"❌ Error testing evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests and summarize results."""
    print("\n=== RAG System Test Suite ===\n")
    
    # Keep track of passing tests
    tests_run = 0
    tests_passed = 0
    
    # Test knowledge base
    tests_run += 1
    if test_knowledge_base():
        tests_passed += 1
    
    # Test document retrieval
    tests_run += 1
    if test_document_retrieval():
        tests_passed += 1
    
    # Test example questions
    tests_run += 1
    if test_example_questions():
        tests_passed += 1
    
    # Test classic RAG
    tests_run += 1
    if test_classic_rag():
        tests_passed += 1
    
    # Test evaluation
    tests_run += 1
    if test_evaluation():
        tests_passed += 1
    
    # Summarize results
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{tests_run}")
    
    if tests_passed == tests_run:
        print("\n🎉 All tests passed! Your RAG system is working correctly.")
    else:
        print(f"\n⚠️ {tests_run - tests_passed} tests failed. Please fix the issues before running the application.")

if __name__ == "__main__":
    main() 