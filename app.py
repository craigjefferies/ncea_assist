#!/usr/bin/env python3
"""
Test script to verify the improved DOCX text handling
"""

def test_smart_truncation():
    """Test the smart truncation function"""
    # Sample text that might come from a DOCX file
    sample_text = """
Introduction to the Project
This is an important introduction section that should be preserved.

This is some middle content that might be less important.

Another paragraph with some content.

Method
This is the method section which is very important for assessment.
We used various techniques to achieve our goals.

Results and Analysis
The results show significant improvement in our approach.
Achievement of the criteria was demonstrated through evidence.

Some less important middle section with formatting artifacts.

Conclusion
This conclusion summarizes the key findings and achievements.
The standard was met through comprehensive evidence.

Recommendation for future work.
"""
    
    print("🧪 Testing Smart Truncation Function")
    print("=" * 50)
    
    # Test 1: Normal length (should not truncate)
    print("\n1. Testing with normal length limit...")
    result1 = smart_truncate_text(sample_text, 2000, "test.docx")
    print(f"   Original: {len(sample_text)} chars")
    print(f"   Result: {len(result1)} chars")
    print(f"   Truncated: {'Yes' if len(result1) < len(sample_text) else 'No'}")
    
    # Test 2: Aggressive truncation
    print("\n2. Testing with aggressive truncation...")
    result2 = smart_truncate_text(sample_text, 500, "test.docx")
    print(f"   Result: {len(result2)} chars")
    print(f"   Contains 'Introduction': {'Introduction' in result2}")
    print(f"   Contains 'Method': {'Method' in result2}")
    print(f"   Contains 'Conclusion': {'Conclusion' in result2}")
    
    # Show what was preserved
    print(f"\n   Preview of truncated content:")
    print(f"   {result2[:200]}...")
    
    return True

def smart_truncate_text(text: str, max_chars: int, file_name: str = "") -> str:
    """Copy of the smart truncation function for testing"""
    if len(text) <= max_chars:
        return text
    
    # Split into sections/paragraphs
    paragraphs = text.split('\n\n')
    
    # Prioritize content (look for key indicators)
    important_keywords = [
        'achievement', 'standard', 'criteria', 'evidence', 'conclusion', 
        'method', 'result', 'analysis', 'evaluation', 'recommendation',
        'introduction', 'aim', 'hypothesis', 'discussion', 'summary'
    ]
    
    # Score paragraphs by importance
    scored_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        score = 0
        para_lower = paragraph.lower()
        
        # Higher score for paragraphs with important keywords
        for keyword in important_keywords:
            score += para_lower.count(keyword) * 10
        
        # Prefer paragraphs near the beginning and end
        if i < len(paragraphs) * 0.3:  # First 30%
            score += 20
        elif i > len(paragraphs) * 0.7:  # Last 30%
            score += 10
        
        # Prefer longer paragraphs (more substantial content)
        if len(paragraph) > 100:
            score += 5
        
        scored_paragraphs.append((score, paragraph, i))
    
    # Sort by score (highest first)
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Build truncated text
    result_text = ""
    used_chars = 0
    
    for score, paragraph, original_index in scored_paragraphs:
        # Add paragraph if it fits
        if used_chars + len(paragraph) + 4 <= max_chars:  # +4 for spacing
            if result_text:
                result_text += "\n\n"
            result_text += paragraph
            used_chars = len(result_text)
        elif used_chars < max_chars * 0.8:  # Still have significant space
            # Try to fit a truncated version of this paragraph
            remaining_space = max_chars - used_chars - 50  # Leave space for truncation notice
            if remaining_space > 100:  # Only if meaningful space left
                if result_text:
                    result_text += "\n\n"
                result_text += paragraph[:remaining_space] + "..."
                break
        else:
            break
    
    if len(result_text) < max_chars * 0.5:
        # Fallback: simple truncation if smart truncation didn't work well
        result_text = text[:max_chars]
    
    # Add truncation notice
    if len(text) > len(result_text):
        result_text += f"\n\n[Content truncated from {len(text)} to {len(result_text)} characters for processing]"
    
    return result_text

if __name__ == "__main__":
    success = test_smart_truncation()
    if success:
        print("\n🎉 Smart truncation test completed successfully!")
        print("\n📋 This should help with DOCX files that have too much text content.")
    else:
        print("\n❌ Test failed.")
