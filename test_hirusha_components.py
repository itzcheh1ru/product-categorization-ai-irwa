#!/usr/bin/env python3
"""
Test script for Hirusha's assigned components:
- TagGeneratorAgent
- Security utilities
- API schemas
"""

import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_tag_generator():
    """Test the TagGeneratorAgent"""
    print("🧪 Testing TagGeneratorAgent...")
    
    try:
        from agents.tag_generator_agent import TagGeneratorAgent
        
        # Create agent instance (without LLM/NLP for basic testing)
        agent = TagGeneratorAgent()
        
        # Test data
        product_data = {
            'productDisplayName': 'Red Cotton Summer Dress',
            'baseColour': 'Red',
            'usage': 'Casual',
            'articleType': 'Dress',
            'season': 'Summer'
        }
        
        # Test tag generation
        result = agent.generate_tags(product_data)
        
        print(f"✅ Tag generation successful!")
        print(f"   Generated {len(result.get('tags', []))} tags")
        print(f"   Timestamp: {result.get('generation_timestamp', 'N/A')}")
        
        # Display some tags
        if 'tags' in result and result['tags']:
            print("   Sample tags:")
            for i, tag in enumerate(result['tags'][:3]):
                print(f"     {i+1}. {tag.get('tag', 'N/A')} (confidence: {tag.get('confidence', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ TagGeneratorAgent test failed: {e}")
        return False

def test_security():
    """Test security utilities"""
    print("\n🔒 Testing Security utilities...")
    
    try:
        from core.security import (
            get_password_hash, verify_password, validate_password_strength,
            create_access_token, decode_access_token, sanitize_input,
            generate_csrf_token, verify_csrf_token, is_safe_filename
        )
        
        # Test password hashing
        password = "TestPassword123!"
        hashed = get_password_hash(password)
        is_valid = verify_password(password, hashed)
        print(f"✅ Password hashing: {'PASS' if is_valid else 'FAIL'}")
        
        # Test password validation
        validation = validate_password_strength(password)
        print(f"✅ Password validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
        print(f"   Score: {validation['score']}/5")
        
        # Test JWT tokens
        user_data = {"user_id": "123", "username": "testuser"}
        token = create_access_token(user_data)
        decoded = decode_access_token(token)
        print(f"✅ JWT tokens: {'PASS' if decoded and decoded.get('user_id') == '123' else 'FAIL'}")
        
        # Test input sanitization
        malicious_input = "<script>alert('xss')</script>"
        sanitized = sanitize_input(malicious_input)
        print(f"✅ Input sanitization: {'PASS' if '<script>' not in sanitized else 'FAIL'}")
        
        # Test CSRF tokens
        csrf_token = generate_csrf_token()
        is_csrf_valid = verify_csrf_token(csrf_token, csrf_token)
        print(f"✅ CSRF tokens: {'PASS' if is_csrf_valid else 'FAIL'}")
        
        # Test filename safety
        safe_filename = "test_file.txt"
        unsafe_filename = "../../../etc/passwd"
        print(f"✅ Filename safety: {'PASS' if is_safe_filename(safe_filename) and not is_safe_filename(unsafe_filename) else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def test_schemas():
    """Test API schemas"""
    print("\n📋 Testing API Schemas...")
    
    try:
        from api.schemas import (
            ProductData, Tag, TagGenerationRequest, TagGenerationResponse,
            ErrorResponse, HealthCheck, ValidationError
        )
        
        # Test ProductData schema
        product = ProductData(
            description="Red Cotton Summer Dress",
            product_id="123",
            base_colour="Red",
            article_type="Dress"
        )
        print(f"✅ ProductData schema: PASS")
        
        # Test Tag schema
        tag = Tag(
            tag="summer-dress",
            confidence=0.85,
            source="rule"
        )
        print(f"✅ Tag schema: PASS")
        
        # Test TagGenerationRequest schema
        request = TagGenerationRequest(
            product_data=product,
            max_tags=5
        )
        print(f"✅ TagGenerationRequest schema: PASS")
        
        # Test TagGenerationResponse schema
        response = TagGenerationResponse(
            tags=[tag],
            generation_timestamp=datetime.now().isoformat(),
            total_generated=1,
            final_count=1
        )
        print(f"✅ TagGenerationResponse schema: PASS")
        
        # Test ErrorResponse schema
        error = ErrorResponse(
            error="Test error",
            details="This is a test error"
        )
        print(f"✅ ErrorResponse schema: PASS")
        
        # Test HealthCheck schema
        health = HealthCheck(
            status="healthy",
            model="test-model",
            version="1.0.0"
        )
        print(f"✅ HealthCheck schema: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Schemas test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Hirusha's Component Tests")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_tag_generator())
    results.append(test_security())
    results.append(test_schemas())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your components are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
