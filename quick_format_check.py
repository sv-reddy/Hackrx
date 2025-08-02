#!/usr/bin/env python3
"""
Quick HackRX Format Checker
Simple validation tool for input/output format compliance
"""

import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def quick_format_check():
    """Quick format compliance check"""
    
    print("⚡ Quick HackRX Format Compliance Check")
    print("=" * 50)
    
    # Configuration
    BASE_URL = "http://localhost:8008"
    AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN")
    
    if not AUTH_TOKEN:
        print("❌ HACKRX_AUTH_TOKEN not found in environment")
        return False
    
    # Test request (simplified)
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    try:
        print("🔍 Testing format compliance...")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response format
            checks = []
            
            # 1. Has 'answers' field
            has_answers = "answers" in data
            checks.append(("Has 'answers' field", has_answers))
            
            # 2. 'answers' is a list
            answers_is_list = isinstance(data.get("answers"), list)
            checks.append(("'answers' is a list", answers_is_list))
            
            # 3. Correct number of answers
            answer_count_match = len(data.get("answers", [])) == len(test_request["questions"])
            checks.append(("Answer count matches questions", answer_count_match))
            
            # 4. All answers are strings
            all_strings = all(isinstance(a, str) for a in data.get("answers", []))
            checks.append(("All answers are strings", all_strings))
            
            # 5. Content-Type is JSON
            is_json = "application/json" in response.headers.get("content-type", "")
            checks.append(("Response is JSON", is_json))
            
            # Print results
            all_passed = True
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}")
                if not passed:
                    all_passed = False
            
            # Show sample response
            print(f"\n📋 Sample Response:")
            print(f"   Questions: {len(test_request['questions'])}")
            print(f"   Answers: {len(data.get('answers', []))}")
            print(f"   Format: {{'answers': [...]}}")
            
            # Sample answers
            answers = data.get("answers", [])
            for i, answer in enumerate(answers[:2]):  # Show first 2
                truncated = answer[:80] + "..." if len(answer) > 80 else answer
                print(f"   Answer {i+1}: {truncated}")
            
            if all_passed:
                print(f"\n🎉 FORMAT COMPLIANCE: PERFECT ✅")
                print("✅ Ready for HackRX submission!")
            else:
                print(f"\n⚠️  FORMAT COMPLIANCE: ISSUES FOUND ❌")
                print("❌ Fix format issues before submission")
                
            return all_passed
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the server running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_json_structure(json_obj):
    """Validate JSON structure matches HackRX spec"""
    print("\n🔍 JSON Structure Validation:")
    
    # Expected structure
    expected = {
        "type": "object",
        "required": ["answers"],
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    errors = []
    
    # Check if it's an object
    if not isinstance(json_obj, dict):
        errors.append("Response must be a JSON object")
        return False, errors
    
    # Check required field
    if "answers" not in json_obj:
        errors.append("Missing required field: 'answers'")
    else:
        answers = json_obj["answers"]
        if not isinstance(answers, list):
            errors.append("Field 'answers' must be an array")
        else:
            for i, answer in enumerate(answers):
                if not isinstance(answer, str):
                    errors.append(f"Answer {i+1} must be a string")
    
    valid = len(errors) == 0
    
    if valid:
        print("✅ JSON structure is valid")
    else:
        print("❌ JSON structure has errors:")
        for error in errors:
            print(f"   • {error}")
    
    return valid, errors

if __name__ == "__main__":
    success = quick_format_check()
    
    if success:
        print("\n" + "="*50)
        print("🏆 YOUR API IS HACKRX COMPLIANT! 🏆")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("⚠️  COMPLIANCE ISSUES DETECTED ⚠️")
        print("="*50)
