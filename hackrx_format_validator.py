#!/usr/bin/env python3
"""
HackRX Format Validator
Comprehensive checker for input and output format compliance
"""

import os
import json
import requests
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HackRXFormatValidator:
    """Validates HackRX API input/output format compliance"""
    
    def __init__(self, base_url: str = "https://1hpl2234-8008.inc1.devtunnels.ms/"):
        self.base_url = base_url
        self.auth_token = os.getenv("HACKRX_AUTH_TOKEN")
        if not self.auth_token:
            raise ValueError("HACKRX_AUTH_TOKEN environment variable is required")
        
        self.required_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        
        # Reference sample from HackRX specification
        self.sample_request = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?",
                "What is the waiting period for cataract surgery?",
                "Are the medical expenses for an organ donor covered under this policy?",
                "What is the No Claim Discount (NCD) offered in this policy?",
                "Is there a benefit for preventive health check-ups?",
                "How does the policy define a 'Hospital'?",
                "What is the extent of coverage for AYUSH treatments?",
                "Are there any sub-limits on room rent and ICU charges for Plan A?"
            ]
        }
        
        self.expected_sample_answers = [
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
            "The policy has a specific waiting period of two (2) years for cataract surgery.",
            "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
            "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
            "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
            "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
            "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
            "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
        ]
    
    def validate_input_format(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input request format against HackRX specification"""
        errors = []
        warnings = []
        
        print("🔍 Validating Input Format...")
        
        # Check required fields
        if "documents" not in request_data:
            errors.append("Missing required field: 'documents'")
        else:
            if not isinstance(request_data["documents"], str):
                errors.append("Field 'documents' must be a string (URL)")
            elif not request_data["documents"].startswith("http"):
                warnings.append("Field 'documents' should be a valid HTTP/HTTPS URL")
        
        if "questions" not in request_data:
            errors.append("Missing required field: 'questions'")
        else:
            if not isinstance(request_data["questions"], list):
                errors.append("Field 'questions' must be a list")
            elif len(request_data["questions"]) == 0:
                errors.append("Field 'questions' cannot be empty")
            else:
                for i, question in enumerate(request_data["questions"]):
                    if not isinstance(question, str):
                        errors.append(f"Question {i+1} must be a string")
                    elif len(question.strip()) == 0:
                        errors.append(f"Question {i+1} cannot be empty")
        
        # Check for extra fields
        allowed_fields = {"documents", "questions"}
        extra_fields = set(request_data.keys()) - allowed_fields
        if extra_fields:
            warnings.append(f"Extra fields found: {extra_fields}")
        
        # Format validation results
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "field_count": len(request_data),
            "question_count": len(request_data.get("questions", []))
        }
        
        if result["valid"]:
            print("✅ Input format is valid")
        else:
            print(f"❌ Input format has {len(errors)} errors")
            
        for error in errors:
            print(f"   ❌ Error: {error}")
        for warning in warnings:
            print(f"   ⚠️  Warning: {warning}")
            
        return result
    
    def validate_output_format(self, response_data: Dict[str, Any], expected_question_count: int) -> Dict[str, Any]:
        """Validate output response format against HackRX specification"""
        errors = []
        warnings = []
        
        print("🔍 Validating Output Format...")
        
        # Check required field: answers
        if "answers" not in response_data:
            errors.append("Missing required field: 'answers'")
        else:
            answers = response_data["answers"]
            if not isinstance(answers, list):
                errors.append("Field 'answers' must be a list")
            else:
                # Check answer count matches question count
                if len(answers) != expected_question_count:
                    errors.append(f"Answer count ({len(answers)}) doesn't match question count ({expected_question_count})")
                
                # Check each answer is a string
                for i, answer in enumerate(answers):
                    if not isinstance(answer, str):
                        errors.append(f"Answer {i+1} must be a string, got {type(answer).__name__}")
                    elif len(answer.strip()) == 0:
                        warnings.append(f"Answer {i+1} is empty")
                    elif answer.strip() == "unable to process":
                        warnings.append(f"Answer {i+1} is 'unable to process'")
        
        # Check for metadata (optional but expected)
        if "metadata" in response_data:
            metadata = response_data["metadata"]
            if not isinstance(metadata, dict):
                warnings.append("Field 'metadata' should be a dictionary")
        
        # Check for extra top-level fields
        allowed_fields = {"answers", "metadata"}
        extra_fields = set(response_data.keys()) - allowed_fields
        if extra_fields:
            warnings.append(f"Extra top-level fields: {extra_fields}")
        
        # Format validation results
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "answer_count": len(response_data.get("answers", [])),
            "has_metadata": "metadata" in response_data,
            "all_answers_strings": all(isinstance(a, str) for a in response_data.get("answers", []))
        }
        
        if result["valid"]:
            print("✅ Output format is valid")
        else:
            print(f"❌ Output format has {len(errors)} errors")
            
        for error in errors:
            print(f"   ❌ Error: {error}")
        for warning in warnings:
            print(f"   ⚠️  Warning: {warning}")
            
        return result
    
    def validate_headers(self, response_headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate HTTP response headers"""
        print("🔍 Validating HTTP Headers...")
        
        content_type = response_headers.get("content-type", "").lower()
        is_json = "application/json" in content_type
        
        result = {
            "content_type_valid": is_json,
            "content_type": content_type
        }
        
        if is_json:
            print("✅ Content-Type is application/json")
        else:
            print(f"❌ Content-Type should be application/json, got: {content_type}")
            
        return result
    
    def test_api_endpoint(self) -> Dict[str, Any]:
        """Test the actual API endpoint with sample data"""
        print("🚀 Testing API Endpoint...")
        
        endpoint_url = f"{self.base_url}/hackrx/run"
        
        try:
            # Test the API call
            start_time = time.time()
            response = requests.post(
                endpoint_url,
                headers=self.required_headers,
                json=self.sample_request,
                timeout=60
            )
            end_time = time.time()
            
            # Basic response validation
            response_time = end_time - start_time
            status_code = response.status_code
            
            print(f"📊 Response Status: {status_code}")
            print(f"⏱️  Response Time: {response_time:.2f}s")
            
            if status_code == 200:
                try:
                    response_data = response.json()
                    print("✅ Valid JSON response received")
                    
                    # Validate formats
                    input_validation = self.validate_input_format(self.sample_request)
                    output_validation = self.validate_output_format(response_data, len(self.sample_request["questions"]))
                    header_validation = self.validate_headers(dict(response.headers))
                    
                    return {
                        "success": True,
                        "status_code": status_code,
                        "response_time": response_time,
                        "input_validation": input_validation,
                        "output_validation": output_validation,
                        "header_validation": header_validation,
                        "response_data": response_data
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
                    return {
                        "success": False,
                        "error": f"Invalid JSON response: {e}",
                        "status_code": status_code,
                        "response_time": response_time
                    }
            else:
                print(f"❌ HTTP Error: {status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    print(f"Error response: {response.text}")
                
                return {
                    "success": False,
                    "error": f"HTTP {status_code}",
                    "status_code": status_code,
                    "response_time": response_time
                }
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return {
                "success": False,
                "error": "Request timeout"
            }
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - is the server running?")
            return {
                "success": False,
                "error": "Connection error"
            }
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error scenarios"""
        print("\n🧪 Testing Edge Cases...")
        
        edge_cases = [
            {
                "name": "Single Question",
                "data": {
                    "documents": self.sample_request["documents"],
                    "questions": ["What is the grace period for premium payment?"]
                }
            },
            {
                "name": "Empty Questions List",
                "data": {
                    "documents": self.sample_request["documents"],
                    "questions": []
                }
            },
            {
                "name": "Very Long Question",
                "data": {
                    "documents": self.sample_request["documents"],
                    "questions": ["What " + "is " * 100 + "the grace period for premium payment?"]
                }
            }
        ]
        
        results = {}
        
        for case in edge_cases:
            print(f"\n--- Testing: {case['name']} ---")
            
            try:
                # Validate input format first
                input_validation = self.validate_input_format(case['data'])
                
                if input_validation['valid']:
                    response = requests.post(
                        f"{self.base_url}/hackrx/run",
                        headers=self.required_headers,
                        json=case['data'],
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        output_validation = self.validate_output_format(
                            response_data, 
                            len(case['data']['questions'])
                        )
                        
                        results[case['name']] = {
                            "input_valid": True,
                            "api_success": True,
                            "output_valid": output_validation['valid']
                        }
                        print(f"✅ {case['name']}: Passed")
                    else:
                        results[case['name']] = {
                            "input_valid": True,
                            "api_success": False,
                            "status_code": response.status_code
                        }
                        print(f"❌ {case['name']}: API returned {response.status_code}")
                else:
                    results[case['name']] = {
                        "input_valid": False,
                        "errors": input_validation['errors']
                    }
                    print(f"⚠️  {case['name']}: Invalid input format")
                    
            except Exception as e:
                results[case['name']] = {
                    "error": str(e)
                }
                print(f"❌ {case['name']}: Exception - {e}")
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete format validation suite"""
        print("🎯 HackRX Format Validation Suite")
        print("=" * 60)
        print(f"Testing API: {self.base_url}/hackrx/run")
        print(f"Auth Token: {self.auth_token[:20]}...")
        print("=" * 60)
        
        # Test main endpoint
        main_test = self.test_api_endpoint()
        
        # Test edge cases if main test passed
        edge_test_results = {}
        if main_test.get("success"):
            edge_test_results = self.test_edge_cases()
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT")
        print("=" * 60)
        
        if main_test.get("success"):
            input_valid = main_test["input_validation"]["valid"]
            output_valid = main_test["output_validation"]["valid"]
            headers_valid = main_test["header_validation"]["content_type_valid"]
            
            print(f"✅ API Endpoint: Working")
            print(f"✅ Authentication: Valid")
            print(f"✅ Input Format: {'Valid' if input_valid else 'Invalid'}")
            print(f"✅ Output Format: {'Valid' if output_valid else 'Invalid'}")
            print(f"✅ HTTP Headers: {'Valid' if headers_valid else 'Invalid'}")
            print(f"⏱️  Response Time: {main_test['response_time']:.2f}s")
            
            # Compliance score
            compliance_checks = [input_valid, output_valid, headers_valid]
            compliance_score = sum(compliance_checks) / len(compliance_checks) * 100
            
            print(f"🏆 Format Compliance: {compliance_score:.0f}%")
            
            if compliance_score == 100:
                print("🎉 PERFECT COMPLIANCE - Ready for HackRX submission!")
            elif compliance_score >= 80:
                print("✅ GOOD COMPLIANCE - Minor issues to address")
            else:
                print("⚠️  NEEDS IMPROVEMENT - Address format issues")
                
        else:
            print(f"❌ API Endpoint: Failed - {main_test.get('error')}")
            print("🚫 Cannot validate formats - fix API issues first")
        
        # Summary
        return {
            "main_test": main_test,
            "edge_tests": edge_test_results,
            "overall_success": main_test.get("success", False)
        }

def main():
    """Main function to run format validation"""
    try:
        validator = HackRXFormatValidator()
        results = validator.run_comprehensive_validation()
        
        if results["overall_success"]:
            print("\n🎯 FORMAT VALIDATION: PASSED ✅")
        else:
            print("\n🎯 FORMAT VALIDATION: FAILED ❌")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()
