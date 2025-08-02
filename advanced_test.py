"""
Advanced RAG System Comprehensive Test
Focus: Maximum accuracy evaluation with performance monitoring
"""

import os
import requests
import json
import time
import statistics
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_advanced_rag():
    """Comprehensive test of the advanced RAG system"""
    
    base_url = "https://1hpl2234-8008.inc1.devtunnels.ms/"
    
    # Get auth token from environment
    auth_token = os.getenv("HACKRX_AUTH_TOKEN")
    if not auth_token:
        raise ValueError("HACKRX_AUTH_TOKEN environment variable is required")
    
    # Authorization header as per HackRX requirements
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    test_cases = [
        {
            "question": "What is the exact name of this insurance policy?",
            "type": "definition",
            "expected_keywords": ["national", "parivar", "mediclaim", "plus", "policy"],
            "difficulty": "basic"
        },
        {
            "question": "What is the grace period for premium payment?",
            "type": "quantitative",
            "expected_keywords": ["grace", "period", "premium", "thirty", "days"],
            "difficulty": "basic"
        },
        {
            "question": "What are the specific exclusions for domiciliary hospitalization?",
            "type": "exclusions",
            "expected_keywords": ["domiciliary", "exclusions", "treatment", "three", "days"],
            "difficulty": "advanced"
        },
        {
            "question": "How is notification of claim defined and what is the process?",
            "type": "procedural",
            "expected_keywords": ["notification", "claim", "process", "intimating", "company"],
            "difficulty": "intermediate"
        },
        {
            "question": "What medical expenses are covered under this policy?",
            "type": "coverage",
            "expected_keywords": ["medical", "expenses", "covered", "treatment", "hospital"],
            "difficulty": "intermediate"
        },
        {
            "question": "What is the waiting period for pre-existing diseases?",
            "type": "quantitative",
            "expected_keywords": ["waiting", "period", "pre-existing", "thirty-six", "months"],
            "difficulty": "intermediate"
        },
        {
            "question": "Are organ donor medical expenses covered and under what conditions?",
            "type": "coverage",
            "expected_keywords": ["organ", "donor", "medical", "expenses", "covered"],
            "difficulty": "advanced"
        },
        {
            "question": "What are the room rent and ICU charge limits for Plan A?",
            "type": "quantitative",
            "expected_keywords": ["room", "rent", "icu", "charges", "percent", "sum"],
            "difficulty": "advanced"
        },
        {
            "question": "How does the policy define a Hospital?",
            "type": "definition",
            "expected_keywords": ["hospital", "beds", "nursing", "medical", "practitioners"],
            "difficulty": "intermediate"
        },
        {
            "question": "What is the No Claim Discount offered and how does it work?",
            "type": "procedural",
            "expected_keywords": ["no", "claim", "discount", "premium", "renewal"],
            "difficulty": "intermediate"
        }
    ]
    
    print("🚀 ADVANCED RAG SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Testing against: {base_url}")
    print(f"Focus: Maximum Accuracy + <10s Latency")
    print(f"Test Questions: {len(test_cases)}")
    print("=" * 60)
    
    # Health check
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        health_data = health_response.json()
        print(f"✅ System Status: {health_data.get('message', 'Unknown')}")
        print(f"📊 Cache Stats: {health_data.get('cache_stats', {})}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return
    
    # Clear cache for clean test
    try:
        requests.get(f"{base_url}/cache/clear", timeout=5)
        print("🧹 Cache cleared for clean test")
    except:
        pass
    
    # Initial document processing test
    print("\n📄 Testing Document Processing...")
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": ["What type of insurance policy is this?"]
    }
    
    try:
        response = requests.post(f"{base_url}/hackrx/run", json=test_payload, headers=headers, timeout=90)
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            answer_text = answers[0] if answers else ""
            metadata = result.get("metadata", {})
            print(f"✅ Document processed: {len(answer_text)} chars response")
            print(f"📊 Chunks created: {metadata.get('total_chunks', 'N/A')}")
            print(f"⚡ Processing time: {metadata.get('processing_time_seconds', 0):.2f}s")
        else:
            print(f"❌ Document processing failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return
    
    # Run comprehensive accuracy tests
    print("\n🎯 Running Accuracy & Performance Tests...")
    
    results = []
    latencies = []
    accuracy_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)}: {test_case['type'].upper()} ({test_case['difficulty']}) ---")
        print(f"Q: {test_case['question']}")
        
        try:
            start_time = time.time()
            payload = {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [test_case["question"]]
            }
            
            response = requests.post(f"{base_url}/hackrx/run", json=payload, headers=headers, timeout=30)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get("answers", [])
                answer = answers[0] if answers else ""
                metadata = result.get("metadata", {})
                
                # Enhanced accuracy calculation
                keyword_matches = sum(1 for keyword in test_case["expected_keywords"] 
                                    if keyword.lower() in answer.lower())
                keyword_score = (keyword_matches / len(test_case["expected_keywords"])) * 100
                
                # Query type bonus
                type_bonus = 0
                answer_lower = answer.lower()
                
                if test_case["type"] == "definition" and any(term in answer_lower for term in ["defined", "definition", "means"]):
                    type_bonus = 5
                elif test_case["type"] == "quantitative" and any(char.isdigit() for char in answer):
                    type_bonus = 10
                elif test_case["type"] == "procedural" and any(term in answer_lower for term in ["process", "procedure", "step"]):
                    type_bonus = 5
                elif test_case["type"] == "coverage" and any(term in answer_lower for term in ["covered", "includes", "benefits"]):
                    type_bonus = 5
                elif test_case["type"] == "exclusions" and any(term in answer_lower for term in ["excluded", "not covered", "limitations"]):
                    type_bonus = 5
                
                # Specificity bonus (penalize generic answers)
                specificity_bonus = 0
                if "not specified" not in answer_lower and "not available" not in answer_lower:
                    if len(answer) > 50:  # Substantial answer
                        specificity_bonus = 5
                
                # Calculate final accuracy
                final_accuracy = min(100, keyword_score + type_bonus + specificity_bonus)
                accuracy_scores.append(final_accuracy)
                
                # Performance assessment
                if latency <= 5.0:
                    speed_rating = "🟢 Excellent"
                elif latency <= 8.0:
                    speed_rating = "🟡 Good"
                elif latency <= 10.0:
                    speed_rating = "🟠 Acceptable"
                else:
                    speed_rating = "🔴 Slow"
                
                print(f"✅ Response time: {latency:.2f}s {speed_rating}")
                print(f"📊 Keyword matches: {keyword_score:.1f}% ({keyword_matches}/{len(test_case['expected_keywords'])})")
                if type_bonus > 0:
                    print(f"🎯 Type bonus: +{type_bonus}%")
                if specificity_bonus > 0:
                    print(f"💎 Specificity bonus: +{specificity_bonus}%")
                print(f"🏆 Final accuracy: {final_accuracy:.1f}%")
                print(f"📝 Answer: {answer[:120]}...")
                
                results.append({
                    "question": test_case["question"],
                    "type": test_case["type"],
                    "difficulty": test_case["difficulty"],
                    "accuracy": final_accuracy,
                    "latency": latency,
                    "keyword_score": keyword_score,
                    "type_bonus": type_bonus,
                    "specificity_bonus": specificity_bonus,
                    "cache_hits": metadata.get("cache_hits", 0)
                })
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Comprehensive analysis
    if results:
        avg_accuracy = statistics.mean(accuracy_scores)
        median_accuracy = statistics.median(accuracy_scores)
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        
        print("\n" + "=" * 60)
        print("🏆 ADVANCED RAG SYSTEM RESULTS")
        print("=" * 60)
        
        # Accuracy Analysis
        print(f"🎯 ACCURACY METRICS:")
        print(f"  • Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  • Median Accuracy: {median_accuracy:.1f}%")
        print(f"  • Min Accuracy: {min(accuracy_scores):.1f}%")
        print(f"  • Max Accuracy: {max(accuracy_scores):.1f}%")
        print(f"  • Std Deviation: {statistics.stdev(accuracy_scores):.1f}%")
        
        # Performance Analysis
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  • Average Latency: {avg_latency:.2f}s")
        print(f"  • Median Latency: {median_latency:.2f}s")
        print(f"  • Min Latency: {min(latencies):.2f}s")
        print(f"  • Max Latency: {max(latencies):.2f}s")
        print(f"  • Questions under 5s: {sum(1 for l in latencies if l <= 5.0)}/{len(latencies)}")
        print(f"  • Questions under 10s: {sum(1 for l in latencies if l <= 10.0)}/{len(latencies)}")
        
        # Query Type Analysis
        type_performance = {}
        for result in results:
            query_type = result["type"]
            if query_type not in type_performance:
                type_performance[query_type] = []
            type_performance[query_type].append(result["accuracy"])
        
        print(f"\n📊 QUERY TYPE PERFORMANCE:")
        for query_type, accuracies in type_performance.items():
            avg_type_accuracy = statistics.mean(accuracies)
            print(f"  • {query_type.title()}: {avg_type_accuracy:.1f}% avg ({len(accuracies)} tests)")
        
        # Difficulty Analysis
        difficulty_performance = {}
        for result in results:
            difficulty = result["difficulty"]
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = []
            difficulty_performance[difficulty].append(result["accuracy"])
        
        print(f"\n🎚️ DIFFICULTY ANALYSIS:")
        for difficulty, accuracies in difficulty_performance.items():
            avg_diff_accuracy = statistics.mean(accuracies)
            print(f"  • {difficulty.title()}: {avg_diff_accuracy:.1f}% avg ({len(accuracies)} tests)")
        
        # Final Assessment
        print(f"\n🏅 FINAL SYSTEM ASSESSMENT:")
        
        # Accuracy grade
        if avg_accuracy >= 90:
            accuracy_grade = "🟢 Excellent"
        elif avg_accuracy >= 80:
            accuracy_grade = "🟡 Good"
        elif avg_accuracy >= 70:
            accuracy_grade = "🟠 Fair"
        else:
            accuracy_grade = "🔴 Needs Improvement"
        
        # Performance grade
        if avg_latency <= 5.0:
            performance_grade = "🟢 Excellent"
        elif avg_latency <= 8.0:
            performance_grade = "🟡 Good"
        elif avg_latency <= 10.0:
            performance_grade = "🟠 Acceptable"
        else:
            performance_grade = "🔴 Too Slow"
        
        print(f"  🎯 Accuracy: {accuracy_grade} ({avg_accuracy:.1f}%)")
        print(f"  ⚡ Performance: {performance_grade} ({avg_latency:.2f}s avg)")
        
        # Overall score
        accuracy_weight = 0.7
        performance_weight = 0.3
        
        # Normalize performance score (10s = 0%, 0s = 100%)
        performance_score = max(0, (10 - avg_latency) / 10 * 100)
        overall_score = (accuracy_weight * avg_accuracy) + (performance_weight * performance_score)
        
        print(f"  🏆 Overall Score: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            status = "🟢 Production Ready"
            recommendation = "Deploy immediately - excellent performance"
        elif overall_score >= 75:
            status = "🟡 Near Production"
            recommendation = "Minor optimizations recommended"
        elif overall_score >= 65:
            status = "🟠 Development Stage"
            recommendation = "Requires improvement before production"
        else:
            status = "🔴 Needs Significant Work"
            recommendation = "Major improvements required"
        
        print(f"  📋 Status: {status}")
        print(f"  💡 Recommendation: {recommendation}")
        
        return {
            "average_accuracy": avg_accuracy,
            "average_latency": avg_latency,
            "overall_score": overall_score,
            "tests_completed": len(results),
            "production_ready": overall_score >= 85
        }
    
    else:
        print("❌ No results to analyze")
        return None

if __name__ == "__main__":
    print(f"🕒 Test started at: {datetime.now().strftime('%H:%M:%S')}")
    results = test_advanced_rag()
    if results:
        print(f"\n🏁 FINAL RESULT: {results['average_accuracy']:.1f}% accuracy, {results['average_latency']:.2f}s avg latency")
    print(f"🕒 Test completed at: {datetime.now().strftime('%H:%M:%S')}")
