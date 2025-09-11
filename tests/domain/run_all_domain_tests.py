#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE DOMAIN VALIDATION SUITE

This script runs all domain-specific tests to validate the mathematical
and logical correctness of the anomaly-grid-py library across multiple
domains of knowledge.
"""

import sys
import os
import time
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

# Import all domain test suites
from test_domain_1_markov_chain_mathematics import TestMarkovChainMathematics
from test_domain_2_probability_theory import TestProbabilityTheory
from test_domain_4_anomaly_detection_logic import TestAnomalyDetectionLogic
from test_domain_5_sequence_analysis import TestSequenceAnalysis


class DomainValidationSuite:
    """Comprehensive domain validation test suite."""
    
    def __init__(self):
        """Initialize the validation suite."""
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_domains(self) -> Dict[str, bool]:
        """
        Run all domain tests and return results.
        
        Returns
        -------
        results : dict
            Dictionary mapping domain names to success status.
        """
        print("🔬 ANOMALY GRID COMPREHENSIVE DOMAIN VALIDATION")
        print("=" * 50)
        print("Testing mathematical and logical correctness across multiple domains")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # Define all domain tests
        domain_tests = [
            ("Domain 1: Markov Chain Mathematics", TestMarkovChainMathematics),
            ("Domain 2: Probability Theory", TestProbabilityTheory),
            ("Domain 4: Anomaly Detection Logic", TestAnomalyDetectionLogic),
            ("Domain 5: Sequence Analysis", TestSequenceAnalysis)
        ]
        
        total_domains = len(domain_tests)
        passed_domains = 0
        
        # Run each domain test
        for domain_name, test_class in domain_tests:
            print(f"\n{'='*60}")
            print(f"🧪 RUNNING {domain_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Initialize test suite
                test_suite = test_class()
                test_suite.setup_method()
                
                # Run domain-specific tests
                if hasattr(test_suite, 'run_domain_1_tests'):
                    success = test_suite.run_domain_1_tests()
                elif hasattr(test_suite, 'run_domain_2_tests'):
                    success = test_suite.run_domain_2_tests()
                elif hasattr(test_suite, 'run_domain_4_tests'):
                    success = test_suite.run_domain_4_tests()
                elif hasattr(test_suite, 'run_domain_5_tests'):
                    success = test_suite.run_domain_5_tests()
                else:
                    # Fallback: run individual test methods
                    success = self._run_individual_tests(test_suite, domain_name)
                
                self.results[domain_name] = success
                
                if success:
                    passed_domains += 1
                    print(f"\n✅ {domain_name} - ALL TESTS PASSED")
                else:
                    print(f"\n❌ {domain_name} - SOME TESTS FAILED")
                    
            except Exception as e:
                print(f"\n💥 {domain_name} - CRITICAL ERROR: {e}")
                self.results[domain_name] = False
        
        self.end_time = time.time()
        
        # Print comprehensive summary
        self._print_final_summary(passed_domains, total_domains)
        
        return self.results
    
    def _run_individual_tests(self, test_suite, domain_name: str) -> bool:
        """
        Fallback method to run individual test methods.
        
        Parameters
        ----------
        test_suite : object
            Test suite instance.
        domain_name : str
            Name of the domain being tested.
            
        Returns
        -------
        success : bool
            True if all tests passed.
        """
        # Get all test methods
        test_methods = [method for method in dir(test_suite) 
                       if method.startswith('test_') and callable(getattr(test_suite, method))]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(test_suite, method_name)
                method()
                passed_tests += 1
                print(f"✅ {method_name}")
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
        
        print(f"\n{domain_name} Results: {passed_tests}/{total_tests} tests passed")
        return passed_tests == total_tests
    
    def _print_final_summary(self, passed_domains: int, total_domains: int):
        """
        Print comprehensive final summary.
        
        Parameters
        ----------
        passed_domains : int
            Number of domains that passed all tests.
        total_domains : int
            Total number of domains tested.
        """
        elapsed_time = self.end_time - self.start_time
        
        print("\n" + "="*70)
        print("🏆 COMPREHENSIVE DOMAIN VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   • Total Domains Tested: {total_domains}")
        print(f"   • Domains Passed: {passed_domains}")
        print(f"   • Domains Failed: {total_domains - passed_domains}")
        print(f"   • Success Rate: {passed_domains/total_domains*100:.1f}%")
        print(f"   • Total Execution Time: {elapsed_time:.2f} seconds")
        
        print(f"\n📋 DETAILED RESULTS:")
        for domain_name, success in self.results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   • {domain_name}: {status}")
        
        # Provide interpretation
        print(f"\n🔍 INTERPRETATION:")
        if passed_domains == total_domains:
            print("   🎉 EXCELLENT: All domains passed!")
            print("   📈 The library demonstrates strong mathematical and logical correctness")
            print("   🚀 Ready for production use with high confidence")
        elif passed_domains >= total_domains * 0.8:
            print("   👍 GOOD: Most domains passed")
            print("   📊 The library shows solid fundamental correctness")
            print("   ⚠️  Review failed domains for potential improvements")
        elif passed_domains >= total_domains * 0.6:
            print("   ⚠️  FAIR: Some domains passed")
            print("   🔧 The library needs improvements in failed areas")
            print("   📝 Consider addressing fundamental issues before production")
        else:
            print("   ❌ POOR: Many domains failed")
            print("   🚨 The library has significant correctness issues")
            print("   🛠️  Major revisions needed before production use")
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        failed_domains = [name for name, success in self.results.items() if not success]
        
        if not failed_domains:
            print("   • Library is ready for production deployment")
            print("   • Consider performance optimization and additional edge case testing")
            print("   • Document the validated mathematical properties for users")
        else:
            print("   • Focus on improving the following domains:")
            for domain in failed_domains:
                print(f"     - {domain}")
            print("   • Review underlying algorithms and implementations")
            print("   • Consider additional training data or parameter tuning")
        
        print("\n" + "="*70)
        
        if passed_domains == total_domains:
            print("🎯 VALIDATION COMPLETE: LIBRARY MATHEMATICALLY SOUND")
        else:
            print("⚠️  VALIDATION COMPLETE: IMPROVEMENTS NEEDED")
        
        print("="*70)


def main():
    """Main execution function."""
    # Create and run validation suite
    suite = DomainValidationSuite()
    results = suite.run_all_domains()
    
    # Determine exit code
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)