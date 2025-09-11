#!/usr/bin/env python3
"""
Realistic examples demonstrating the anomaly-grid-py Python package

This example shows practical use cases with meaningful data that demonstrates
the actual value of anomaly detection in real-world scenarios.
"""

import anomaly_grid_py
import time
import random
import numpy as np

def web_server_log_analysis():
    """Example 1: Web Server Log Analysis"""
    print("🌐 WEB SERVER LOG ANALYSIS")
    print("=" * 50)
    
    # Create detector for web server logs
    detector = anomaly_grid_py.AnomalyDetector(max_order=4)
    
    # Simulate normal web server access patterns (1000 requests)
    normal_patterns = [
        ["GET", "/", "200"],
        ["GET", "/login", "200"],
        ["POST", "/login", "302"],
        ["GET", "/dashboard", "200"],
        ["GET", "/api/users", "200"],
        ["GET", "/profile", "200"],
        ["POST", "/logout", "302"],
        ["GET", "/", "200"],
        ["GET", "/about", "200"],
        ["GET", "/contact", "200"],
        ["GET", "/products", "200"],
        ["GET", "/api/products", "200"],
    ]
    
    # Generate training data with realistic patterns
    training_data = []
    for _ in range(200):  # 200 sessions
        session_length = random.randint(3, 8)
        for _ in range(session_length):
            pattern = random.choice(normal_patterns)
            training_data.extend(pattern)
    
    print(f"📚 Training with {len(training_data)} log entries...")
    start_time = time.time()
    # Convert flat list to list of sequences for the new API
    sequences = []
    current_seq = []
    for item in training_data:
        current_seq.append(item)
        if len(current_seq) >= 5:  # Create sequences of length 5
            sequences.append(current_seq.copy())
            current_seq = current_seq[1:]  # Sliding window
    
    detector.fit(sequences)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.4f}s")
    
    # Test normal user behavior
    print("\n🟢 Testing Normal User Session:")
    normal_session = ["GET", "/", "200", "GET", "/login", "200", "POST", "/login", "302", "GET", "/dashboard", "200", "GET", "/profile", "200"]
    # Convert to sequence format
    normal_sequences = [normal_session[i:i+5] for i in range(len(normal_session)-4)]
    normal_scores = detector.predict_proba(normal_sequences)
    normal_anomalies = sum(1 for score in normal_scores if score > 0.1)
    print(f"   Session: {' → '.join(normal_session[:5])}...")
    print(f"   Anomalies detected: {normal_anomalies}")
    
    # Test suspicious behavior patterns
    print("\n🔴 Testing Suspicious Behaviors:")
    
    # SQL Injection attempt
    sql_injection = ["GET", "/", "200", "GET", "/login", "200", "POST", "/login", "500", "GET", "/admin", "403", "GET", "/admin/users", "403"]
    sql_sequences = [sql_injection[i:i+5] for i in range(len(sql_injection)-4)]
    sql_scores = detector.predict_proba(sql_sequences)
    sql_anomalies = sum(1 for score in sql_scores if score > 0.1)
    print(f"   SQL Injection Pattern: {sql_anomalies} anomalies detected")
    if sql_anomalies > 0:
        max_score = max(sql_scores)
        print(f"     🚨 Alert: Max anomaly score: {max_score:.3f}")
    
    # Brute force attack
    brute_force = ["POST", "/login", "401"] * 10 + ["GET", "/admin", "403"]
    brute_sequences = [brute_force[i:i+5] for i in range(len(brute_force)-4)]
    brute_scores = detector.predict_proba(brute_sequences)
    brute_anomalies = sum(1 for score in brute_scores if score > 0.1)
    print(f"   Brute Force Pattern: {brute_anomalies} anomalies detected")
    
    # Directory traversal
    directory_traversal = ["GET", "/", "200", "GET", "/../etc/passwd", "404", "GET", "/../admin", "404", "GET", "/admin/config", "403"]
    traversal_sequences = [directory_traversal[i:i+5] for i in range(len(directory_traversal)-4)]
    traversal_scores = detector.predict_proba(traversal_sequences)
    traversal_anomalies = sum(1 for score in traversal_scores if score > 0.1)
    print(f"   Directory Traversal: {traversal_anomalies} anomalies detected")
    
    # Performance metrics
    metrics = detector.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"   Training time: {metrics['training_time_ms']} ms")
    print(f"   Memory usage: {metrics['memory_bytes'] / 1024:.1f} KB")
    print(f"   Context patterns: {metrics['context_count']}")

def user_behavior_analysis():
    """Example 2: User Behavior Analysis"""
    print("\n\n👤 USER BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    detector = anomaly_grid_py.AnomalyDetector(max_order=3)
    
    # Normal user behavior patterns
    normal_behaviors = [
        ["login", "view_dashboard", "view_profile", "logout"],
        ["login", "view_dashboard", "view_reports", "download_report", "logout"],
        ["login", "view_dashboard", "edit_profile", "save_profile", "logout"],
        ["login", "view_dashboard", "view_settings", "change_password", "logout"],
        ["login", "view_dashboard", "create_document", "edit_document", "save_document", "logout"],
        ["login", "view_dashboard", "view_analytics", "export_data", "logout"],
    ]
    
    # Generate training data (500 user sessions)
    training_data = []
    for _ in range(500):
        behavior = random.choice(normal_behaviors)
        training_data.extend(behavior)
        # Add some variation
        if random.random() < 0.3:
            training_data.extend(["view_help", "logout"])
    
    print(f"📚 Training with {len(training_data)} user actions...")
    # Convert to sequences
    sequences = []
    current_seq = []
    for item in training_data:
        current_seq.append(item)
        if len(current_seq) >= 4:  # Create sequences of length 4
            sequences.append(current_seq.copy())
            current_seq = current_seq[1:]  # Sliding window
    
    detector.fit(sequences)
    
    # Test scenarios
    scenarios = {
        "Normal User": ["login", "view_dashboard", "view_profile", "edit_profile", "logout"],
        "Data Exfiltration": ["login", "download_report", "download_report", "download_report", "export_data", "export_data"],
        "Privilege Escalation": ["login", "view_dashboard", "access_admin_panel", "create_admin_user", "delete_logs"],
        "Account Takeover": ["login", "change_password", "change_email", "delete_security_questions", "logout"],
        "Insider Threat": ["login", "view_all_users", "export_user_data", "access_financial_data", "download_database"]
    }
    
    print("\n🔍 Analyzing User Behavior Patterns:")
    for scenario_name, actions in scenarios.items():
        # Convert to sequences for analysis
        action_sequences = [actions[i:i+4] for i in range(len(actions)-3)] if len(actions) >= 4 else [actions]
        scores = detector.predict_proba(action_sequences)
        anomalies = sum(1 for score in scores if score > 0.1)
        risk_level = "🟢 LOW" if anomalies <= 1 else "🟡 MEDIUM" if anomalies <= 3 else "🔴 HIGH"
        
        print(f"\n{risk_level} {scenario_name}:")
        print(f"   Actions: {' → '.join(actions)}")
        print(f"   Risk Score: {anomalies}/{len(action_sequences)} anomalous sequences")
        
        if anomalies > 0:
            max_score = max(scores)
            print(f"   🚨 Alert: Max anomaly score: {max_score:.1%}")

def iot_sensor_monitoring():
    """Example 3: IoT Sensor Data Monitoring"""
    print("\n\n🌡️ IOT SENSOR DATA MONITORING")
    print("=" * 50)
    
    detector = anomaly_grid_py.AnomalyDetector(max_order=3)
    
    # Simulate normal sensor readings (temperature, humidity, pressure states)
    normal_states = ["temp_normal", "humidity_normal", "pressure_normal"]
    seasonal_patterns = [
        ["temp_high", "humidity_high", "pressure_normal"],  # Summer
        ["temp_low", "humidity_low", "pressure_high"],      # Winter
        ["temp_normal", "humidity_high", "pressure_low"],   # Spring/Fall
    ]
    
    # Generate training data (2000 sensor readings)
    training_data = []
    for _ in range(2000):
        if random.random() < 0.7:
            # Normal conditions
            training_data.extend(normal_states)
        else:
            # Seasonal variations
            pattern = random.choice(seasonal_patterns)
            training_data.extend(pattern)
    
    print(f"📚 Training with {len(training_data)} sensor readings...")
    # Convert to sequences
    sequences = []
    current_seq = []
    for item in training_data:
        current_seq.append(item)
        if len(current_seq) >= 3:  # Create sequences of length 3
            sequences.append(current_seq.copy())
            current_seq = current_seq[1:]  # Sliding window
    
    detector.fit(sequences)
    
    # Test different scenarios
    print("\n🔍 Monitoring Sensor Anomalies:")
    
    # Normal operation
    normal_readings = ["temp_normal", "humidity_normal", "pressure_normal"] * 3
    normal_sequences = [normal_readings[i:i+3] for i in range(len(normal_readings)-2)]
    normal_scores = detector.predict_proba(normal_sequences)
    normal_anomalies = sum(1 for score in normal_scores if score > 0.1)
    print(f"🟢 Normal Operation: {normal_anomalies} anomalies in {len(normal_readings)} readings")
    
    # Equipment malfunction
    malfunction = ["temp_critical", "temp_critical", "humidity_critical", "pressure_critical", "sensor_error"]
    malfunction_sequences = [malfunction[i:i+3] for i in range(len(malfunction)-2)]
    malfunction_scores = detector.predict_proba(malfunction_sequences)
    malfunction_anomalies = sum(1 for score in malfunction_scores if score > 0.1)
    print(f"🔴 Equipment Malfunction: {malfunction_anomalies} anomalies detected")
    
    # Gradual degradation
    degradation = ["temp_normal", "temp_high", "temp_critical", "humidity_normal", "humidity_high", "sensor_drift"]
    degradation_sequences = [degradation[i:i+3] for i in range(len(degradation)-2)]
    degradation_scores = detector.predict_proba(degradation_sequences)
    degradation_anomalies = sum(1 for score in degradation_scores if score > 0.1)
    print(f"🟡 Gradual Degradation: {degradation_anomalies} anomalies detected")
    
    # Environmental event
    storm = ["pressure_low", "pressure_critical", "humidity_high", "temp_low", "wind_high", "power_fluctuation"]
    storm_sequences = [storm[i:i+3] for i in range(len(storm)-2)]
    storm_scores = detector.predict_proba(storm_sequences)
    storm_anomalies = sum(1 for score in storm_scores if score > 0.1)
    print(f"⛈️ Storm Event: {storm_anomalies} anomalies detected")
    
    if storm_anomalies > 0:
        max_score = max(storm_scores)
        max_idx = np.argmax(storm_scores)
        print(f"   🌪️ Weather Alert: '{storm_sequences[max_idx]}' (strength: {max_score:.3f})")

def network_traffic_analysis():
    """Example 4: Network Traffic Analysis"""
    print("\n\n🌐 NETWORK TRAFFIC ANALYSIS")
    print("=" * 50)
    
    detector = anomaly_grid_py.AnomalyDetector(max_order=4)
    
    # Normal network traffic patterns
    normal_traffic = [
        ["tcp_syn", "tcp_ack", "http_request", "http_response", "tcp_fin"],
        ["tcp_syn", "tcp_ack", "https_request", "https_response", "tcp_fin"],
        ["udp_dns_query", "udp_dns_response"],
        ["tcp_syn", "tcp_ack", "smtp_connect", "smtp_auth", "smtp_send", "smtp_close", "tcp_fin"],
        ["tcp_syn", "tcp_ack", "ssh_connect", "ssh_auth", "ssh_session", "ssh_close", "tcp_fin"],
    ]
    
    # Generate training data (1500 network flows)
    training_data = []
    for _ in range(1500):
        flow = random.choice(normal_traffic)
        training_data.extend(flow)
    
    print(f"📚 Training with {len(training_data)} network events...")
    start_time = time.time()
    # Convert to sequences
    sequences = []
    current_seq = []
    for item in training_data:
        current_seq.append(item)
        if len(current_seq) >= 4:  # Create sequences of length 4
            sequences.append(current_seq.copy())
            current_seq = current_seq[1:]  # Sliding window
    
    detector.fit(sequences)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.4f}s ({len(training_data)/training_time:.0f} events/sec)")
    
    # Test network security scenarios
    print("\n🔍 Network Security Analysis:")
    
    # Port scan detection
    port_scan = ["tcp_syn", "tcp_rst"] * 10 + ["tcp_syn", "tcp_ack", "service_banner"]
    scan_sequences = [port_scan[i:i+4] for i in range(len(port_scan)-3)]
    scan_scores = detector.predict_proba(scan_sequences)
    scan_anomalies = sum(1 for score in scan_scores if score > 0.1)
    print(f"🔴 Port Scan: {scan_anomalies} anomalies in {len(port_scan)} packets")
    
    # DDoS attack
    ddos = ["tcp_syn"] * 20 + ["tcp_timeout"] * 5
    ddos_sequences = [ddos[i:i+4] for i in range(len(ddos)-3)]
    ddos_scores = detector.predict_proba(ddos_sequences)
    ddos_anomalies = sum(1 for score in ddos_scores if score > 0.1)
    print(f"🔴 DDoS Attack: {ddos_anomalies} anomalies detected")
    
    # Data exfiltration
    exfiltration = ["tcp_syn", "tcp_ack", "large_upload", "large_upload", "large_upload", "tcp_fin"]
    exfil_sequences = [exfiltration[i:i+4] for i in range(len(exfiltration)-3)]
    exfil_scores = detector.predict_proba(exfil_sequences)
    exfil_anomalies = sum(1 for score in exfil_scores if score > 0.1)
    print(f"🟡 Data Exfiltration: {exfil_anomalies} anomalies detected")
    
    # Normal business traffic
    normal_business = ["tcp_syn", "tcp_ack", "https_request", "https_response", "tcp_fin"]
    normal_sequences = [normal_business[i:i+4] for i in range(len(normal_business)-3)]
    normal_scores = detector.predict_proba(normal_sequences)
    normal_anomalies = sum(1 for score in normal_scores if score > 0.1)
    print(f"🟢 Normal Business: {normal_anomalies} anomalies detected")
    
    # Show detailed analysis for the most suspicious activity
    if ddos_anomalies > 0:
        print(f"\n🚨 DDoS Attack Analysis:")
        high_scores = [(i, score) for i, score in enumerate(ddos_scores) if score > 0.1]
        for i, (seq_idx, score) in enumerate(high_scores[:3]):
            print(f"   Alert {i+1}: '{ddos_sequences[seq_idx]}' (threat level: {score:.1%})")

def performance_benchmark():
    """Performance benchmark with realistic data sizes"""
    print("\n\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Test with different data sizes
    sizes = [1000, 5000, 10000, 25000]
    
    for size in sizes:
        detector = anomaly_grid_py.AnomalyDetector(max_order=3)
        
        # Generate realistic event patterns
        events = []
        patterns = ["login", "action", "logout", "error", "retry", "success"]
        for _ in range(size):
            events.append(random.choice(patterns))
        
        # Measure training performance
        start_time = time.perf_counter()
        # Convert to sequences for training
        sequences = []
        for i in range(len(events) - 2):
            sequences.append(events[i:i+3])
        
        detector.fit(sequences)
        training_time = time.perf_counter() - start_time
        
        # Measure detection performance
        test_events = [random.choice(patterns) for _ in range(100)]
        start_time = time.perf_counter()
        # Convert test events to sequences
        test_sequences = []
        for i in range(len(test_events) - 2):
            test_sequences.append(test_events[i:i+3])
        
        scores = detector.predict_proba(test_sequences)
        results = sum(1 for score in scores if score > 0.1)
        detection_time = time.perf_counter() - start_time
        
        # Get memory usage
        metrics = detector.get_performance_metrics()
        memory_kb = metrics['memory_bytes'] / 1024
        
        print(f"📊 {size:5d} events: train={training_time:.4f}s ({size/training_time:6.0f} evt/s), "
              f"detect={detection_time:.4f}s, memory={memory_kb:.1f}KB")

def main():
    """Run all realistic examples"""
    print("🔍 ANOMALY DETECTION - REALISTIC EXAMPLES")
    print("=" * 60)
    print("Demonstrating practical anomaly detection with meaningful data")
    print("=" * 60)
    
    try:
        # Run all examples
        web_server_log_analysis()
        user_behavior_analysis()
        iot_sensor_monitoring()
        network_traffic_analysis()
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n💡 Key Takeaways:")
        print("   • Anomaly detection works effectively on sequential data")
        print("   • Higher anomaly counts indicate more suspicious behavior")
        print("   • Performance scales well with data size")
        print("   • Different thresholds can be tuned for sensitivity")
        print("   • Real-world patterns are successfully learned and detected")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()