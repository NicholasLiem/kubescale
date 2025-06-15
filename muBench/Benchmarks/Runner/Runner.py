from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import sched
import time
import threading
from TimingError import TimingError
import requests
import json
import sys
import os
import shutil
import importlib
from pprint import pprint

import argparse
import argcomplete



class Counter(object):
    def __init__(self, start = 0):
        self.lock = threading.Lock()
        self.value = start
    def increase(self):
        self.lock.acquire()
        try:
            self.value = self.value + 1
        finally:
            self.lock.release()
    def decrease(self):
        self.lock.acquire()
        try:
            self.value = self.value - 1
        finally:
            self.lock.release()


def do_requests(event, stats, local_latency_stats):
    global processed_requests, last_print_time_ms, error_requests, pending_requests, is_in_spike
    # pprint(workload[event]["services"])
    # for services in event["services"]:
        # print(services)
    processed_requests.increase()    
    try:
        now_ms = time.time_ns() // 1_000_000
        if runner_type=="greedy":
            pending_requests.increase()
        
        r = requests.get(f"{ms_access_gateway}/{event['service']}")
        pending_requests.decrease()
        
        if r.status_code != 200:
            print("Response Status Code", r.status_code)
            error_requests.increase()

        req_latency_ms = int(r.elapsed.total_seconds()*1000)
        stats.append(f"{now_ms} \t {req_latency_ms} \t {r.status_code} \t {processed_requests.value} \t {pending_requests.value}")
        local_latency_stats.append(req_latency_ms)
        
        if now_ms > last_print_time_ms + 1_000:
            print(f"Processed request {processed_requests.value}, latency {req_latency_ms} ms, pending requests {pending_requests.value}, status code {r.status_code}")
            last_print_time_ms = now_ms
        return event['time'], req_latency_ms
    except Exception as err:
        print("Error: %s" % err)


def job_assignment(v_pool, v_futures, event, stats, local_latency_stats):
    global timing_error_requests, pending_requests, is_in_spike
    try:
        worker = v_pool.submit(do_requests, event, stats, local_latency_stats)
        v_futures.append(worker)
        if runner_type!="greedy":
            pending_requests.increase()
        
        active_threads = len([f for f in v_futures if not f.done()])
        if pending_requests.value > threads: 
            # maximum capacity of thread pool reached, request is queued (not an issue for greedy runner)
            if runner_type!="greedy":
                timing_error_requests += 1
                # Enhanced error message with spike indicator
                spike_indicator = "🔥 DURING SPIKE 🔥" if is_in_spike else ""
                print(f"Error: All Pool Threads are busy. {spike_indicator} Active threads: {active_threads}/{threads}. It's impossible to respect the requests timing! :(")
                raise TimingError(event['time'])
    except TimingError as err:
        pass  # Error message already printed above

def file_runner(workload=None):
    global start_time, stats, local_latency_stats

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    if len(sys.argv) > 1 and workload is None:
        workload_file = sys.argv[1]
    else:
        workload_file = workload

    with open(workload_file) as f:
        workload = json.load(f)
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()

    for event in workload:
        # in seconds
        # s.enter(event["time"], 1, job_assignment, argument=(pool, futures, event))
        # in milliseconds
        s.enter((event["time"]/1000+2), 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f" % (len(workload), error_requests.value, timing_error_requests, avg_latency, 1.0*len(workload)/run_duration_sec))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests.value,
                "timing_error_number": timing_error_requests,
                "total_request": len(workload),
                "error_request": error_requests.value,
                "runner_results_file": f"{output_path}/{result_file}_{workload_var.split('/')[-1].split('.')[0]}.txt"
                }
        run_after_workload(args)

def greedy_runner():
    global start_time, stats, local_latency_stats, runner_parameters

    if 'ingress_service' in runner_parameters.keys():
        srv=runner_parameters['ingress_service']
    else:
        srv = 's0'

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event={'service':srv,'time':0}
    slow_start_end = 32 # number requests with initial delays
    slow_start_delay = 0.1
    # put every request in the thread pool scheduled at time 0 (in case with initial slow start spread to reduce initial concurrency)
    for i in range(workload_events):
        if i < slow_start_end :
            event_time =  i * slow_start_delay
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f" % (workload_events, error_requests.value, timing_error_requests, avg_latency, 1.0*workload_events/run_duration_sec))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests,
                "timing_error_number": timing_error_requests,
                "total_request": workload_events,
                "error_request": error_requests,
                "runner_results_file": f"{output_path}/{result_file}.txt"
                }
        run_after_workload(args)

def periodic_runner():
    global start_time, stats, local_latency_stats, runner_parameters, is_in_spike

    # Initialize the global spike indicator
    is_in_spike = False

    if 'rate' in runner_parameters.keys():
        rate = runner_parameters['rate']
    else:
        rate = 1
    
    if 'ingress_service' in runner_parameters.keys():
        srv = runner_parameters['ingress_service']
    else:
        srv = 's0'

    if 'minutes_to_train' in runner_parameters.keys():
        minutes_to_train = runner_parameters['minutes_to_train']
    else:
        minutes_to_train = 0
    
    # Wave pattern configuration
    wave_pattern_enabled = False
    wave_patterns = []
    baseline_interval = 120  # Default baseline interval between waves
    
    if 'wave_pattern' in runner_parameters.keys() and runner_parameters['wave_pattern'].get('enabled', False):
        wave_pattern_enabled = True
        wave_patterns = runner_parameters['wave_pattern'].get('waves', [])
        pattern_type = runner_parameters['wave_pattern'].get('pattern_type', 'sequential')
        baseline_interval = runner_parameters['wave_pattern'].get('baseline_interval', 120)
        
        print(f"Wave pattern enabled: {pattern_type}")
        print(f"Baseline interval between waves: {baseline_interval}s")
        for i, wave in enumerate(wave_patterns):
            print(f"  Wave {i+1} ({wave.get('name', f'wave_{i+1}')}): {wave.get('multiplier', 1)}x rate for {wave.get('duration', 30)}s every {wave.get('interval', 300)}s")
    
    # Fallback to legacy spike configuration if wave pattern is disabled
    if not wave_pattern_enabled:
        if 'spike_interval' in runner_parameters.keys():
            spike_interval = runner_parameters['spike_interval']
        else:
            spike_interval = 30
        
        if 'spike_multiplier' in runner_parameters.keys():
            spike_multiplier = runner_parameters['spike_multiplier']
        else:
            spike_multiplier = 5
            
        if 'spike_duration' in runner_parameters.keys():
            spike_duration = runner_parameters['spike_duration']
        else:
            spike_duration = 5
    
    if 'debug_logging' in runner_parameters.keys():
        debug_logging = runner_parameters['debug_logging']
    else:
        debug_logging = True
    
    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    
    # Convert minutes_to_train to seconds
    seconds_to_train = minutes_to_train * 60
    
    if minutes_to_train > 0:
        print(f"Training phase: {minutes_to_train} minutes ({seconds_to_train} seconds) at baseline rate: {rate} req/sec")
    
    if wave_pattern_enabled:
        max_multiplier = max([wave.get('multiplier', 1) for wave in wave_patterns])
        print(f"Base rate: {rate} req/sec, Max spike rate: {rate * max_multiplier} req/sec")
    else:
        print(f"Base rate: {rate} req/sec, Spike rate: {rate * spike_multiplier} req/sec")
        print(f"Spike interval: {spike_interval}s, Spike duration: {spike_duration}s")
    
    print(f"Thread pool size: {threads}")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event = {'service': srv, 'time': 0}
    offset = 10
    
    # Track status for logging
    last_spike_state = False
    last_status_time = time.time()
    status_interval = 5
    current_wave_info = {"active": False, "name": "baseline", "multiplier": 1}
    
    def get_current_wave_multiplier(current_time, in_training_phase):
        """Calculate the current rate multiplier based on wave patterns with baseline intervals"""
        if in_training_phase or not wave_pattern_enabled:
            if not wave_pattern_enabled and not in_training_phase:
                # Use legacy spike logic
                is_spike_period = ((current_time - seconds_to_train) % spike_interval) < spike_duration
                return spike_multiplier if is_spike_period else 1, is_spike_period, "legacy_spike" if is_spike_period else "baseline"
            return 1, False, "training" if in_training_phase else "baseline"
        
        time_since_start = current_time - seconds_to_train
        active_waves = []
        
        # Check each wave pattern with offset support for sequential pattern
        for i, wave in enumerate(wave_patterns):
            interval = wave.get('interval', 300)
            duration = wave.get('duration', 30)
            multiplier = wave.get('multiplier', 1)
            name = wave.get('name', f'wave_{i+1}')
            offset = wave.get('offset', 0)  # New: support for wave offset
            
            # Calculate time position within this wave's cycle including offset
            time_in_cycle = (time_since_start - offset) % interval
            
            # Check if we're in this wave's active period and past the offset
            if time_since_start >= offset and time_in_cycle < duration:
                # For sequential pattern with offsets, check if enough baseline time has passed
                if pattern_type == "custom":
                    # For custom pattern, waves are scheduled with specific offsets
                    # Check if any other wave was recently active
                    other_waves_recently_active = False
                    
                    for j, other_wave in enumerate(wave_patterns):
                        if i == j:  # Skip self
                            continue
                        
                        other_offset = other_wave.get('offset', 0)
                        other_interval = other_wave.get('interval', 300)
                        other_duration = other_wave.get('duration', 30)
                        other_time_in_cycle = (time_since_start - other_offset) % other_interval
                        
                        # Check if other wave ended recently (within baseline_interval)
                        if (time_since_start >= other_offset and 
                            other_duration <= other_time_in_cycle <= (other_duration + baseline_interval)):
                            # Allow sequential waves with proper 60s spacing
                            time_diff = abs((time_since_start - other_offset) % other_interval - 
                                          (time_since_start - offset) % interval)
                            if time_diff < baseline_interval and abs(offset - other_offset) >= baseline_interval:
                                other_waves_recently_active = False
                            else:
                                other_waves_recently_active = True
                                break
                    
                    # Only activate this wave if no conflicting wave is active
                    if not other_waves_recently_active:
                        active_waves.append({
                            'multiplier': multiplier,
                            'name': name,
                            'remaining': duration - time_in_cycle
                        })
                else:
                    # Original logic for non-custom patterns
                    other_waves_recently_active = False
                    
                    for j, other_wave in enumerate(wave_patterns):
                        if i == j:  # Skip self
                            continue
                        
                        other_interval = other_wave.get('interval', 300)
                        other_duration = other_wave.get('duration', 30)
                        other_time_in_cycle = time_since_start % other_interval
                        
                        # Check if other wave ended recently (within baseline_interval)
                        if other_duration <= other_time_in_cycle <= (other_duration + baseline_interval):
                            other_waves_recently_active = True
                            break
                    
                    # Only activate this wave if no other wave was recently active
                    if not other_waves_recently_active:
                        active_waves.append({
                            'multiplier': multiplier,
                            'name': name,
                            'remaining': duration - time_in_cycle
                        })
        
        if active_waves:
            # Use the highest multiplier if multiple waves are somehow still active
            max_wave = max(active_waves, key=lambda x: x['multiplier'])
            return max_wave['multiplier'], True, max_wave['name']
        
        return 1, False, "baseline"
    
    time_counter = 0
    i = 0
    while i < workload_events:
        current_time = offset + time_counter
        in_training_phase = current_time < (offset + seconds_to_train)
        
        # Get current wave multiplier
        current_multiplier, is_wave_active, wave_name = get_current_wave_multiplier(current_time, in_training_phase)
        
        # Update global spike indicator
        is_in_spike = is_wave_active and not in_training_phase
        
        # Log wave transitions
        wave_state_changed = (current_wave_info["active"] != is_wave_active or 
                            current_wave_info["name"] != wave_name or
                            current_wave_info["multiplier"] != current_multiplier)
        
        if debug_logging and wave_state_changed and not in_training_phase:
            current_wave_info = {"active": is_wave_active, "name": wave_name, "multiplier": current_multiplier}
            
            if is_wave_active:
                print("\n" + "="*60)
                print(f"🌊 WAVE STARTING: {wave_name.upper()} at {current_time:.2f}s 🌊")
                print(f"Rate increasing from {rate} req/sec to {rate * current_multiplier} req/sec ({current_multiplier}x)")
                print("="*60 + "\n")
            else:
                print("\n" + "="*60)
                print(f"✓ WAVE ENDING: Returning to baseline at {current_time:.2f}s")
                print(f"Rate returning to {rate} req/sec")
                print(f"Next wave in minimum {baseline_interval}s")
                print("="*60 + "\n")
        
        # Training phase logging
        if debug_logging and in_training_phase and not last_spike_state:
            print("\n" + "="*60)
            print(f"🔬 TRAINING PHASE at {current_time:.2f}s 🔬")
            print(f"Running at baseline rate: {rate} req/sec")
            print("="*60 + "\n")
            last_spike_state = True
        elif debug_logging and not in_training_phase and last_spike_state:
            print("\n" + "="*60)
            print(f"🔬 TRAINING PHASE COMPLETED at {current_time:.2f}s 🔬")
            print(f"Switching to wave pattern mode")
            print("="*60 + "\n")
            last_spike_state = False
        
        # Periodic status logging
        now = time.time()
        if debug_logging and now - last_status_time > status_interval:
            active_threads = len([f for f in futures if not f.done()])
            thread_usage_pct = (active_threads / threads) * 100
            
            if in_training_phase:
                status_prefix = "🔬 TRAINING"
                training_remaining = seconds_to_train - (current_time - offset)
                timing_info = f"Training remaining: {training_remaining:.1f}s"
            else:
                status_prefix = f"🌊 {wave_name.upper()}" if is_wave_active else "📊 BASELINE"
                timing_info = f"Current rate: {rate * current_multiplier:.2f} req/sec ({current_multiplier}x)"
            
            print(f"[{status_prefix}] Time: {current_time:.2f}s, {timing_info}")
            print(f"  → Active threads: {active_threads}/{threads} ({thread_usage_pct:.1f}%)")
            print(f"  → Completed: {i}/{workload_events} requests ({(i/workload_events)*100:.1f}%)")
            
            if active_threads > threads * 0.8:
                print(f"  ⚠️  WARNING: Thread pool usage high ({thread_usage_pct:.1f}%)")
            
            last_status_time = now
        
        # Calculate current rate
        current_rate = rate * current_multiplier
        
        # Schedule the request
        event_time = current_time
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))
        
        # Calculate next request timing
        time_counter += 1.0/current_rate
        i += 1

    # Add final debug message
    if debug_logging:
        print(f"\n[DEBUG] All {workload_events} events scheduled. Execution will now start.")
    
    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f" % (workload_events, error_requests.value, timing_error_requests, avg_latency, workload_events/run_duration_sec))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests,
                "timing_error_number": timing_error_requests,
                "total_request": workload_events,
                "error_request": error_requests,
                "runner_results_file": f"{output_path}/{result_file}.txt"
                }
        run_after_workload(args)

def gradual_increase_runner():
    global start_time, stats, local_latency_stats, runner_parameters

    if 'ingress_service' in runner_parameters.keys():
        srv = runner_parameters['ingress_service']
    else:
        srv = 's0'

    # Get gradual increase parameters
    initial_rate = runner_parameters.get('initial_rate', 1)  # Starting requests per second
    final_rate = runner_parameters.get('final_rate', 10)    # Ending requests per second
    increase_duration = runner_parameters.get('increase_duration', 300)  # Duration to reach final rate (seconds)
    
    if 'minutes_to_train' in runner_parameters.keys():
        minutes_to_train = runner_parameters['minutes_to_train']
    else:
        minutes_to_train = 0
    
    seconds_to_train = minutes_to_train * 60

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    print(f"Gradual Increase Mode:")
    print(f"  Initial rate: {initial_rate} req/sec")
    print(f"  Final rate: {final_rate} req/sec") 
    print(f"  Increase duration: {increase_duration} seconds")
    print(f"  Training phase: {minutes_to_train} minutes")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event = {'service': srv, 'time': 0}
    
    offset = 10
    time_counter = 0
    i = 0
    
    def get_current_rate(current_time):
        """Calculate current rate based on gradual increase"""
        if current_time < (offset + seconds_to_train):
            # Training phase - use initial rate
            return initial_rate
        
        # Calculate time since training ended
        time_since_training = current_time - (offset + seconds_to_train)
        
        if time_since_training >= increase_duration:
            # After increase duration - use final rate
            return final_rate
        
        # During increase phase - linear interpolation
        progress = time_since_training / increase_duration
        current_rate = initial_rate + (final_rate - initial_rate) * progress
        return current_rate
    
    last_logged_rate = 0
    last_log_time = 0
    log_interval = 30  # Log every 30 seconds
    
    while i < workload_events:
        current_time = offset + time_counter
        current_rate = get_current_rate(current_time)
        
        # Log rate changes periodically
        if current_time - last_log_time >= log_interval:
            if abs(current_rate - last_logged_rate) > 0.1:  # Only log if rate changed significantly
                phase = "training" if current_time < (offset + seconds_to_train) else "gradual_increase"
                print(f"[{phase.upper()}] Time: {current_time:.1f}s, Current rate: {current_rate:.2f} req/sec")
                last_logged_rate = current_rate
            last_log_time = current_time
        
        # Schedule the request
        event_time = current_time
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))
        
        # Calculate next request timing based on current rate
        time_counter += 1.0/current_rate
        i += 1

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f" % (workload_events, error_requests.value, timing_error_requests, avg_latency, 1.0*workload_events/run_duration_sec))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests,
                "timing_error_number": timing_error_requests,
                "total_request": workload_events,
                "error_request": error_requests,
                "runner_results_file": f"{output_path}/{result_file}.txt"
                }
        run_after_workload(args)

def gradual_decrease_runner():
    global start_time, stats, local_latency_stats, runner_parameters

    if 'ingress_service' in runner_parameters.keys():
        srv = runner_parameters['ingress_service']
    else:
        srv = 's0'

    # Get gradual decrease parameters
    initial_rate = runner_parameters.get('initial_rate', 10)  # Starting requests per second (high)
    final_rate = runner_parameters.get('final_rate', 1)      # Ending requests per second (low)
    decrease_duration = runner_parameters.get('decrease_duration', 300)  # Duration to reach final rate (seconds)
    
    if 'minutes_to_train' in runner_parameters.keys():
        minutes_to_train = runner_parameters['minutes_to_train']
    else:
        minutes_to_train = 0
    
    seconds_to_train = minutes_to_train * 60

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    print(f"Gradual Decrease Mode:")
    print(f"  Initial rate: {initial_rate} req/sec")
    print(f"  Final rate: {final_rate} req/sec") 
    print(f"  Decrease duration: {decrease_duration} seconds")
    print(f"  Training phase: {minutes_to_train} minutes")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event = {'service': srv, 'time': 0}
    
    offset = 10
    time_counter = 0
    i = 0
    
    def get_current_rate(current_time):
        """Calculate current rate based on gradual decrease"""
        if current_time < (offset + seconds_to_train):
            # Training phase - use initial rate
            return initial_rate
        
        # Calculate time since training ended
        time_since_training = current_time - (offset + seconds_to_train)
        
        if time_since_training >= decrease_duration:
            # After decrease duration - use final rate
            return final_rate
        
        # During decrease phase - linear interpolation (decreasing)
        progress = time_since_training / decrease_duration
        current_rate = initial_rate - (initial_rate - final_rate) * progress
        return current_rate
    
    last_logged_rate = 0
    last_log_time = 0
    log_interval = 30  # Log every 30 seconds
    
    while i < workload_events:
        current_time = offset + time_counter
        current_rate = get_current_rate(current_time)
        
        # Log rate changes periodically
        if current_time - last_log_time >= log_interval:
            if abs(current_rate - last_logged_rate) > 0.1:  # Only log if rate changed significantly
                phase = "training" if current_time < (offset + seconds_to_train) else "gradual_decrease"
                print(f"[{phase.upper()}] Time: {current_time:.1f}s, Current rate: {current_rate:.2f} req/sec")
                last_logged_rate = current_rate
            last_log_time = current_time
        
        # Schedule the request
        event_time = current_time
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))
        
        # Calculate next request timing based on current rate
        time_counter += 1.0/current_rate
        i += 1

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Request rate (req/sec) %.6f" % (workload_events, error_requests.value, timing_error_requests, avg_latency, 1.0*workload_events/run_duration_sec))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests,
                "timing_error_number": timing_error_requests,
                "total_request": workload_events,
                "error_request": error_requests,
                "runner_results_file": f"{output_path}/{result_file}.txt"
                }
        run_after_workload(args)

def constant_runner():
    global start_time, stats, local_latency_stats, runner_parameters

    if 'ingress_service' in runner_parameters.keys():
        srv = runner_parameters['ingress_service']
    else:
        srv = 's0'

    # Get constant rate parameter
    rate = runner_parameters.get('rate', 1)  # Default to 1 req/sec
    
    if 'minutes_to_train' in runner_parameters.keys():
        minutes_to_train = runner_parameters['minutes_to_train']
    else:
        minutes_to_train = 0
    
    seconds_to_train = minutes_to_train * 60

    stats = list()
    print("###############################################")
    print("############   Run Forrest Run!!   ############")
    print("###############################################")
    print(f"Constant Rate Mode:")
    print(f"  Rate: {rate} req/sec (constant)")
    print(f"  Training phase: {minutes_to_train} minutes")
    print(f"  Total events: {workload_events}")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event = {'service': srv, 'time': 0}
    
    offset = 10
    time_counter = 0
    i = 0
    
    # Simple constant rate calculation
    request_interval = 1.0 / rate  # Time between requests in seconds
    
    last_log_time = 0
    log_interval = 60  # Log every 60 seconds
    
    while i < workload_events:
        current_time = offset + time_counter
        
        # Log progress periodically
        if current_time - last_log_time >= log_interval:
            progress_pct = (i / workload_events) * 100
            print(f"[CONSTANT] Time: {current_time:.1f}s, Rate: {rate} req/sec, Progress: {i}/{workload_events} ({progress_pct:.1f}%)")
            last_log_time = current_time
        
        # Schedule the request
        event_time = current_time
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))
        
        # Move to next request time (constant interval)
        time_counter += request_interval
        i += 1

    start_time = time.time()
    print("Start Time:", datetime.now().strftime("%H:%M:%S.%f - %g/%m/%Y"))
    s.run()

    wait(futures)
    run_duration_sec = time.time() - start_time
    avg_latency = 1.0*sum(local_latency_stats)/len(local_latency_stats)

    print("###############################################")
    print("###########   Stop Forrest Stop!!   ###########")
    print("###############################################")
    
    actual_rate = workload_events / run_duration_sec
    print("Run Duration (sec): %.6f" % run_duration_sec, "Total Requests: %d - Error Request: %d - Timing Error Requests: %d - Average Latency (ms): %.6f - Actual Request rate (req/sec) %.6f" % (workload_events, error_requests.value, timing_error_requests, avg_latency, actual_rate))

    if run_after_workload is not None:
        args = {"run_duration_sec": run_duration_sec,
                "last_print_time_ms": last_print_time_ms,
                "requests_processed": processed_requests,
                "timing_error_number": timing_error_requests,
                "total_request": workload_events,
                "error_request": error_requests,
                "runner_results_file": f"{output_path}/{result_file}.txt"
                }
        run_after_workload(args)

### Main

RUNNER_PATH = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-file', action='store', dest='parameters_file',
                    help='The Runner Parameters file', default=f'{RUNNER_PATH}/RunnerParameters.json')

argcomplete.autocomplete(parser)

try:
    args = parser.parse_args()
except ImportError:
    print("Import error, there are missing dependencies to install.  'apt-get install python3-argcomplete "
          "&& activate-global-python-argcomplete3' may solve")
except AttributeError:
    parser.print_help()
except Exception as err:
    print("Error:", err)

parameters_file_path = args.parameters_file

last_print_time_ms = 0
run_after_workload = None
timing_error_requests = 0
processed_requests = Counter()
is_in_spike = False  # Global flag to track if we're currently in a spike period
error_requests = Counter()
pending_requests = Counter()



try:
    with open(parameters_file_path) as f:
        params = json.load(f)
    runner_parameters = params['RunnerParameters']
    runner_type = runner_parameters['workload_type'] # {workload (default), greedy}
    workload_events = runner_parameters['workload_events'] # n. request for greedy
    ms_access_gateway = runner_parameters["ms_access_gateway"] # nginx access gateway ip
    workloads = runner_parameters["workload_files_path_list"] 
    threads = runner_parameters["thread_pool_size"] # n. parallel threads
    round = runner_parameters["workload_rounds"]  # number of repetition rounds
    result_file = runner_parameters["result_file"]  # number of repetition rounds
    if "OutputPath" in params.keys() and len(params["OutputPath"]) > 0:
        output_path = params["OutputPath"]
        if output_path.endswith("/"):
            output_path = output_path[:-1]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = RUNNER_PATH
    if "AfterWorkloadFunction" in params.keys() and len(params["AfterWorkloadFunction"]) > 0:
        sys.path.append(params["AfterWorkloadFunction"]["file_path"])
        run_after_workload = getattr(importlib.import_module(params["AfterWorkloadFunction"]["file_path"].split("/")[-1]),
                                     params["AfterWorkloadFunction"]["function_name"])

except Exception as err:
    print("ERROR: in Runner Parameters,", err)
    exit(1)


## Check if "workloads" is a directory path, if so take all the workload files inside it
if os.path.isdir(workloads[0]):
    dir_workloads = workloads[0]
    workloads = list()
    src_files = os.listdir(dir_workloads)
    for file_name in src_files:
        full_file_name = os.path.join(dir_workloads, file_name)
        if os.path.isfile(full_file_name):
            workloads.append(full_file_name)


stats = list()
local_latency_stats = list()
start_time = 0.0

if runner_type=="greedy":
    greedy_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))

elif runner_type=="periodic": 
    periodic_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))
elif runner_type=="gradual_increase": 
    gradual_increase_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))
elif runner_type=="gradual_decrease": 
    gradual_decrease_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))
elif runner_type=="constant": 
    constant_runner()
    with open(f"{output_path}/{result_file}.txt", "w") as f:
        f.writelines("\n".join(stats))
else:
    # default runner is "file" type
    for cnt, workload_var in enumerate(workloads):
        for x in range(round):
            print("Round: %d -- workload: %s" % (x+1, workload_var))
            processed_requests.value = 0
            timing_error_requests = 0
            error_requests.value = 0
            file_runner(workload_var)
            print("***************************************")
        if cnt != len(workloads) - 1:
            print("Sleep for 100 sec to allow completion of previus requests")
            time.sleep(100)
        with open(f"{output_path}/{result_file}_{workload_var.split('/')[-1].split('.')[0]}.txt", "w") as f:
            f.writelines("\n".join(stats))

