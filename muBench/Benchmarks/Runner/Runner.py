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
        rate=runner_parameters['rate']
    else:
        rate = 1
    
    if 'ingress_service' in runner_parameters.keys():
        srv=runner_parameters['ingress_service']
    else:
        srv = 's0'

    if 'minutes_to_train' in runner_parameters.keys():
        minutes_to_train = runner_parameters['minutes_to_train']  # minutes to run normal traffic before spikes
    else:
        minutes_to_train = 0
        
    if 'spike_interval' in runner_parameters.keys():
        spike_interval = runner_parameters['spike_interval']  # seconds between spikes
    else:
        spike_interval = 30
    
    if 'spike_multiplier' in runner_parameters.keys():
        spike_multiplier = runner_parameters['spike_multiplier']  # how much to multiply the rate during spike
    else:
        spike_multiplier = 5
        
    if 'spike_duration' in runner_parameters.keys():
        spike_duration = runner_parameters['spike_duration']  # how long spike lasts in seconds
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
    
    print(f"Base rate: {rate} req/sec, Spike rate: {rate * spike_multiplier} req/sec")
    print(f"Spike interval: {spike_interval}s, Spike duration: {spike_duration}s")
    print(f"Thread pool size: {threads}")
    
    s = sched.scheduler(time.time, time.sleep)
    pool = ThreadPoolExecutor(threads)
    futures = list()
    event={'service':srv,'time':0}
    offset=10 # initial delay to allow the insertion of events in the event list
    
    # Track spike status for logging
    last_spike_state = False
    active_threads = 0
    last_status_time = time.time()
    status_interval = 5  # Log status every 5 seconds
    
    time_counter = 0
    i = 0
    while i < workload_events:
        # Determine current time and whether we're still in training phase
        current_time = offset + time_counter
        in_training_phase = current_time < (offset + seconds_to_train)
        
        # Determine if we're in a spike period (only after training phase)
        is_spike_period = False if in_training_phase else ((current_time - seconds_to_train) % spike_interval) < spike_duration
        
        # Log training phase status
        if debug_logging and in_training_phase and last_spike_state != in_training_phase:
            print("\n" + "="*60)
            print(f"🔬 TRAINING PHASE STARTED at {current_time:.2f}s 🔬")
            print(f"Running at baseline rate: {rate} req/sec")
            print(f"Training duration: {minutes_to_train} minutes ({seconds_to_train} seconds)")
            print("="*60 + "\n")
            last_spike_state = in_training_phase
            is_in_spike = False
        
        # Log transition from training to spike pattern
        if debug_logging and not in_training_phase and last_spike_state and not is_spike_period:
            print("\n" + "="*60)
            print(f"🔬 TRAINING PHASE COMPLETED at {current_time:.2f}s 🔬")
            print(f"Switching to spike pattern mode")
            print(f"First spike will start soon")
            print("="*60 + "\n")
            last_spike_state = False
        
        # Log spike transitions with enhanced visibility (after training phase)
        if not in_training_phase and debug_logging and is_spike_period != last_spike_state:
            is_in_spike = is_spike_period  # Update global spike indicator
            if is_spike_period:
                print("\n" + "="*60)
                print(f"🔥 SPIKE STARTING at {current_time:.2f}s 🔥")
                print(f"Rate increasing from {rate} req/sec to {rate * spike_multiplier} req/sec")
                print(f"Expected duration: {spike_duration} seconds")
                print("="*60 + "\n")
            else:
                print("\n" + "="*60)
                print(f"✓ SPIKE ENDING at {current_time:.2f}s")
                print(f"Rate returning to {rate} req/sec")
                print(f"Next spike in {spike_interval - spike_duration} seconds")
                print("="*60 + "\n")
            last_spike_state = is_spike_period
        
        # Enhanced periodic status logging
        now = time.time()
        if debug_logging and now - last_status_time > status_interval:
            active_threads = len([f for f in futures if not f.done()])
            thread_usage_pct = (active_threads / threads) * 100
            
            if in_training_phase:
                status_prefix = "🔬 TRAINING"
                training_remaining = seconds_to_train - (current_time - offset)
                timing_info = f"Training remaining: {training_remaining:.1f}s ({training_remaining/60:.1f} min)"
            else:
                status_prefix = "🔥 SPIKE STATUS" if is_spike_period else "BASELINE STATUS"
                # Add more detailed information about spike timing
                if is_spike_period:
                    spike_time_base = current_time - seconds_to_train
                    time_into_spike = spike_time_base % spike_interval
                    time_remaining = spike_duration - time_into_spike
                    timing_info = f"Spike time remaining: {time_remaining:.1f}s"
                else:
                    spike_time_base = current_time - seconds_to_train
                    time_to_next_spike = spike_interval - (spike_time_base % spike_interval)
                    timing_info = f"Next spike in: {time_to_next_spike:.1f}s"
            
            print(f"[{status_prefix}] Time: {current_time:.2f}s, {timing_info}")
            print(f"  → Active threads: {active_threads}/{threads} ({thread_usage_pct:.1f}%)")
            print(f"  → Completed: {i}/{workload_events} requests ({(i/workload_events)*100:.1f}%)")
            
            if active_threads > threads * 0.8:
                print(f"  ⚠️  WARNING: Thread pool usage high ({thread_usage_pct:.1f}%)")
            
            last_status_time = now
        
        # Determine current rate (only affected by spikes after training phase)
        current_rate = rate * spike_multiplier if (not in_training_phase and is_spike_period) else rate
        
        # Schedule the request
        event_time = current_time
        s.enter(event_time, 1, job_assignment, argument=(pool, futures, event, stats, local_latency_stats))
        
        # Calculate next request timing based on current rate
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
 
