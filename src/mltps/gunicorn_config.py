bind = "0.0.0.0:5000"
workers = 1
worker_class = "sync"
timeout = 300
keepalive = 5
worker_connections = 100
max_requests = 500
max_requests_jitter = 50
preload_app = False

# Memory management
memory_limit = 1024 * 1024 * 1024 * 1.5  # 1.5GB limit
worker_tmp_dir = "/dev/shm"  # Use shared memory for temporary files

def post_worker_init(worker):
    """This hook runs after a worker is initialized"""
    worker.log.info("✅ Worker initialized")

def worker_int(worker):
    """Handle worker interruption gracefully"""
    worker.log.info("🛑 Worker interrupted, cleaning up...")