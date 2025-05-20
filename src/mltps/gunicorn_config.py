import threading
from app import prediction_loop

def post_worker_init(worker):
    """
    This hook runs after a worker is initialized but before it starts processing requests.
    We use it to start our prediction thread.
    """
    prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
    prediction_thread.start()