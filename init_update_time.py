import os
from datetime import datetime

def init_update_time():
    """
    Initialize the last_update.txt file with the current time if it doesn't exist.
    """
    timestamp_file = os.path.join(os.getcwd(), "data", "last_update.txt")
    os.makedirs(os.path.dirname(timestamp_file), exist_ok=True)
    
    if not os.path.exists(timestamp_file):
        with open(timestamp_file, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Created {timestamp_file} with current timestamp")
    else:
        print(f"{timestamp_file} already exists")

if __name__ == "__main__":
    init_update_time() 