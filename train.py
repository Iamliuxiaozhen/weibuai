import os
import sys
from modelscope.hub.snapshot_download import snapshot_download


def download_qwen_model():
    # Enforce the system to use the local mainland China domain
    os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"

    # Specify the target ModelScope repository ID for Qwen2.5-3B-Instruct
    model_id = "qwen/Qwen2.5-3B-Instruct"

    # Define the local directory path matching your training script settings
    local_dir = "./qwen3-3b"

    print("=== [Initialization] ===")
    print(f"Target Model Repository: {model_id}")
    print(f"Connection Endpoint    : ModelScope Mainland Mirror")
    print(f"Local Storage Path     : {os.path.abspath(local_dir)}")
    print("Fetching files... Please do not close the terminal.\n")

    try:
        # Execute the snapshot download
        # Ignore non-essential model formats to save disk space and time
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=local_dir,
            ignore_file_patterns=[r'\.msgpack$', r'\.h5$', r'\.ot$']
        )
        print("\n=== [Download Completed Successfully] ===")
        print("All target weights are perfectly structured at:")
        print(f"-> {model_dir}")

    except Exception as e:
        print("\n=== [Download Task Failed] ===", file=sys.stderr)
        print(f"Error Diagnostic: {str(e)}", file=sys.stderr)
        print("Suggestion: Please check your disk storage quota or token permissions.", file=sys.stderr)


if __name__ == "__main__":
    download_qwen_model()