# Based on https://github.com/BassiDavide/YouTube_Hybrid_Interactions_Analysis
# by D. Bassi, M. J. Maggini, R. Vieira and M. Pereira-Fariña,
# "A Pipeline for the Analysis of User Interactions in YouTube Comments:
# A Hybridization of LLMs and Rule-Based Methods," 2024 11th International
# Conference on Social Networks Analysis, Management and Security (SNAMS),
# Gran Canaria, Spain, 2024, pp. 146-153, doi: 10.1109/SNAMS64316.2024.10883781.

import os
import json
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
QUOTA_SLEEP_SECONDS = int(os.getenv("QUOTA_SLEEP_SECONDS", 3600))
IP_BLOCK_SLEEP_SECONDS = int(os.getenv("IP_BLOCK_SLEEP_SECONDS", 30))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 10))


def build_youtube_client():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def _parse_403_reason(e: HttpError) -> str:
    """Extract the error reason string from a 403 HttpError body."""
    try:
        return json.loads(e.content)["error"]["errors"][0].get("reason", "")
    except Exception:
        return ""


# 403 reasons that mean the daily quota is truly exhausted
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}

# 403 reasons that are permanent for this resource — re-raise so the caller skips it
_SKIP_REASONS = {
    "commentsDisabled",
    "forbidden",
    "accessNotConfigured",
    "insufficientPermissions",
}


def execute_with_retry(request):
    """
    Execute a YouTube API request with retry logic:
    - 403 quota reasons: sleep QUOTA_SLEEP_SECONDS, then retry
    - 403 skip reasons (commentsDisabled, forbidden, …): re-raise immediately
    - 403 unknown: log the reason and re-raise so the caller can decide
    - IP block / persistent errors: sleep IP_BLOCK_SLEEP_SECONDS, prompt VPN rotation, retry indefinitely
    - 5xx transient: exponential backoff up to RETRY_ATTEMPTS times
    """
    transient_attempts = 0

    while True:
        try:
            return request.execute()

        except HttpError as e:
            status = e.resp.status

            if status == 403:
                reason = _parse_403_reason(e)
                if reason in _QUOTA_REASONS:
                    print(f"[API] Quota exceeded (reason={reason}). Sleeping {QUOTA_SLEEP_SECONDS}s before retry...")
                    time.sleep(QUOTA_SLEEP_SECONDS)
                else:
                    # commentsDisabled, forbidden, etc. — not a quota issue, don't sleep
                    print(f"[API] 403 error (reason={reason!r}) — skipping this request")
                    raise

            elif status == 429:
                print(f"[API] Rate limited (429). Sleeping {IP_BLOCK_SLEEP_SECONDS}s. "
                      "Rotate VPN if this persists, then the script will retry automatically.")
                time.sleep(IP_BLOCK_SLEEP_SECONDS)

            elif 500 <= status < 600:
                transient_attempts += 1
                if transient_attempts > RETRY_ATTEMPTS:
                    raise
                wait = 2 ** transient_attempts
                print(f"[API] Server error ({status}). Retrying in {wait}s (attempt {transient_attempts})...")
                time.sleep(wait)

            else:
                raise

        except Exception as e:
            # Catches connection errors, timeouts, etc. — treat as possible IP block
            print(f"[API] Connection error: {e}")
            print(f"[API] Sleeping {IP_BLOCK_SLEEP_SECONDS}s. Rotate VPN if IP-blocked, then the script will continue.")
            time.sleep(IP_BLOCK_SLEEP_SECONDS)
