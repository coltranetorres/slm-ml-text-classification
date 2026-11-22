"""Manual smoke test: real Gemma on a handful of tickets (offline, from the HF cache)."""

import time

from slmbench.zeroshot import ZeroShotClassifier

LABELS = ["Billing and Payments", "Technical Support", "Human Resources"]
TICKETS = [
    "I was charged twice for my subscription this month, please refund one payment.",
    "The application crashes every time I open the reports page.",
    "I need to update my parental leave paperwork.",
    "",
    "Rückerstattung für doppelte Abbuchung 😀",
]

clf = ZeroShotClassifier(LABELS)
for t in TICKETS:
    s = time.perf_counter()
    pred = clf.predict([t])[0]
    print(f"{time.perf_counter() - s:.2f}s  {pred!r}  <- {t[:50]!r}")
