"""A busy loop for occupying one logical processor, for the headroom test.

Takes a duration in seconds so a leaked spinner dies on its own rather than
running until someone notices. Paper 17's first Pi campaign leaked 36 power
samplers that ran for hours and drove the load average to 32; a self-limiting
worker is the cheap way not to repeat that.
"""
import sys
import time

limit = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
end = time.time() + limit
while time.time() < end:
    pass
