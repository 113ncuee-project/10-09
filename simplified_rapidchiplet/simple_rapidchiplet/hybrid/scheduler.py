"""Deterministic dependency/resource calendar scheduling for layer pipelines.

Transfers reserve their directed route and endpoint resources for their modeled
service interval. This is a conservative analytical model, not packet/VC timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import bisect
import heapq
import math
from functools import cached_property


@dataclass(frozen=True)
class Job:
    id: str
    dependencies: tuple[str, ...]
    resources: tuple[str, ...]
    duration_s: float
    kind: str = "compute"
    energy_j: float = 0.0
    priority: tuple = ()
    metadata: dict | None = None

    def __post_init__(self):
        if self.duration_s < 0 or not math.isfinite(self.duration_s):
            raise ValueError("job duration must be finite and nonnegative")
        if self.energy_j < 0 or not math.isfinite(self.energy_j):
            raise ValueError("job energy must be finite and nonnegative")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("a job cannot reserve a resource twice")


@dataclass(frozen=True)
class ScheduledJob:
    job: Job
    start_s: float
    finish_s: float

    def to_dict(self):
        return {**asdict(self.job), "start_s": self.start_s, "finish_s": self.finish_s}


@dataclass(frozen=True)
class Schedule:
    events: tuple[ScheduledJob, ...]
    makespan_s: float
    resource_busy_s: dict[str, float]

    @cached_property
    def event_map(self):
        return {event.job.id: event for event in self.events}

    def validate(self):
        events = self.event_map
        calendars = {}
        for event in self.events:
            if any(events[dep].finish_s > event.start_s + 1e-15 for dep in event.job.dependencies):
                raise ValueError("scheduled job precedes a data dependency")
            for resource in event.job.resources:
                calendars.setdefault(resource, []).append((event.start_s, event.finish_s, event.job.id))
        for resource, spans in calendars.items():
            spans.sort()
            if any(left[1] > right[0] + 1e-15 for left, right in zip(spans, spans[1:])):
                raise ValueError(f"overlapping jobs on {resource}")
        return True


def _earliest_slot(calendars, resources, ready, duration):
    if duration == 0:
        return ready
    cursor = ready
    while True:
        postponed = cursor
        for resource in resources:
            spans = calendars.get(resource, ())
            index = max(0, bisect.bisect_left(spans, (cursor, -math.inf, "")) - 1)
            for index in range(index, len(spans)):
                start, finish, _ = spans[index]
                if start >= cursor + duration - 1e-18:
                    break
                if finish > cursor + 1e-18:
                    postponed = max(postponed, finish)
                    break
        if postponed <= cursor:
            return cursor
        cursor = postponed


def schedule_jobs(jobs, *, policy="priority"):
    """Deterministic list scheduling into resource-calendar gaps.

    ``priority`` uses caller stage/microbatch priority, then dependency release.
    It scales to full DNN traces. ``earliest`` selects globally earliest ready
    jobs and is retained for small hand checks, but can be quadratic for large
    fanout. Neither policy claims an optimal packet or pipeline schedule.
    """
    if policy not in {"priority", "earliest"}:
        raise ValueError("unknown list scheduling policy")
    jobs = tuple(jobs)
    by_id = {job.id: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate job ID")
    if any(dep not in by_id or dep == job.id for job in jobs for dep in job.dependencies):
        raise ValueError("unknown or self-referential job dependency")
    pending_count = {job.id: len(set(job.dependencies)) for job in jobs}
    consumers = {job.id: [] for job in jobs}
    for job in jobs:
        for dependency in set(job.dependencies):
            consumers[dependency].append(job.id)
    calendars, done, events, busy = {}, {}, [], {}
    ready = []
    releases = {}
    def enqueue(name):
        job = by_id[name]
        released = max((done[dependency].finish_s for dependency in job.dependencies), default=0.0)
        releases[name] = released
        if policy == "earliest":
            start = _earliest_slot(calendars, job.resources, released, job.duration_s)
            heapq.heappush(ready, (start, job.priority, name))
        else:
            heapq.heappush(ready, (job.priority, released, name))
    for name, count in pending_count.items():
        if count == 0:
            enqueue(name)
    while ready:
        first, second, name = heapq.heappop(ready)
        job = by_id[name]
        # Adding reservations can only move a ready job later. Heap entries are
        # lower bounds: refresh lazily until the minimum is still current. This
        # gives the same deterministic earliest-ready list schedule without a
        # quadratic scan over thousands of independent multicast jobs.
        actual = _earliest_slot(calendars, job.resources, releases[name], job.duration_s)
        if policy == "earliest" and actual > first + 1e-18:
            heapq.heappush(ready, (actual, second, name))
            continue
        start = actual
        event = ScheduledJob(job, start, start + job.duration_s)
        for resource in job.resources:
            if job.duration_s:
                bisect.insort(calendars.setdefault(resource, []), (event.start_s, event.finish_s, name))
            busy[resource] = busy.get(resource, 0.0) + job.duration_s
        done[name] = event
        events.append(event)
        for consumer in consumers[name]:
            pending_count[consumer] -= 1
            if pending_count[consumer] == 0:
                enqueue(consumer)
    if len(done) != len(jobs):
        raise ValueError("job graph contains a dependency cycle")
    result = Schedule(tuple(events), max((event.finish_s for event in events), default=0.0), busy)
    result.validate()
    return result


def peak_interval_bytes(intervals):
    """Peak simultaneous bytes; release precedes allocation at equal timestamps."""
    changes = []
    for start, finish, amount in intervals:
        if amount < 0 or finish < start:
            raise ValueError("invalid residency interval")
        if finish > start and amount:
            changes.extend(((start, amount), (finish, -amount)))
    live = peak = 0
    for _, delta in sorted(changes, key=lambda event: (event[0], event[1])):
        live += delta
        peak = max(peak, live)
    return peak
