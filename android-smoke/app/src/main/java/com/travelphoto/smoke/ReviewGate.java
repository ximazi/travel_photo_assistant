package com.travelphoto.smoke;

/** Single-UI-thread local stub: match generation, strictly fresh sequence, and deadline. */
final class ReviewGate {
    record Ticket(long version, long token, long frameFloor, long dueMs) {}
    private long version = 1;
    private long nextToken;
    private Ticket pending;
    private long selectedFrame = -1;

    long version() { return version; }
    void invalidate() { version++; clear(); }
    Ticket begin(long frameFloor, long nowMs, long timeoutMs) {
        if (frameFloor < 0 || pending != null) return null;
        return pending = new Ticket(version, ++nextToken, frameFloor, nowMs + timeoutMs);
    }
    Ticket latch(long frame, long nowMs) {
        if (pending == null || frame <= pending.frameFloor() || nowMs >= pending.dueMs() || selectedFrame >= 0) return null;
        selectedFrame = frame;
        return pending;
    }
    boolean expire(Ticket ticket, long nowMs) {
        if (!isPending(ticket) || nowMs < ticket.dueMs()) return false;
        clear();
        return true;
    }
    boolean accept(Ticket ticket, long frame, long nowMs) {
        if (!isPending(ticket) || frame <= ticket.frameFloor() || frame != selectedFrame || nowMs >= ticket.dueMs()) return false;
        clear();
        return true;
    }
    private void clear() { pending = null; selectedFrame = -1; }
    private boolean isPending(Ticket ticket) {
        return ticket != null && ticket.equals(pending) && ticket.version() == version;
    }
}
