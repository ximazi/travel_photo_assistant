package com.travelphoto.smoke;

public final class ReviewGateTest {
    private static void check(boolean value, String label) {
        if (!value) throw new AssertionError(label);
    }
    public static void main(String[] args) {
        ReviewGate gate = new ReviewGate();
        check(gate.begin(-1, 0, 5000) == null, "no current frame");
        var old = gate.begin(11, 0, 5000);
        check(gate.begin(12, 1, 5000) == null, "single in-flight review");
        check(gate.latch(10, 2) == null, "older frame ignored");
        check(gate.latch(11, 2) == null, "same frame ignored");
        check(!gate.accept(old, 11, 3), "pre-request frame not accepted");
        check(gate.latch(12, 4).equals(old), "new frame latched");
        check(gate.latch(13, 4) == null, "first only");
        gate.invalidate();
        var next = gate.begin(13, 20, 5000);
        check(!gate.accept(old, 12, 21), "old review from invalidated viewport discarded");
        check(!gate.accept(next, 14, 21), "reply without a latched frame discarded");
        check(gate.latch(14, 22).equals(next), "fresh for new generation");
        check(!gate.accept(next, 15, 22), "wrong response frame discarded");
        check(gate.accept(next, 14, 23), "fresh review accepted");
        check(!gate.accept(next, 14, 24), "duplicate reply discarded");
        var slow = gate.begin(14, 30, 5000);
        check(!gate.expire(slow, 5029), "expiry not early");
        check(gate.latch(15, 5030) == null, "no frame after deadline");
        check(!gate.accept(slow, 15, 5030), "deadline is exclusive");
        check(gate.expire(slow, 5030), "expiry clears pending");
        check(gate.begin(16, 5031, 5000) != null, "retry after timeout");
        System.out.println("ReviewGateTest: 18 checks passed");
    }
}

